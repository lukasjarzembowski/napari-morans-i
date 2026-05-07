"""Local Moran's I spatial autocorrelation for 2-D images.

This module is a pure-numpy/scipy port of ``moran_local.m`` from
https://github.com/dcsabaCD225/Moran_Matlab/blob/main/moran_local.m, which
implements the algorithm described in Dávid et al.,
*eLife* (https://doi.org/10.7554/eLife.89361.1).

Algorithm overview
------------------
For an input image :math:`X` of shape ``(M, N)`` the procedure is:

1. **Z-standardise** every pixel: :math:`z_i = (x_i - \\bar x) / \\sigma_x`.
   The MATLAB reference uses *sample* standard deviation (``ddof=1``), which
   is reproduced here for bit-comparable outputs.
2. **Build a spatial weight kernel** ``W`` of shape ``(2*o+1, 2*o+1)`` for an
   order ``o``. The kernel is a Gaussian centred on the middle pixel with
   :math:`\\sigma = (o + 1) / 1.7` (mirroring the MATLAB formula
   ``Gsig = Gx0 / 1.7`` where ``Gx0 = floor(Gm/2)+1``). The centre cell is
   forced to zero so a pixel is never its own neighbour.
3. **Compute the weighted neighbour sum** by 2-D convolution of ``Z`` with
   ``W`` (zero-padded boundary, mode ``'same'``) — exactly what MATLAB's
   ``conv2(Z, W, 'same')`` does. Because Gaussian kernels are symmetric the
   distinction between convolution and correlation is moot.
4. **Normalise by the number of effective neighbours** at each pixel by also
   convolving an all-ones array with ``W``. This automatically handles edge
   pixels which see fewer kernel cells. The resulting array is the *spatial
   lag* :math:`\\tilde z_i`.
5. **Local Moran's I**: :math:`I_i = z_i \\cdot \\tilde z_i`.
6. **Global Moran's I** is the slope of the OLS line fitted through
   :math:`(z_i, \\tilde z_i)`.
7. **Permutation test for pseudo-significance**: the Z-array is randomly
   shuffled ``n_repeats`` times. For each shuffle a new :math:`I_i^*` is
   computed, and pixels for which ``|I_i^*| >= |I_i|`` (preserving sign) are
   counted. The pseudo p-value is ``count / (n_repeats + 1)``.
8. **Cluster classification**: significant pixels are categorised into the
   four LISA quadrants — High-High (HH), Low-Low (LL), Low-High (LH) and
   High-Low (HL).

Sign convention for the permutation test
----------------------------------------
The MATLAB reference distinguishes the two tails:

.. code-block:: matlab

   poz_nagy(kep_I > 0  & kep_I_rnd >= kep_I) = 1;   % positive obs
   neg_kics(kep_I < 0 & kep_I_rnd <= kep_I) = 1;   % negative obs

A high observed :math:`I_i` is "extreme" when the random :math:`I_i^*` is at
least as large; a low observed :math:`I_i` is extreme when the random one is
at least as small. We replicate that exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Cluster code for non-significant pixels.
CLUSTER_NS: int = 0
#: Cluster code for High-High pixels (z > 0 and lag > 0, significant).
CLUSTER_HH: int = 1
#: Cluster code for Low-Low pixels (z < 0 and lag < 0, significant).
CLUSTER_LL: int = 2
#: Cluster code for Low-High pixels (z < 0 and lag > 0, significant outliers).
CLUSTER_LH: int = 3
#: Cluster code for High-Low pixels (z > 0 and lag < 0, significant outliers).
CLUSTER_HL: int = 4

#: Significance levels accepted by the plugin's UI and core API. These match
#: the MATLAB reference exactly: ``szign = [0.05; 0.01; 0.001; 0.0001]``.
VALID_SIG_LEVELS: tuple[float, ...] = (0.05, 0.01, 0.001, 0.0001)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MoransResult:
    """Container for the outputs of a Local Moran's I analysis.

    Attributes
    ----------
    local_i
        2-D array, shape ``(M, N)``. Pixel-wise Local Moran's I value
        :math:`I_i = z_i \\cdot \\tilde z_i`.
    global_i
        Scalar global Moran's I — the OLS slope of :math:`\\tilde z_i` versus
        :math:`z_i`.
    z
        2-D array, the standardised input image.
    lagged
        2-D array, the spatial-lag image
        :math:`\\tilde z_i = (W \\ast z)_i / (W \\ast \\mathbf{1})_i`.
    p_values
        2-D array of pseudo p-values from the permutation test.
    clusters
        2-D ``uint8`` array of cluster codes
        (``0``=NS, ``1``=HH, ``2``=LL, ``3``=LH, ``4``=HL). Suitable for
        display as a napari ``Labels`` layer.
    sig_map
        2-D ``uint8`` array — ``1`` where ``p_values <= sig_level`` else ``0``.
    sig_level
        The significance threshold used to derive ``clusters`` and
        ``sig_map``.
    order
        The Moran's order (kernel half-width) used.
    n_repeats
        The number of permutation iterations used.
    """

    local_i: np.ndarray
    global_i: float
    z: np.ndarray
    lagged: np.ndarray
    p_values: np.ndarray
    clusters: np.ndarray
    sig_map: np.ndarray
    sig_level: float
    order: int
    n_repeats: int


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def z_normalize(image: np.ndarray) -> np.ndarray:
    """Z-standardise an image (mean 0, sample std 1).

    Mirrors MATLAB's ``Zi = (ko - mean(ko)) ./ std(ko, 0, 1)`` which uses the
    *sample* standard deviation (``ddof=1``).

    Parameters
    ----------
    image
        Input image of any shape and any numeric dtype. It is cast to
        ``float64`` internally so the result is always floating-point.

    Returns
    -------
    np.ndarray
        Float array of the same shape as ``image``. If the input has zero
        variance (constant image) an array of zeros is returned to avoid a
        division by zero — Local Moran's I is undefined in that degenerate
        case.
    """
    img = np.asarray(image, dtype=float)
    mean = img.mean()
    # MATLAB's std() defaults to N-1 normalisation, so we use ddof=1.
    std = img.std(ddof=1)
    if std == 0:
        return np.zeros_like(img)
    return (img - mean) / std


def gaussian_weight_matrix(order: int) -> np.ndarray:
    """Build the Gaussian spatial-weights kernel of a given Moran order.

    The kernel is :math:`(2 \\cdot \\text{order} + 1) \\times
    (2 \\cdot \\text{order} + 1)` with values

    .. math::

        W(r, c) = \\exp\\!\\left(-\\frac{(r - c_0)^2 + (c - c_0)^2}
                                    {2 \\sigma^2}\\right),
        \\qquad \\sigma = \\frac{c_0 + 1}{1.7},

    where ``c_0 = order`` is the 0-indexed centre. The centre cell is set to
    zero so the pixel itself is never weighted as its own neighbour. The
    constant ``1.7`` is taken verbatim from the MATLAB reference.

    Parameters
    ----------
    order
        Number of neighbour rings around the centre pixel. Must be ``>= 1``.

    Returns
    -------
    np.ndarray
        2-D float64 weight matrix ``W`` with ``W[order, order] == 0``.

    Raises
    ------
    ValueError
        If ``order < 1``.
    """
    if order < 1:
        raise ValueError(f'order must be >= 1, got {order}')

    size = 1 + 2 * order
    centre = order  # 0-indexed centre row / column
    # MATLAB: Gsig = Gx0 / 1.7 with Gx0 = floor(Gm/2) + 1 = order + 1
    sigma = (order + 1) / 1.7

    # Build coordinate grid relative to centre.
    rr, cc = np.mgrid[0:size, 0:size]
    weights = np.exp(
        -((rr - centre) ** 2 + (cc - centre) ** 2) / (2 * sigma**2)
    )
    weights[centre, centre] = 0.0
    return weights


def local_morans_i(
    z: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Local Moran's I and the spatial-lag image.

    Step-by-step:

    1. ``WZ = conv2(z, weights, mode='same', boundary='fill', fillvalue=0)``
       — the weighted sum of each pixel's neighbourhood.
    2. ``nS = conv2(ones_like(z), weights, ...)`` — the (weighted) number of
       neighbours actually contributing at each pixel; this falls off near
       the image border.
    3. ``lag = WZ / nS`` — normalised spatial lag :math:`\\tilde z_i`.
    4. ``I = z * lag`` — pixel-wise Local Moran's I.

    Parameters
    ----------
    z
        2-D Z-standardised input image (use :func:`z_normalize`).
    weights
        2-D weight kernel (use :func:`gaussian_weight_matrix`).

    Returns
    -------
    local_i, lagged : tuple of np.ndarray
        Both arrays are floats with the same shape as ``z``.
    """
    # Sum of weighted neighbour values at each pixel.
    weighted_sum = convolve2d(
        z, weights, mode='same', boundary='fill', fillvalue=0
    )
    # Number of effective neighbours at each pixel.
    ones = np.ones_like(z)
    n_neighbours = convolve2d(
        ones, weights, mode='same', boundary='fill', fillvalue=0
    )
    # Guard against division by zero at corners with degenerate kernels.
    safe_n = np.where(n_neighbours == 0, 1.0, n_neighbours)
    lagged = weighted_sum / safe_n
    local_i = z * lagged
    return local_i, lagged


def global_morans_i(z: np.ndarray, lagged: np.ndarray) -> float:
    """Compute the global Moran's I as the OLS slope of ``lagged`` vs ``z``.

    This matches MATLAB's ``Pf = polyfit(xpl, ypl, 1); Pf(1)``. ``xpl`` is
    ``z`` flattened and ``ypl`` is ``lagged`` flattened.

    Parameters
    ----------
    z
        2-D Z-standardised image.
    lagged
        2-D spatial-lag image.

    Returns
    -------
    float
        Global Moran's I. Returns ``0.0`` if the input has zero variance,
        because the slope is then undefined.
    """
    x = z.ravel()
    y = lagged.ravel()
    if np.allclose(x, x[0]):
        # Degenerate: no variance in z, slope is undefined. Match MATLAB's
        # tendency to "do something sensible" by returning 0.
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _permutation_pass(
    z: np.ndarray,
    weights: np.ndarray,
    n_neighbours_safe: np.ndarray,
    local_i: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one shuffle of the permutation test.

    Equivalent of one iteration of the MATLAB ``for p = 1:ism`` loop.

    Returns
    -------
    np.ndarray
        ``uint32`` array of zeros and ones — ``1`` where the permuted
        :math:`I_i^*` was at least as extreme as the observed :math:`I_i`
        (in the same tail).
    """
    flat = z.ravel()
    permuted = rng.permutation(flat).reshape(z.shape)

    weighted_sum = convolve2d(
        permuted, weights, mode='same', boundary='fill', fillvalue=0
    )
    lag_perm = weighted_sum / n_neighbours_safe
    local_i_perm = z * lag_perm

    # Two-tailed condition matching the MATLAB reference exactly.
    pos_extreme = np.logical_and(local_i > 0, local_i_perm >= local_i)
    neg_extreme = np.logical_and(local_i < 0, local_i_perm <= local_i)
    return (pos_extreme | neg_extreme).astype(np.uint32)


def classify_clusters(
    z: np.ndarray,
    lagged: np.ndarray,
    p_values: np.ndarray,
    sig_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``(z, lagged, p_values)`` to LISA cluster codes and a sig mask.

    Reproduces the MATLAB block:

    .. code-block:: matlab

       kl(P_perm <= szign(szi) & xpl > 0 & ypl > 0) = 1;  % HH
       kl(P_perm <= szign(szi) & xpl < 0 & ypl < 0) = 2;  % LL
       kl(P_perm <= szign(szi) & xpl < 0 & ypl > 0) = 3;  % LH
       kl(P_perm <= szign(szi) & xpl > 0 & ypl < 0) = 4;  % HL

    Parameters
    ----------
    z
        2-D Z-standardised image.
    lagged
        2-D spatial-lag image.
    p_values
        2-D array of pseudo p-values.
    sig_level
        Significance threshold; must be one of :data:`VALID_SIG_LEVELS`.

    Returns
    -------
    clusters, sig_map : tuple of np.ndarray
        ``clusters`` is a ``uint8`` cluster-code array (codes
        ``CLUSTER_NS / HH / LL / LH / HL``). ``sig_map`` is a ``uint8`` mask
        with ``1`` where ``p_values <= sig_level``.

    Raises
    ------
    ValueError
        If ``sig_level`` is not one of :data:`VALID_SIG_LEVELS`.
    """
    if sig_level not in VALID_SIG_LEVELS:
        raise ValueError(
            f'sig_level must be one of {VALID_SIG_LEVELS}, got {sig_level}'
        )

    sig = p_values <= sig_level
    clusters = np.zeros(z.shape, dtype=np.uint8)
    clusters[sig & (z > 0) & (lagged > 0)] = CLUSTER_HH
    clusters[sig & (z < 0) & (lagged < 0)] = CLUSTER_LL
    clusters[sig & (z < 0) & (lagged > 0)] = CLUSTER_LH
    clusters[sig & (z > 0) & (lagged < 0)] = CLUSTER_HL

    sig_map = sig.astype(np.uint8)
    return clusters, sig_map


# --------------------------------------------------------------------------- #
# High-level entry points
# --------------------------------------------------------------------------- #


def _validate_inputs(
    image: np.ndarray, order: int, sig_level: float, n_repeats: int
) -> np.ndarray:
    """Validate and normalise inputs for both the sync and async API.

    Returns the image as a float64 ndarray; raises ``ValueError`` for any
    invalid argument.
    """
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(f'image must be 2-D, got shape {img.shape}')
    if img.size == 0:
        raise ValueError('image must be non-empty')
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f'order must be an integer >= 1, got {order!r}')
    if 1 + 2 * order > min(img.shape):
        raise ValueError(
            f'order={order} produces a {1 + 2 * order}x{1 + 2 * order} kernel '
            f'which is larger than the image shape {img.shape}'
        )
    if not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ValueError(
            f'n_repeats must be an integer >= 1, got {n_repeats!r}'
        )
    if sig_level not in VALID_SIG_LEVELS:
        raise ValueError(
            f'sig_level must be one of {VALID_SIG_LEVELS}, got {sig_level!r}'
        )
    return img.astype(float)


def morans_compute(
    image: np.ndarray,
    order: int = 1,
    sig_level: float = 0.05,
    n_repeats: int = 200,
    rng: np.random.Generator | None = None,
) -> Generator[int, None, MoransResult]:
    """Generator-based Local Moran's I that yields progress as percentages.

    Designed to be wrapped by ``napari.qt.threading.thread_worker`` so the
    GUI can update a progress bar without blocking the viewer. The generator
    yields integer values in the range ``0..100`` after every permutation
    iteration and returns a :class:`MoransResult` once finished.

    The synchronous variant :func:`compute_morans_i` simply drains this
    generator.

    Parameters
    ----------
    image
        2-D input image. Cast to ``float64``.
    order
        Moran neighbourhood order. Default ``1`` matches the MATLAB demo.
    sig_level
        Pseudo-significance threshold; one of :data:`VALID_SIG_LEVELS`.
    n_repeats
        Number of permutation iterations. The MATLAB GUI default is ``99``;
        we adopt ``200`` here for better empirical p-values, matching the
        plugin's default. Must be ``>= 1``.
    rng
        Optional :class:`numpy.random.Generator`. If ``None`` a fresh,
        seeded-from-OS generator is used. Pass an explicit ``rng`` for
        reproducibility in tests.

    Yields
    ------
    int
        Progress percentage in ``[0, 100]``.

    Returns
    -------
    MoransResult
        Final result. Captured by ``thread_worker`` and emitted via the
        ``returned`` signal.
    """
    img = _validate_inputs(image, order, sig_level, n_repeats)

    z = z_normalize(img)
    weights = gaussian_weight_matrix(order)
    local_i, lagged = local_morans_i(z, weights)
    g_i = global_morans_i(z, lagged)

    if rng is None:
        rng = np.random.default_rng()

    # Pre-compute the per-pixel neighbour count once. Inside the loop we only
    # need to re-convolve the *shuffled* z, never the all-ones array.
    ones = np.ones_like(z)
    n_neighbours = convolve2d(
        ones, weights, mode='same', boundary='fill', fillvalue=0
    )
    n_neighbours_safe = np.where(n_neighbours == 0, 1.0, n_neighbours)

    counts = np.zeros(z.shape, dtype=np.uint32)
    for i in range(n_repeats):
        counts += _permutation_pass(
            z, weights, n_neighbours_safe, local_i, rng
        )
        # Yield progress as an integer percentage. Always include 100 at the
        # final iteration even with rounding.
        yield int(round(100 * (i + 1) / n_repeats))

    p_values = counts / (n_repeats + 1)
    clusters, sig_map = classify_clusters(z, lagged, p_values, sig_level)

    return MoransResult(
        local_i=local_i,
        global_i=g_i,
        z=z,
        lagged=lagged,
        p_values=p_values,
        clusters=clusters,
        sig_map=sig_map,
        sig_level=sig_level,
        order=order,
        n_repeats=n_repeats,
    )


def compute_morans_i(
    image: np.ndarray,
    order: int = 1,
    sig_level: float = 0.05,
    n_repeats: int = 200,
    rng: np.random.Generator | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> MoransResult:
    """Synchronous Local Moran's I — drains :func:`morans_compute`.

    Convenience wrapper for non-GUI use, e.g. from a Jupyter notebook or
    inside a unit test. If you pass a ``progress_callback`` it will be
    invoked once per permutation iteration with the current percentage.

    Parameters
    ----------
    image, order, sig_level, n_repeats, rng
        See :func:`morans_compute`.
    progress_callback
        Optional callable ``f(percent: int) -> None``.

    Returns
    -------
    MoransResult
        The full analysis output.
    """
    gen = morans_compute(image, order, sig_level, n_repeats, rng=rng)
    while True:
        try:
            percent = next(gen)
        except StopIteration as stop:
            return stop.value
        if progress_callback is not None:
            progress_callback(percent)
