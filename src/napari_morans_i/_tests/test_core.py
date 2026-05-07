"""Tests for ``napari_morans_i._core``.

These tests have no dependency on Qt or napari, so they run in any
environment with numpy/scipy/pytest installed. Together with
``test_widget.py`` they should give 100% coverage of the package.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_morans_i import _core
from napari_morans_i._core import (
    CLUSTER_HH,
    CLUSTER_HL,
    CLUSTER_LH,
    CLUSTER_LL,
    CLUSTER_NS,
    VALID_SIG_LEVELS,
    MoransResult,
    classify_clusters,
    compute_morans_i,
    gaussian_weight_matrix,
    global_morans_i,
    local_morans_i,
    morans_compute,
    z_normalize,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _blocky_image(shape=(30, 30), seed: int = 42) -> np.ndarray:
    """Build a deterministic image with two opposite-signed blocks."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=float)
    img[5:15, 5:15] = 1.0
    img[20:28, 20:28] = -1.0
    img += 0.1 * rng.standard_normal(shape)
    return img


# --------------------------------------------------------------------------- #
# z_normalize
# --------------------------------------------------------------------------- #


class TestZNormalize:
    def test_basic_properties(self):
        img = np.arange(100, dtype=float).reshape(10, 10)
        z = z_normalize(img)
        assert z.shape == img.shape
        # mean should be very close to zero
        assert abs(z.mean()) < 1e-10
        # ddof=1 (sample std) should be 1
        assert abs(z.std(ddof=1) - 1.0) < 1e-10

    def test_constant_image_returns_zeros(self):
        img = np.full((5, 5), 7.0)
        z = z_normalize(img)
        assert np.array_equal(z, np.zeros((5, 5)))

    def test_integer_input_is_floated(self):
        img = np.arange(9).reshape(3, 3)
        z = z_normalize(img)
        assert z.dtype.kind == 'f'


# --------------------------------------------------------------------------- #
# gaussian_weight_matrix
# --------------------------------------------------------------------------- #


class TestGaussianWeightMatrix:
    def test_shape_for_each_order(self):
        for o in range(1, 6):
            W = gaussian_weight_matrix(o)
            assert W.shape == (1 + 2 * o, 1 + 2 * o)

    def test_centre_is_zero(self):
        W = gaussian_weight_matrix(3)
        assert W[3, 3] == 0.0

    def test_is_symmetric(self):
        W = gaussian_weight_matrix(2)
        np.testing.assert_allclose(W, W.T)
        np.testing.assert_allclose(W, W[::-1, ::-1])

    def test_max_off_centre_is_at_first_neighbour(self):
        W = gaussian_weight_matrix(2)
        # The four direct neighbours of the centre have the largest
        # weight after the (zero) centre cell.
        order = 2
        cardinal = [
            W[order - 1, order],
            W[order + 1, order],
            W[order, order - 1],
            W[order, order + 1],
        ]
        # All four equal (4-fold symmetry).
        assert all(abs(c - cardinal[0]) < 1e-12 for c in cardinal)
        # And those neighbours are the largest non-zero weights.
        flat = W.flatten()
        flat = flat[flat > 0]
        assert max(cardinal) == pytest.approx(flat.max())

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            gaussian_weight_matrix(0)
        with pytest.raises(ValueError):
            gaussian_weight_matrix(-2)


# --------------------------------------------------------------------------- #
# local_morans_i
# --------------------------------------------------------------------------- #


class TestLocalMoransI:
    def test_shape_matches_input(self):
        z = np.random.default_rng(0).standard_normal((20, 25))
        W = gaussian_weight_matrix(1)
        local, lag = local_morans_i(z, W)
        assert local.shape == z.shape
        assert lag.shape == z.shape

    def test_zero_input_yields_zero(self):
        z = np.zeros((10, 10))
        W = gaussian_weight_matrix(1)
        local, lag = local_morans_i(z, W)
        assert np.allclose(local, 0)
        assert np.allclose(lag, 0)

    def test_positive_correlation_for_blocky_input(self):
        img = _blocky_image()
        z = z_normalize(img)
        W = gaussian_weight_matrix(1)
        local_i, _ = local_morans_i(z, W)
        # Most pixels inside the blocks should have the same sign as their
        # neighbourhood, hence z * lag > 0 on the interior.
        interior_pos = local_i[6:14, 6:14]
        assert (interior_pos > 0).mean() > 0.9


# --------------------------------------------------------------------------- #
# global_morans_i
# --------------------------------------------------------------------------- #


class TestGlobalMoransI:
    def test_strongly_positive_for_blocky_image(self):
        img = _blocky_image()
        z = z_normalize(img)
        W = gaussian_weight_matrix(1)
        _, lag = local_morans_i(z, W)
        gi = global_morans_i(z, lag)
        assert gi > 0.5

    def test_returns_python_float(self):
        z = np.random.default_rng(0).standard_normal((10, 10))
        W = gaussian_weight_matrix(1)
        _, lag = local_morans_i(z, W)
        gi = global_morans_i(z, lag)
        assert isinstance(gi, float)

    def test_zero_variance_short_circuits(self):
        z = np.zeros((10, 10))
        lag = np.zeros((10, 10))
        assert global_morans_i(z, lag) == 0.0


# --------------------------------------------------------------------------- #
# classify_clusters
# --------------------------------------------------------------------------- #


class TestClassifyClusters:
    def test_only_significant_pixels_are_labelled(self):
        z = np.array([[1.0, -1.0], [1.0, -1.0]])
        lag = np.array([[1.0, -1.0], [-1.0, 1.0]])
        # Only top-left and top-right are below threshold.
        p = np.array([[0.01, 0.01], [0.5, 0.5]])
        clusters, sig = classify_clusters(z, lag, p, sig_level=0.05)
        # Top-left: z>0, lag>0 -> HH.  Top-right: z<0, lag<0 -> LL.
        assert clusters[0, 0] == CLUSTER_HH
        assert clusters[0, 1] == CLUSTER_LL
        assert clusters[1, 0] == CLUSTER_NS
        assert clusters[1, 1] == CLUSTER_NS
        assert (sig == np.array([[1, 1], [0, 0]], dtype=np.uint8)).all()

    def test_all_four_quadrants(self):
        z = np.array([[1.0, -1.0], [-1.0, 1.0]])
        lag = np.array([[1.0, -1.0], [1.0, -1.0]])
        p = np.full((2, 2), 0.001)
        clusters, _ = classify_clusters(z, lag, p, sig_level=0.01)
        # (0,0): z>0, lag>0 -> HH
        # (0,1): z<0, lag<0 -> LL
        # (1,0): z<0, lag>0 -> LH
        # (1,1): z>0, lag<0 -> HL
        assert clusters[0, 0] == CLUSTER_HH
        assert clusters[0, 1] == CLUSTER_LL
        assert clusters[1, 0] == CLUSTER_LH
        assert clusters[1, 1] == CLUSTER_HL

    def test_invalid_sig_level_raises(self):
        z = np.zeros((2, 2))
        lag = np.zeros((2, 2))
        p = np.zeros((2, 2))
        with pytest.raises(ValueError):
            classify_clusters(z, lag, p, sig_level=0.1)


# --------------------------------------------------------------------------- #
# Permutation pass
# --------------------------------------------------------------------------- #


class TestPermutationPass:
    def test_pass_returns_uint32_mask(self):
        z = np.random.default_rng(0).standard_normal((10, 10))
        W = gaussian_weight_matrix(1)
        local_i, _ = local_morans_i(z, W)
        from scipy.signal import convolve2d

        n_neigh = convolve2d(
            np.ones_like(z), W, mode='same', boundary='fill', fillvalue=0
        )
        n_neigh = np.where(n_neigh == 0, 1.0, n_neigh)
        out = _core._permutation_pass(
            z, W, n_neigh, local_i, np.random.default_rng(1)
        )
        assert out.shape == z.shape
        assert out.dtype == np.uint32
        # Mask has only 0s and 1s.
        assert set(np.unique(out)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# morans_compute / compute_morans_i
# --------------------------------------------------------------------------- #


class TestMoransCompute:
    def test_generator_yields_progress_then_returns(self):
        img = _blocky_image()
        gen = morans_compute(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=5,
            rng=np.random.default_rng(0),
        )
        progress = list(gen)
        assert progress == [20, 40, 60, 80, 100]
        # The return value of a generator is captured in StopIteration.value
        # by the consumer; we get it back in compute_morans_i below.

    def test_compute_returns_filled_result(self):
        img = _blocky_image()
        result = compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=10,
            rng=np.random.default_rng(0),
        )
        assert isinstance(result, MoransResult)
        assert result.local_i.shape == img.shape
        assert result.lagged.shape == img.shape
        assert result.z.shape == img.shape
        assert result.p_values.shape == img.shape
        assert result.clusters.shape == img.shape
        assert result.sig_map.shape == img.shape
        assert result.clusters.dtype == np.uint8
        assert result.sig_map.dtype == np.uint8
        assert result.order == 1
        assert result.n_repeats == 10
        assert result.sig_level == 0.05
        assert isinstance(result.global_i, float)

    def test_progress_callback_invoked_each_step(self):
        img = _blocky_image()
        seen = []
        compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=5,
            rng=np.random.default_rng(0),
            progress_callback=seen.append,
        )
        assert seen == [20, 40, 60, 80, 100]

    def test_pseudo_pvalues_are_in_unit_interval(self):
        img = _blocky_image()
        result = compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=20,
            rng=np.random.default_rng(0),
        )
        assert (result.p_values >= 0).all()
        assert (result.p_values <= 1).all()

    def test_default_rng_is_used_when_none(self):
        # Just exercise the branch where rng=None.
        img = _blocky_image((10, 10))
        result = compute_morans_i(img, order=1, sig_level=0.05, n_repeats=2)
        assert result.local_i.shape == img.shape

    def test_blocky_image_global_i_is_strongly_positive(self):
        img = _blocky_image()
        result = compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=10,
            rng=np.random.default_rng(0),
        )
        assert result.global_i > 0.5

    def test_random_image_global_i_is_near_zero(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((40, 40))
        result = compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=10,
            rng=np.random.default_rng(0),
        )
        # Pure noise → no spatial structure → global I close to 0.
        assert abs(result.global_i) < 0.2

    def test_reproducible_with_same_seed(self):
        img = _blocky_image()
        r1 = compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=10,
            rng=np.random.default_rng(123),
        )
        r2 = compute_morans_i(
            img,
            order=1,
            sig_level=0.05,
            n_repeats=10,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(r1.local_i, r2.local_i)
        np.testing.assert_array_equal(r1.p_values, r2.p_values)
        np.testing.assert_array_equal(r1.clusters, r2.clusters)
        np.testing.assert_array_equal(r1.sig_map, r2.sig_map)

    def test_higher_order_runs(self):
        img = _blocky_image()
        result = compute_morans_i(
            img,
            order=3,
            sig_level=0.01,
            n_repeats=5,
            rng=np.random.default_rng(0),
        )
        assert result.order == 3


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


class TestInputValidation:
    @pytest.mark.parametrize(
        'image',
        [
            np.zeros(5),  # 1-D
            np.zeros((3, 3, 3)),  # 3-D
        ],
    )
    def test_rejects_non_2d_images(self, image):
        with pytest.raises(ValueError, match='2-D'):
            compute_morans_i(image, order=1, sig_level=0.05, n_repeats=2)

    def test_rejects_empty_image(self):
        with pytest.raises(ValueError, match='non-empty'):
            compute_morans_i(
                np.zeros((0, 0)), order=1, sig_level=0.05, n_repeats=2
            )

    @pytest.mark.parametrize('order', [0, -1, 1.5, '1'])
    def test_rejects_bad_order(self, order):
        with pytest.raises(ValueError):
            compute_morans_i(
                np.zeros((10, 10)), order=order, sig_level=0.05, n_repeats=2
            )

    def test_rejects_kernel_larger_than_image(self):
        with pytest.raises(ValueError, match='larger than the image'):
            compute_morans_i(
                np.zeros((5, 5)), order=10, sig_level=0.05, n_repeats=2
            )

    @pytest.mark.parametrize('n', [0, -1, 2.5, '200'])
    def test_rejects_bad_n_repeats(self, n):
        with pytest.raises(ValueError):
            compute_morans_i(
                np.zeros((10, 10)), order=1, sig_level=0.05, n_repeats=n
            )

    def test_rejects_unknown_sig_level(self):
        with pytest.raises(ValueError, match='sig_level'):
            compute_morans_i(
                np.zeros((10, 10)), order=1, sig_level=0.1, n_repeats=2
            )

    def test_valid_sig_levels_constant_matches_matlab(self):
        # The MATLAB implementation hard-codes
        # ``szign = [0.05; 0.01; 0.001; 0.0001]``.
        assert VALID_SIG_LEVELS == (0.05, 0.01, 0.001, 0.0001)


# --------------------------------------------------------------------------- #
# Cluster code constants
# --------------------------------------------------------------------------- #


def test_cluster_constants_match_matlab_codes():
    """The MATLAB script uses 0=NS, 1=HH, 2=LL, 3=LH, 4=HL."""
    assert CLUSTER_NS == 0
    assert CLUSTER_HH == 1
    assert CLUSTER_LL == 2
    assert CLUSTER_LH == 3
    assert CLUSTER_HL == 4
