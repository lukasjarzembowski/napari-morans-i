"""Sample data for napari-morans-i.

Provides a small synthetic 2-D image with clear spatial autocorrelation
(two opposing high/low blocks) so a user can quickly try the plugin from
``File -> Open Sample -> Moran's I``.
"""

from __future__ import annotations

import numpy as np

#: Default size of the demo image, kept small so a permutation test runs
#: in seconds even for novice users.
_DEFAULT_SHAPE: tuple[int, int] = (64, 64)


def make_sample_data() -> list[tuple[np.ndarray, dict, str]]:
    """Return a single LayerDataTuple describing a demo image.

    The image is built deterministically (fixed seed) so users get the
    same data every time they open the sample.

    Returns
    -------
    list of LayerDataTuple
        A single ``(data, attributes, layer_type)`` tuple suitable to be
        returned from a napari sample-data command.
    """
    rng = np.random.default_rng(seed=12345)
    img = np.zeros(_DEFAULT_SHAPE, dtype=float)
    # Two roughly-opposite-signed blocks → strong positive global Moran's I.
    img[10:30, 10:30] = 1.0
    img[40:55, 40:55] = -1.0
    img += 0.15 * rng.standard_normal(_DEFAULT_SHAPE)
    return [
        (
            img,
            {'name': "Moran's I demo image", 'colormap': 'gray'},
            'image',
        )
    ]
