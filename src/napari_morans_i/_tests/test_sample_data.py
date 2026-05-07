"""Tests for ``napari_morans_i._sample_data``."""

from __future__ import annotations

import numpy as np

from napari_morans_i._sample_data import make_sample_data


def test_returns_one_layer_data_tuple():
    layers = make_sample_data()
    assert isinstance(layers, list)
    assert len(layers) == 1
    data, meta, layer_type = layers[0]
    assert layer_type == 'image'
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape == (64, 64)
    assert meta['name']
    assert meta['colormap'] == 'gray'


def test_sample_is_deterministic():
    a = make_sample_data()[0][0]
    b = make_sample_data()[0][0]
    np.testing.assert_array_equal(a, b)


def test_sample_has_block_structure():
    img = make_sample_data()[0][0]
    # Means inside the planted blocks should be far from zero.
    assert img[10:30, 10:30].mean() > 0.5
    assert img[40:55, 40:55].mean() < -0.5
