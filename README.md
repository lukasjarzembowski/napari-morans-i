# napari-morans-i

[![License BSD-3](https://img.shields.io/pypi/l/napari-morans-i.svg?color=green)](https://github.com/lukasjarzembowski/napari-morans-i/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-morans-i.svg?color=green)](https://pypi.org/project/napari-morans-i)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-morans-i.svg?color=green)](https://python.org)
[![tests](https://github.com/lukasjarzembowski/napari-morans-i/workflows/tests/badge.svg)](https://github.com/lukasjarzembowski/napari-morans-i/actions)
[![codecov](https://codecov.io/gh/lukasjarzembowski/napari-morans-i/branch/main/graph/badge.svg)](https://codecov.io/gh/lukasjarzembowski/napari-morans-i)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-morans-i)](https://napari-hub.org/plugins/napari-morans-i)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A [napari] plugin that computes **Local Moran's I** spatial-autocorrelation maps
of 2-D images, with a permutation-test–based significance map and a categorical
cluster map (HH / LL / HL / LH / NS).

It is a Python port of the MATLAB reference implementation
[`moran_local.m`](https://github.com/dcsabaCD225/Moran_Matlab/blob/main/moran_local.m)
used in:

> Dávid Cs. *et al.*, **"Spatial gene expression patterns in the cortex…"**,
> *eLife* (2024). [doi:10.7554/eLife.89361.1](https://doi.org/10.7554/eLife.89361.1)

This code was entirely created using Claude Opus 4.7 (2026-05-06). Prompts have been archived in the ```prompts.txt``` file.

---

## Features

- **Layer chooser** — operates on any 2-D `Image` layer in the current viewer.
- **Configurable Moran order** — controls the size of the Gaussian
  neighbourhood kernel `(2·order + 1)²`.
- **Four standard significance levels** — 0.05, 0.01, 0.001, 0.0001.
- **Permutation test** with user-defined number of repeats (default *n* = 200).
- **Threaded execution** via `napari.qt.threading.thread_worker`, with a live
  progress bar and a *Cancel* button — the viewer stays responsive throughout.
- **Three output layers** added back to the viewer for inspection:
  | Suffix            | Type   | Meaning                                      |
  | ----------------- | ------ | -------------------------------------------- |
  | `_LocalI`         | Image  | Local Moran's I per pixel (continuous)       |
  | `_Significance`   | Image  | Binary mask: pixels with *p* ≤ sig\_level    |
  | `_Clusters`       | Labels | Categorical cluster code (see legend below)  |

### Cluster legend

| Code | Class | Meaning                                                  |
| ---- | ----- | -------------------------------------------------------- |
| 0    | NS    | Not significant                                          |
| 1    | HH    | High-value pixel surrounded by high-value neighbours     |
| 2    | LL    | Low-value pixel surrounded by low-value neighbours       |
| 3    | LH    | Low-value pixel surrounded by high-value neighbours      |
| 4    | HL    | High-value pixel surrounded by low-value neighbours      |

These codes match the original MATLAB source one-for-one.

---

## Usage

1. Launch napari and open or drag-drop an image.
2. Open **Plugins → Moran's I Analysis**.
3. (Optional) Try the bundled demo via **File → Open Sample → napari-morans-i:
   Moran's I sample (blocks)**.
4. Choose the input image layer, the Moran *order*, the significance level,
   and the number of permutations.
5. Click **Calculate**. Progress is reported live; **Cancel** stops the
   permutation test cleanly.
6. Inspect the three new layers added to the viewer.

---

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-morans-i` via [pip]:

```bash
pip install napari-morans-i
```

If napari is not already installed, you can install `napari-morans-i` with napari and Qt via:

```bash
pip install "napari-morans-i[all]"
```


To install latest development version:

```bash
pip install git+https://github.com/lukasjarzembowski/napari-morans-i.git
```

---

## Algorithm — step-by-step

The implementation lives in `napari_morans_i/_core.py` and reproduces the
MATLAB script step-for-step. Each step has a corresponding pure function so it
is independently testable.

### 1. Z-normalisation (`z_normalize`)

Convert the input image *X* into z-scores using the **sample** standard
deviation (`ddof=1`), to match MATLAB's default `std`:

```
z = (X − mean(X)) / std(X, ddof=1)
```

If the image is constant (`std = 0`) we short-circuit to an all-zeros array.

### 2. Gaussian neighbourhood weights (`gaussian_weight_matrix`)

Build a `(2·order + 1) × (2·order + 1)` Gaussian kernel `W` with
`σ = (order + 1) / 1.7` and **the centre cell zeroed**, exactly as in the
MATLAB reference. Zeroing the centre means each pixel never contributes to
its own neighbourhood average.

### 3. Lagged values (`local_morans_i`)

Two zero-padded 2-D convolutions (`scipy.signal.convolve2d`,
`mode='same'`, `boundary='fill'`):

```
WZ  = conv2(z,         W, 'same')   # weighted sum of neighbours
nS  = conv2(ones_like(z), W, 'same')   # normaliser, edge-correction
lagged = WZ / nS                    # mean of standardised neighbours
local_i = z * lagged                # Local Moran's I per pixel
```

The `nS` normaliser corrects for kernel weight that "falls off the edge" of
the image, which is why we use zero-padding rather than a periodic boundary.

### 4. Global Moran's I (`global_morans_i`)

The global statistic is the **OLS slope** of `lagged` regressed on `z` —
i.e. `cov(z, lagged) / var(z)`. Computed with `numpy.polyfit`. Returns
`0.0` when `z` has zero variance.

### 5. Permutation test (`_permutation_pass`, inside `morans_compute`)

For each repetition `1…n_repeats`:

1. Shuffle `z` to obtain `z_perm`.
2. Recompute the permuted lagged map and `local_i_perm`.
3. **Sign-aware tally**: for each pixel, increment a counter when
   `local_i_perm` lies in the same tail (positive vs. negative) as the
   observed `local_i_obs` and has equal-or-greater magnitude.

After `n_repeats` permutations, the pseudo *p*-value per pixel is
`(count + 1) / (n_repeats + 1)` — the standard `+1` correction that keeps
*p* strictly positive. Sign-aware tallying matches the MATLAB reference
and is **not** equivalent to a two-sided absolute-value test.

### 6. Cluster classification (`classify_clusters`)

Threshold the *p*-values at the user-chosen significance level, then
classify each significant pixel by the signs of `z` and `lagged`:

| `z`  | `lagged` | label | code |
| ---- | -------- | ----- | ---- |
| ≥ 0  | ≥ 0      | HH    | 1    |
| < 0  | < 0      | LL    | 2    |
| < 0  | ≥ 0      | LH    | 3    |
| ≥ 0  | < 0      | HL    | 4    |

Non-significant pixels get code `0` (NS).

### Threading model

The whole pipeline is exposed as a *generator* (`morans_compute`) that yields
integer progress percentages in `[0, 100]`. The widget wraps it in
`@napari.qt.threading.thread_worker`:

```python
@thread_worker
def _run():
    result = yield from morans_compute(image, order=…, sig_level=…, n_repeats=…)
    return result
```

`yield from` automatically forwards every progress integer to the worker's
`yielded` Qt-signal (which drives the progress bar) and forwards the final
return value to the `returned` signal (which adds the output layers).

`Cancel` calls `worker.quit()`, which stops the generator at its next
`yield`, leaving the viewer responsive at all times.

---

## Citation

If you use this plugin in academic work, please cite both the original paper

> Dávid Cs. *et al.* (2024). *eLife*, doi:10.7554/eLife.89361.1

and napari (Sofroniew *et al.*, 2022, doi:10.5281/zenodo.3555620).

[napari]: https://napari.org

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-morans-i" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/lukasjarzembowski/napari-morans-i/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
