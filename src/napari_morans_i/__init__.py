"""napari-morans-i: Local Moran's I spatial autocorrelation for napari.

This plugin computes Local Moran's I — a measure of *spatial autocorrelation*
introduced into the imaging context by Dávid et al.
(https://doi.org/10.7554/eLife.89361.1) — directly inside napari.

Public API
----------
- :func:`compute_morans_i`: Synchronous, high-level Moran's I computation.
- :func:`morans_compute`: Generator-based variant that yields progress
  (suitable for use with ``napari.qt.threading.thread_worker``).
- :class:`MoransResult`: Dataclass holding all output arrays.
- :class:`MoransIWidget`: The Qt widget exposed via the napari ``Plugins`` menu.

The MATLAB reference implementation is at
https://github.com/dcsabaCD225/Moran_Matlab/blob/main/moran_local.m and the
Python port here aims to reproduce its outputs to numerical precision while
following modern napari plugin conventions (npe2 manifest, ``thread_worker``
backed long-running tasks, ``qtpy`` for Qt).
"""

# Version is generated at build time by ``setuptools_scm`` and written to
# ``_version.py``. When the package is run from a non-built source tree (e.g.
# a git checkout without ``pip install -e .``), ``_version.py`` is absent and
# we fall back to "unknown".
try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - only triggers in non-built installs
    __version__ = 'unknown'

from ._core import (
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
from ._widget import MoransIWidget

__all__ = [
    '__version__',
    'CLUSTER_HH',
    'CLUSTER_HL',
    'CLUSTER_LH',
    'CLUSTER_LL',
    'CLUSTER_NS',
    'VALID_SIG_LEVELS',
    'MoransIWidget',
    'MoransResult',
    'classify_clusters',
    'compute_morans_i',
    'gaussian_weight_matrix',
    'global_morans_i',
    'local_morans_i',
    'morans_compute',
    'z_normalize',
]
