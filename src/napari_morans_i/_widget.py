"""Napari Qt widget exposing Local Moran's I to the GUI.

The widget lets the user pick:

- An ``Image`` layer from the current viewer.
- A Moran's order (positive integer).
- A significance level (one of ``0.05, 0.01, 0.001, 0.0001``).
- The number of permutations for the Monte-Carlo significance test
  (default ``200``).

The computation is dispatched to a ``napari.qt.threading.thread_worker`` so
the viewer remains responsive; progress is fed back to a ``QProgressBar``
through the worker's ``yielded`` signal and the result is appended to the
viewer as three new layers (Local I image, significance mask, and a Labels
layer of LISA cluster codes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._core import VALID_SIG_LEVELS, MoransResult, morans_compute

if TYPE_CHECKING:  # pragma: no cover - imports for static type-checkers only
    import napari


class MoransIWidget(QWidget):
    """Dock widget that runs Local Moran's I on a chosen napari image layer.

    Parameters
    ----------
    napari_viewer
        The napari Viewer instance the widget is docked into. Stored as
        ``self.viewer`` and used to enumerate ``Image`` layers and to push
        result layers back.

    Attributes
    ----------
    layer_combo : QComboBox
        Drop-down listing every ``Image`` layer currently in the viewer.
    order_spin : QSpinBox
        Moran's order (>=1).
    sig_combo : QComboBox
        Significance threshold selector.
    repeat_spin : QSpinBox
        Number of permutation iterations (default 200).
    run_btn, cancel_btn : QPushButton
        Run / cancel buttons. Mutually exclusive enable state.
    progress : QProgressBar
        0..100 progress bar driven by the worker's ``yielded`` signal.
    status : QLabel
        Single-line status / result text.
    """

    #: Name used for the Local Moran's I image layer added to the viewer.
    LOCAL_I_LAYER_SUFFIX = '_LocalI'
    #: Name used for the significance map layer added to the viewer.
    SIG_LAYER_SUFFIX = '_Significance'
    #: Name used for the LISA cluster Labels layer.
    CLUSTER_LAYER_SUFFIX = '_Clusters'

    def __init__(self, napari_viewer: napari.viewer.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None  # current thread_worker, if any

        self._build_ui()
        self._connect_viewer_events()
        self._refresh_layer_combo()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        """Construct all child widgets and lay them out."""
        outer = QVBoxLayout()
        outer.setAlignment(Qt.AlignTop)

        form = QFormLayout()

        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip('Pick the 2-D image layer to analyse.')
        form.addRow('Image layer:', self.layer_combo)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 64)
        self.order_spin.setValue(1)
        self.order_spin.setToolTip(
            "Moran's order — half-width of the Gaussian neighbourhood "
            'kernel. Higher values consider farther-away pixels.'
        )
        form.addRow("Moran's order:", self.order_spin)

        self.sig_combo = QComboBox()
        for level in VALID_SIG_LEVELS:
            self.sig_combo.addItem(f'{level:g}', level)
        self.sig_combo.setToolTip(
            'Pseudo-significance threshold for the Monte-Carlo test.'
        )
        form.addRow('Significance level:', self.sig_combo)

        self.repeat_spin = QSpinBox()
        self.repeat_spin.setRange(1, 100_000)
        self.repeat_spin.setValue(200)
        self.repeat_spin.setToolTip(
            'Number of random permutations used to estimate the '
            'pseudo p-value at every pixel.'
        )
        form.addRow('Permutations (n):', self.repeat_spin)

        outer.addLayout(form)

        # Action row.
        button_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Moran's I")
        self.run_btn.clicked.connect(self._on_calculate)
        button_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_row.addWidget(self.cancel_btn)

        outer.addLayout(button_row)

        # Progress + status.
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        self.status = QLabel('Ready.')
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        outer.addStretch()
        self.setLayout(outer)

    def _connect_viewer_events(self) -> None:
        """Subscribe to layer-list events so the combo stays in sync."""
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combo)
        # Re-detect renames as well.
        self.viewer.layers.events.changed.connect(self._refresh_layer_combo)

    # ------------------------------------------------------------------ #
    # Layer combo synchronisation
    # ------------------------------------------------------------------ #

    def _refresh_layer_combo(self, event=None) -> None:  # noqa: ARG002
        """Refill ``self.layer_combo`` with the viewer's Image layers.

        Preserves the previous selection if the named layer still exists.
        """
        # Lazy import — keeps top-of-module light per napari best-practices.
        from napari.layers import Image

        previous = self.layer_combo.currentText()
        # Block signals while rebuilding to avoid a flurry of currentTextChanged.
        was_blocked = self.layer_combo.blockSignals(True)
        try:
            self.layer_combo.clear()
            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    self.layer_combo.addItem(layer.name)
            if previous:
                idx = self.layer_combo.findText(previous)
                if idx >= 0:
                    self.layer_combo.setCurrentIndex(idx)
        finally:
            self.layer_combo.blockSignals(was_blocked)

    # ------------------------------------------------------------------ #
    # Run / cancel
    # ------------------------------------------------------------------ #

    def _selected_image(self) -> np.ndarray | None:
        """Return the data of the currently-selected Image layer or None.

        Side-effect: pops up a ``QMessageBox`` warning if no layer is
        selected or the data is not 2-D.
        """
        name = self.layer_combo.currentText()
        if not name or name not in self.viewer.layers:
            QMessageBox.warning(
                self, 'No image', 'Please select an image layer first.'
            )
            return None
        layer = self.viewer.layers[name]
        data = np.asarray(layer.data)
        if data.ndim != 2:
            QMessageBox.warning(
                self,
                'Wrong dimensionality',
                "Local Moran's I currently requires a 2-D image — got an "
                f'array of shape {data.shape}.',
            )
            return None
        return data

    def _on_calculate(self) -> None:
        """Slot for the *Run Moran's I* button — kicks off the worker."""
        image = self._selected_image()
        if image is None:
            return

        order = self.order_spin.value()
        sig_level = float(self.sig_combo.currentData())
        n_repeats = self.repeat_spin.value()
        layer_name = self.layer_combo.currentText()

        self.status.setText(
            f"Running Moran's I (order={order}, n={n_repeats})..."
        )
        self.progress.setValue(0)
        self._set_running(True)

        # Save metadata for the on-finished slot — the worker's return value
        # itself is just the MoransResult, with no layer-name context.
        self._pending_layer_name = layer_name

        self._worker = self._make_worker(image, order, sig_level, n_repeats)
        self._worker.yielded.connect(self._on_progress)
        self._worker.returned.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        # ``finished`` is always emitted (success / error / cancel) so it's
        # the right place to re-enable the run button.
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _make_worker(
        self,
        image: np.ndarray,
        order: int,
        sig_level: float,
        n_repeats: int,
    ):
        """Build and return a ``napari.qt.threading.WorkerBase``.

        Factored out so unit tests can monkey-patch worker construction.
        """
        # Local import — keeps the module importable in environments without
        # a Qt application available (e.g. building docs).
        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            # ``yield from`` propagates each progress value AND the final
            # return value of ``morans_compute``.
            result = yield from morans_compute(
                image, order=order, sig_level=sig_level, n_repeats=n_repeats
            )
            return result

        return _run()

    def _on_cancel(self) -> None:
        """Slot for the *Cancel* button — requests the worker to stop."""
        if self._worker is not None:
            self._worker.quit()
            self.status.setText('Cancelling...')

    # ------------------------------------------------------------------ #
    # Worker callbacks
    # ------------------------------------------------------------------ #

    def _on_progress(self, percent: int) -> None:
        """Update the progress bar (called from the worker thread)."""
        self.progress.setValue(int(percent))

    def _on_finished(self, result: MoransResult) -> None:
        """Receive the final MoransResult and add layers to the viewer."""
        layer_name = getattr(self, '_pending_layer_name', 'image')

        # Local Moran's I as a continuous-valued image.
        self.viewer.add_image(
            result.local_i,
            name=f'{layer_name}{self.LOCAL_I_LAYER_SUFFIX}',
            colormap='viridis',
        )
        # Binary significance mask (1 = significant).
        self.viewer.add_image(
            result.sig_map,
            name=(
                f'{layer_name}{self.SIG_LAYER_SUFFIX}_p<{result.sig_level:g}'
            ),
            colormap='gray',
            contrast_limits=(0, 1),
        )
        # LISA cluster classification — Labels so napari picks a colourmap.
        self.viewer.add_labels(
            result.clusters,
            name=(
                f'{layer_name}{self.CLUSTER_LAYER_SUFFIX}'
                f'_p<{result.sig_level:g}'
            ),
        )

        n_sig = int(result.sig_map.sum())
        self.status.setText(
            f"Done. Global Moran's I = {result.global_i:.4f}; "
            f'{n_sig:,d}/{result.sig_map.size:,d} pixels significant at '
            f'p≤{result.sig_level:g}.'
        )

    def _on_error(self, exc: BaseException) -> None:
        """Slot wired to the worker's ``errored`` signal."""
        self.status.setText(f'Error: {exc}')
        QMessageBox.critical(self, "Moran's I error", str(exc))

    def _on_worker_finished(self) -> None:
        """Always-fired cleanup — re-enable buttons, clear worker handle."""
        self._set_running(False)
        self._worker = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _set_running(self, running: bool) -> None:
        """Toggle button states for the running / idle modes."""
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        # Disable parameter widgets while running.
        for w in (
            self.layer_combo,
            self.order_spin,
            self.sig_combo,
            self.repeat_spin,
        ):
            w.setEnabled(not running)
        if not running:
            # Reset the bar to 0 a moment after completion so it doesn't
            # linger at "100%". Comment-out if you'd rather it stay at 100.
            self.progress.setValue(0)
