"""Tests for ``napari_morans_i._widget``.

Per the napari `testing guidelines
<https://napari.org/dev/plugins/testing_and_publishing/test.html>`_,
the canonical fixture for full integration tests is
``make_napari_viewer_proxy`` (or ``make_napari_viewer``), which spins up a
real Qt-backed :class:`napari.Viewer`.

That fixture, however, requires a working OpenGL context — it eagerly
instantiates ``vispy`` visuals as soon as a layer is added.  To keep the
unit-test suite runnable on **headless CI machines**, we deliberately
substitute :class:`napari.components.ViewerModel`.  The widget never
touches the canvas; it only reads / writes ``viewer.layers`` and listens
to layer events.  ``ViewerModel`` exposes exactly that surface.

If you prefer the canonical ``make_napari_viewer`` fixture (recommended
for local development on a machine with a working Qt + OpenGL stack),
swap the ``viewer`` fixture below for::

    @pytest.fixture
    def viewer(make_napari_viewer):
        return make_napari_viewer()

The Qt application is provided automatically by the ``qtbot`` fixture
from ``pytest-qt``.
"""

from __future__ import annotations

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QObject, Signal

from napari_morans_i._core import MoransResult
from napari_morans_i._widget import MoransIWidget

# --------------------------------------------------------------------------- #
# Synchronous test-double for ``napari.qt.threading.WorkerBase``
# --------------------------------------------------------------------------- #
# The widget ultimately runs the algorithm via ``napari.qt.threading.thread_worker``
# which spins up a real ``QThread``. Real QThreads are non-deterministic on
# heavily-loaded headless CI machines (we have observed multi-minute deadlocks
# when the sandbox is under load) but they are fully exercised inside napari
# itself and we don't need to retest the framework here. To keep our suite
# deterministic and fast we monkey-patch ``thread_worker`` with a passthrough
# that wraps the generator in a ``QObject`` exposing the *same five Qt signals*
# (started / yielded / returned / errored / finished). The widget code under
# test cannot tell the difference — every ``connect()`` call still works, and
# the slot wiring is fully exercised — but everything runs on the main thread.


class _SyncSignalWorker(QObject):
    """Synchronous stand-in for :class:`napari.qt.threading.WorkerBase`."""

    started = Signal()
    yielded = Signal(object)
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, gen):
        super().__init__()
        self._gen = gen
        self._aborted = False

    def start(self):
        self.started.emit()
        try:
            while not self._aborted:
                try:
                    val = next(self._gen)
                except StopIteration as stop:
                    self.returned.emit(stop.value)
                    break
                self.yielded.emit(val)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            self.errored.emit(exc)
        finally:
            self.finished.emit()

    def quit(self):
        self._aborted = True


def _passthrough_thread_worker(func):
    """Decorator that mimics ``thread_worker`` but returns a sync worker."""

    def _factory(*args, **kwargs):
        return _SyncSignalWorker(func(*args, **kwargs))

    return _factory


@pytest.fixture
def sync_worker(monkeypatch):
    """Patch ``napari.qt.threading.thread_worker`` to be synchronous.

    Tests that exercise the full ``Calculate``→layers pipeline use this
    fixture to get rid of QThread non-determinism.
    """
    import napari.qt.threading as nqt

    monkeypatch.setattr(nqt, 'thread_worker', _passthrough_thread_worker)


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def viewer():
    """A pure-Python napari ViewerModel, no GL context required."""
    return ViewerModel()


@pytest.fixture
def widget(viewer, qtbot):
    """Return a ``MoransIWidget`` attached to a fresh ViewerModel."""
    w = MoransIWidget(viewer)
    qtbot.addWidget(w)
    return w


def _add_test_image(viewer, name='test', shape=(20, 20), seed=0):
    """Add a deterministic 2-D image to the viewer; return the layer."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=float)
    img[3:8, 3:8] = 1.0
    img[12:18, 12:18] = -1.0
    img += 0.1 * rng.standard_normal(shape)
    return viewer.add_image(img, name=name)


def _fake_result(shape=(8, 8), sig_level=0.05) -> MoransResult:
    """Build a small MoransResult for testing the on-finished slot."""
    z = np.zeros(shape)
    return MoransResult(
        local_i=np.zeros(shape, dtype=float),
        global_i=0.42,
        z=z,
        lagged=np.zeros(shape, dtype=float),
        p_values=np.zeros(shape, dtype=float),
        clusters=np.zeros(shape, dtype=np.uint8),
        sig_map=np.ones(shape, dtype=np.uint8),
        sig_level=sig_level,
        order=1,
        n_repeats=5,
    )


# --------------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------------- #


class TestUiBuilds:
    def test_widget_constructs(self, widget):
        assert widget.run_btn.isEnabled()
        assert not widget.cancel_btn.isEnabled()
        assert widget.progress.value() == 0
        assert widget.status.text() == 'Ready.'

    def test_default_parameter_values(self, widget):
        assert widget.order_spin.value() == 1
        assert widget.repeat_spin.value() == 200
        assert widget.sig_combo.currentData() == 0.05

    def test_sig_combo_lists_all_supported_levels(self, widget):
        from napari_morans_i._core import VALID_SIG_LEVELS

        actual = [
            widget.sig_combo.itemData(i)
            for i in range(widget.sig_combo.count())
        ]
        assert tuple(actual) == VALID_SIG_LEVELS


# --------------------------------------------------------------------------- #
# Layer combo synchronisation
# --------------------------------------------------------------------------- #


class TestLayerComboSync:
    def test_combo_starts_empty_for_empty_viewer(self, widget):
        assert widget.layer_combo.count() == 0

    def test_combo_updates_when_image_added(self, widget):
        _add_test_image(widget.viewer, name='alpha')
        assert widget.layer_combo.count() == 1
        assert widget.layer_combo.itemText(0) == 'alpha'

    def test_only_image_layers_show(self, widget):
        _add_test_image(widget.viewer, name='img1')
        widget.viewer.add_labels(
            np.zeros((10, 10), dtype=np.int32), name='labels1'
        )
        widget.viewer.add_points(
            np.array([[0.0, 0.0], [1.0, 1.0]]), name='pts'
        )
        names = [
            widget.layer_combo.itemText(i)
            for i in range(widget.layer_combo.count())
        ]
        assert names == ['img1']

    def test_combo_updates_when_layer_removed(self, widget):
        _add_test_image(widget.viewer, name='x')
        _add_test_image(widget.viewer, name='y')
        assert widget.layer_combo.count() == 2
        del widget.viewer.layers['x']
        names = [
            widget.layer_combo.itemText(i)
            for i in range(widget.layer_combo.count())
        ]
        assert names == ['y']

    def test_selection_preserved_across_refresh(self, widget):
        _add_test_image(widget.viewer, name='first')
        _add_test_image(widget.viewer, name='second')
        widget.layer_combo.setCurrentText('second')
        # Adding a new layer triggers a refresh; selection must stick.
        _add_test_image(widget.viewer, name='third')
        assert widget.layer_combo.currentText() == 'second'

    def test_refresh_called_with_no_args(self, widget):
        # Direct call covers the ``event=None`` branch.
        widget._refresh_layer_combo()
        assert widget.layer_combo.count() == 0


# --------------------------------------------------------------------------- #
# Selected-image guard
# --------------------------------------------------------------------------- #


class TestSelectedImageGuard:
    def test_no_layer_pops_warning_returns_none(self, widget, monkeypatch):
        warned = []
        monkeypatch.setattr(
            'qtpy.QtWidgets.QMessageBox.warning',
            lambda *args, **kwargs: warned.append(args),
        )
        assert widget._selected_image() is None
        assert len(warned) == 1

    def test_layer_combo_text_not_in_layers_returns_none(
        self, widget, monkeypatch
    ):
        warned = []
        monkeypatch.setattr(
            'qtpy.QtWidgets.QMessageBox.warning',
            lambda *args, **kwargs: warned.append(args),
        )
        widget.layer_combo.addItem('ghost_layer')
        widget.layer_combo.setCurrentText('ghost_layer')
        assert widget._selected_image() is None
        assert len(warned) == 1

    def test_3d_image_pops_warning_returns_none(self, widget, monkeypatch):
        widget.viewer.add_image(np.zeros((5, 5, 5)), name='vol')
        widget.layer_combo.setCurrentText('vol')
        warned = []
        monkeypatch.setattr(
            'qtpy.QtWidgets.QMessageBox.warning',
            lambda *args, **kwargs: warned.append(args),
        )
        assert widget._selected_image() is None
        assert len(warned) == 1

    def test_2d_image_returns_data(self, widget):
        _add_test_image(widget.viewer, name='ok')
        widget.layer_combo.setCurrentText('ok')
        out = widget._selected_image()
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2


# --------------------------------------------------------------------------- #
# End-to-end calculation through the real thread_worker
# --------------------------------------------------------------------------- #


class TestEndToEndCalculation:
    def test_full_run_adds_three_layers(self, widget, sync_worker):
        """End-to-end: Calculate → algorithm runs → 3 layers added.

        Uses :func:`sync_worker` to swap the QThread-based ``thread_worker``
        for a synchronous Qt-signal worker — same signals, same connections,
        deterministic timing.
        """
        _add_test_image(widget.viewer, name='img')
        widget.layer_combo.setCurrentText('img')
        widget.repeat_spin.setValue(5)  # keep the test fast

        widget._on_calculate()  # synchronous now — returns when worker finishes

        layer_names = [lyr.name for lyr in widget.viewer.layers]
        assert any(n.endswith('_LocalI') for n in layer_names)
        assert any('_Significance' in n for n in layer_names)
        assert any('_Clusters' in n for n in layer_names)
        # Buttons / progress reset.
        assert widget.run_btn.isEnabled()
        assert not widget.cancel_btn.isEnabled()
        assert widget.progress.value() == 0
        assert 'Moran' in widget.status.text()

    def test_no_layer_does_not_start_worker(self, widget, monkeypatch):
        monkeypatch.setattr(
            'qtpy.QtWidgets.QMessageBox.warning',
            lambda *a, **k: None,
        )
        widget._on_calculate()
        assert widget._worker is None
        assert widget.run_btn.isEnabled()

    def test_run_via_button_click(self, widget, sync_worker):
        """Emitting ``run_btn.clicked`` triggers the full pipeline.

        Confirms the QPushButton is wired to ``_on_calculate`` and that the
        downstream worker → layer-creation chain executes. ``clicked.emit()``
        is preferred over ``QPushButton.click()`` / ``qtbot.mouseClick()`` —
        both of those route through the event loop and have been observed to
        hang on heavily-loaded headless CI runners.
        """
        _add_test_image(widget.viewer, name='img')
        widget.layer_combo.setCurrentText('img')
        widget.repeat_spin.setValue(5)

        widget.run_btn.clicked.emit()  # synchronous with sync_worker

        layer_names = [lyr.name for lyr in widget.viewer.layers]
        assert any(n.endswith('_LocalI') for n in layer_names)


# --------------------------------------------------------------------------- #
# Slots — exercised directly with mocks (covers branches without a real run)
# --------------------------------------------------------------------------- #


class TestSlots:
    def test_on_progress_updates_bar(self, widget):
        widget._on_progress(37)
        assert widget.progress.value() == 37

    def test_on_finished_adds_layers_with_correct_names(self, widget):
        _add_test_image(widget.viewer, name='src')
        widget._pending_layer_name = 'src'
        result = _fake_result(sig_level=0.01)
        widget._on_finished(result)
        names = [lyr.name for lyr in widget.viewer.layers]
        assert 'src_LocalI' in names
        assert 'src_Significance_p<0.01' in names
        assert 'src_Clusters_p<0.01' in names
        assert '0.4200' in widget.status.text()

    def test_on_finished_uses_default_layer_name_if_missing(self, widget):
        result = _fake_result()
        widget._on_finished(result)
        names = [lyr.name for lyr in widget.viewer.layers]
        assert any(n.startswith('image') for n in names)

    def test_on_error_pops_message(self, widget, monkeypatch):
        called = []
        monkeypatch.setattr(
            'qtpy.QtWidgets.QMessageBox.critical',
            lambda *a, **k: called.append(a),
        )
        widget._on_error(RuntimeError('boom'))
        assert len(called) == 1
        assert 'boom' in widget.status.text()


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


class TestCancellation:
    def test_cancel_with_no_worker_is_safe(self, widget):
        widget._worker = None
        widget._on_cancel()  # must not raise

    def test_cancel_during_run_quits_the_worker(
        self, widget, sync_worker, qtbot
    ):
        """Verify ``Cancel`` aborts the running worker mid-flight.

        We connect a one-shot slot to ``yielded`` that triggers cancel as
        soon as the worker emits its first progress tick. With the sync
        worker, the next loop iteration sees ``_aborted=True`` and finishes.
        """
        _add_test_image(widget.viewer, name='img', shape=(40, 40))
        widget.layer_combo.setCurrentText('img')
        widget.repeat_spin.setValue(2_000)  # plenty of yields to interrupt

        cancelled = {'done': False}

        def _cancel_on_first_tick(_value):
            if not cancelled['done']:
                cancelled['done'] = True
                widget._on_cancel()

        # Connect *before* _on_calculate so it fires on the very first yield.
        # ``_on_calculate`` creates the worker; we hook in afterwards via a
        # monkey-patched factory.
        original_make = widget._make_worker

        def _patched_make(*args, **kw):
            worker = original_make(*args, **kw)
            worker.yielded.connect(_cancel_on_first_tick)
            return worker

        widget._make_worker = _patched_make

        widget._on_calculate()  # synchronous; returns after cancel completes

        assert cancelled['done']
        assert widget.run_btn.isEnabled()
        assert not widget.cancel_btn.isEnabled()


# --------------------------------------------------------------------------- #
# _set_running toggles
# --------------------------------------------------------------------------- #


class TestSetRunning:
    def test_running_disables_inputs(self, widget):
        widget._set_running(True)
        assert not widget.run_btn.isEnabled()
        assert widget.cancel_btn.isEnabled()
        for w in (
            widget.layer_combo,
            widget.order_spin,
            widget.sig_combo,
            widget.repeat_spin,
        ):
            assert not w.isEnabled()

    def test_idle_re_enables_inputs(self, widget):
        widget._set_running(True)
        widget._set_running(False)
        assert widget.run_btn.isEnabled()
        assert not widget.cancel_btn.isEnabled()
        for w in (
            widget.layer_combo,
            widget.order_spin,
            widget.sig_combo,
            widget.repeat_spin,
        ):
            assert w.isEnabled()
        assert widget.progress.value() == 0


# --------------------------------------------------------------------------- #
# _make_worker inner function body
# --------------------------------------------------------------------------- #
# A sanity check that the inner ``yield from`` inside ``_make_worker`` really
# delegates to ``morans_compute`` and propagates both the progress yields and
# the final return value. We collect emissions from the (synchronous) worker
# directly via Qt-signal callbacks.


class TestMakeWorkerBody:
    def test_inner_body_runs_morans_compute(self, widget, sync_worker):
        rng = np.random.default_rng(0)
        image = rng.normal(size=(16, 16)).astype(np.float32)

        worker = widget._make_worker(
            image, order=1, sig_level=0.05, n_repeats=5
        )

        progress = []
        result_box: list[MoransResult] = []
        worker.yielded.connect(progress.append)
        worker.returned.connect(result_box.append)

        worker.start()

        assert progress == [20, 40, 60, 80, 100]
        assert len(result_box) == 1
        assert isinstance(result_box[0], MoransResult)
        assert result_box[0].local_i.shape == image.shape
