import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_window_opens(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    from cellannotator.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "CellAnnotator"
    from PySide6.QtWidgets import QLabel, QPushButton, QToolButton

    labels = {button.text() for button in window.findChildren(QPushButton)}
    labels.update(button.text() for button in window.findChildren(QToolButton))
    assert {"Draw", "Erase", "Delete", "Redo", "Merge", "Go back", "1", "2", "3", "4", "5"} <= labels
    assert "Quality" in {label.text() for label in window.findChildren(QLabel)}
    actions = {action.text() for action in window.findChildren(type(window.account_action))}
    assert {"Open folder", "Sign in", "Commit", "Layer colors", "Run model…", "Train from rankings…", "Run ranking model"} <= actions
    assert window.pages.currentWidget() is window.home
    assert window.pages.currentWidget() is not window.work
    window.settings.token = "token"
    window.settings.email = "owner@lab.test"

    class _Projects:
        def list_projects(self):
            return [{"id": "p1", "name": "Plates", "status": "open"}]

    window._client = lambda: _Projects()
    window._show_home()
    assert window.pages.currentWidget() is window.home
    assert window.home.gallery.count() == 1
    assert not window.toolbar.isVisible()
    assert window.canvas.masks_visible is True
    assert window.canvas.show_outlines is False
    assert not window.images_dock.isVisible()
    window.close()
    app.processEvents()


def test_drawn_cell_starts_at_grade_five():
    from PySide6.QtWidgets import QApplication

    from cellannotator.segio import MaskSession
    from cellannotator.ui.canvas import Canvas

    QApplication.instance() or QApplication([])
    canvas = Canvas()
    canvas.session = MaskSession.empty(40, 40)
    canvas._stroke = [(5, 5), (24, 5), (24, 24), (5, 24), (6, 6), (7, 7), (8, 8), (5, 5)]
    canvas._close_outline()
    assert canvas.session.quality_of(1) == 5
    assert canvas.session.is_manual(1)


def test_model_masks_start_at_grade_three(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    import numpy as np

    from cellannotator.segio import MaskSession
    from cellannotator.ui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    session = MaskSession.empty(8, 8)
    window.canvas.session = session
    window.current_rel = "a.tif"

    class _Meta:
        links = []

        def drop_slice(self, _relpath):
            return None

    class _Project:
        meta = _Meta()

        def write_working(self, _relpath, _session):
            return None

        def write_working_meta(self):
            return None

    window.project = _Project()
    masks = np.zeros((8, 8), np.int32)
    masks[1:4, 1:4] = 1
    masks[1:4, 5:7] = 2
    window._apply_model("a.tif", masks)
    assert session.quality_of(1) == 3
    assert session.quality_of(2) == 3
    assert not session.is_manual(1)
    session.dirty = False
    window.close()
