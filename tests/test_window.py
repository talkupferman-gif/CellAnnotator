import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_window_opens(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    from cellannotator.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "CellAnnotator"
    from PySide6.QtWidgets import QPushButton, QToolButton

    labels = {button.text() for button in window.findChildren(QPushButton)}
    labels.update(button.text() for button in window.findChildren(QToolButton))
    assert {"Draw", "Erase", "Delete", "Redo", "Merge", "Go back"} <= labels
    actions = {action.text() for action in window.findChildren(type(window.account_action))}
    assert {"Open folder", "Sign in", "Commit", "Layer colors", "Run model…"} <= actions
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
