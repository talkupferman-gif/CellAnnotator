import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_window_opens():
    from PySide6.QtWidgets import QApplication

    from cellannotator.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "CellAnnotator"
    window.close()
    app.processEvents()
