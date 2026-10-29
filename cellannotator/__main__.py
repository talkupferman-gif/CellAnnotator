import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from cellannotator.icons import icon_path
from cellannotator.ui.main_window import MainWindow


def main() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CellAnnotator.Desktop")
    app = QApplication(sys.argv)
    app.setApplicationName("CellAnnotator")
    icon = QIcon(str(icon_path()))
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
