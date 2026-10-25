from __future__ import annotations

import zipfile
from io import BytesIO

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from cellannotator.files import safe_rel


def ask_kind(parent) -> str | None:
    dialog = QDialog(parent)
    dialog.setWindowTitle("What is in this folder?")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Choose how these images relate to each other."))
    unrelated = QRadioButton("Unrelated images. Each file is its own set of cells.")
    stack = QRadioButton("Z-stack. A cell can continue across slices in filename order.")
    unrelated.setChecked(True)
    layout.addWidget(unrelated)
    layout.addWidget(stack)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.Accepted:
        return None
    return "zstack" if stack.isChecked() else "unrelated"


class LoginDialog(QDialog):
    def __init__(self, server_url: str, email: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Account")
        layout = QFormLayout(self)
        self.server = QLineEdit(server_url)
        self.email = QLineEdit(email)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        layout.addRow("Server", self.server)
        layout.addRow("Email", self.email)
        layout.addRow("Password", self.password)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        login = buttons.addButton("Log in", QDialogButtonBox.AcceptRole)
        register = buttons.addButton("Create account", QDialogButtonBox.ActionRole)
        login.clicked.connect(self._login)
        register.clicked.connect(self._register)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.action = "login"

    def _login(self) -> None:
        self.action = "login"
        self.accept()

    def _register(self) -> None:
        self.action = "register"
        self.accept()


class ConflictDialog(QDialog):
    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("These files were both changed")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose which copy to keep for each file. The other files still merge."))
        self.rows: list[tuple[str, QRadioButton, QRadioButton]] = []
        for path in paths:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.addWidget(QLabel(path))
            mine = QRadioButton("Use mine")
            theirs = QRadioButton("Use theirs")
            mine.setChecked(True)
            row_layout.addWidget(mine)
            row_layout.addWidget(theirs)
            layout.addWidget(row)
            self.rows.append((path, mine, theirs))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def resolutions(self) -> dict[str, str]:
        return {path: "mine" if mine.isChecked() else "theirs" for path, mine, _theirs in self.rows}


class HistoryDialog(QDialog):
    def __init__(self, commits: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Commits")
        self.setMinimumSize(520, 360)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Restore loads that commit into your working files. The shared latest stays put."))
        self.list = QListWidget()
        self._ids: list[str] = []
        for commit in reversed(commits):
            self._ids.append(commit["id"])
            self.list.addItem(f"{commit['created_at']}  {commit['author_email']}  {commit['message']}")
        layout.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        restore = buttons.addButton("Restore into working copy", QDialogButtonBox.ActionRole)
        restore.clicked.connect(self._restore)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.reject)
        layout.addWidget(buttons)
        self.chosen: str | None = None

    def _restore(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        self.chosen = self._ids[row]
        self.accept()


class BatchDialog(QDialog):
    def __init__(self, relpaths: list[str], current: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run the model on images")
        self.setMinimumSize(420, 360)
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        for relpath in relpaths:
            item = QListWidgetItem(relpath)
            item.setFlags(item.flags() | Qt_item_flag())
            item.setCheckState(checked if relpath == current else unchecked)
            self.list.addItem(item)
        layout.addWidget(self.list)
        row = QHBoxLayout()
        all_button = QPushButton("All")
        none_button = QPushButton("None")
        all_button.clicked.connect(lambda: self._set_all(True))
        none_button.clicked.connect(lambda: self._set_all(False))
        row.addWidget(all_button)
        row.addWidget(none_button)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, value: bool) -> None:
        state = checked if value else unchecked
        for index in range(self.list.count()):
            self.list.item(index).setCheckState(state)

    def chosen(self) -> list[str]:
        return [
            self.list.item(index).text()
            for index in range(self.list.count())
            if self.list.item(index).checkState() == checked
        ]


def Qt_item_flag():
    from PySide6.QtCore import Qt

    return Qt.ItemIsUserCheckable


from PySide6.QtCore import Qt  # noqa: E402

checked = Qt.Checked
unchecked = Qt.Unchecked


class ProjectListDialog(QDialog):
    def __init__(self, projects: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shared projects")
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        self._ids = []
        for project in projects:
            self._ids.append(project["id"])
            self.list.addItem(f"{project['name']}  ({project['kind']})")
        layout.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def project_id(self) -> str | None:
        row = self.list.currentRow()
        if row < 0:
            return None
        return self._ids[row]


def read_zip(payload: bytes) -> dict[str, bytes]:
    found = {}
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        for name in archive.namelist():
            relpath = safe_rel(name)
            if relpath == "annotation_meta.json" or relpath.endswith("_seg.npy"):
                found[relpath] = archive.read(name)
    return found


def info(parent, text: str) -> None:
    QMessageBox.information(parent, "CellAnnotator", text)


def warn(parent, text: str) -> None:
    QMessageBox.warning(parent, "CellAnnotator", text)
