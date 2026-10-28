"""Opening page: the app name, a mark, sign in, and published projects."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class Mark(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(88, 88)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#22C55E"), 3))
        painter.setBrush(QColor("#1E293B"))
        painter.drawEllipse(6, 6, 76, 76)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#22C55E"))
        for origin in ((28, 26), (46, 34), (34, 48)):
            painter.drawEllipse(origin[0], origin[1], 14, 14)
        painter.end()


def _tile(locked: bool) -> QPixmap:
    mark = Mark()
    pixmap = QPixmap(mark.size())
    pixmap.fill(Qt.transparent)
    mark.render(pixmap)
    if locked:
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#F8FAFC"), 2))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Lock")
        painter.end()
    return pixmap


class HomePage(QWidget):
    signIn = Signal()
    chosen = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(16)
        layout.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(Mark())
        row.addStretch(1)
        layout.addLayout(row)
        title = QLabel("CellAnnotator")
        title.setObjectName("section")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.sign_button = QPushButton("Sign in")
        self.sign_button.setObjectName("primary")
        self.sign_button.clicked.connect(self.signIn.emit)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.sign_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self.empty = QLabel("No published projects yet.")
        self.empty.setObjectName("hint")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        layout.addWidget(self.empty)
        self.gallery = QListWidget()
        self.gallery.setObjectName("gallery")
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setSpacing(18)
        self.gallery.setIconSize(Mark().size())
        self.gallery.itemClicked.connect(self._choose)
        self.gallery.hide()
        layout.addWidget(self.gallery, 1)
        layout.addStretch(1)

    def show_signed_out(self) -> None:
        self.sign_button.show()
        self.gallery.hide()
        self.empty.hide()

    def show_projects(self, projects: list[dict]) -> None:
        self.sign_button.hide()
        self.gallery.clear()
        if not projects:
            self.gallery.hide()
            self.empty.show()
            return
        self.empty.hide()
        self.gallery.show()
        for project in projects:
            label = project["name"]
            if project.get("status") == "locked":
                label = f"{label}    Locked"
            item = QListWidgetItem(QIcon(_tile(project.get("status") == "locked")), label)
            item.setData(Qt.UserRole, project["id"])
            item.setTextAlignment(Qt.AlignHCenter)
            self.gallery.addItem(item)

    def _choose(self, item: QListWidgetItem) -> None:
        project_id = item.data(Qt.UserRole)
        if project_id:
            self.chosen.emit(project_id)


class DetailPage(QWidget):
    back = Signal()
    openProject = Signal()
    saveDescription = Signal()
    toggleLock = Signal()
    invite = Signal()
    showChanges = Signal()
    loadBranch = Signal()
    discardBranch = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(10)
        back = QPushButton("My projects")
        back.clicked.connect(self.back.emit)
        layout.addWidget(back, alignment=Qt.AlignLeft)
        self.name = QLabel("Project")
        self.name.setObjectName("section")
        self.name.setWordWrap(True)
        layout.addWidget(self.name)
        self.creator = QLabel("")
        self.creator.setObjectName("hint")
        layout.addWidget(self.creator)
        layout.addWidget(QLabel("Description"))
        self.description = QTextEdit()
        self.description.setFixedHeight(90)
        layout.addWidget(self.description)
        self.save_description = QPushButton("Save description")
        self.save_description.clicked.connect(self.saveDescription.emit)
        layout.addWidget(self.save_description, alignment=Qt.AlignLeft)
        layout.addWidget(QLabel("Working on this"))
        self.staff = QLabel("")
        self.staff.setWordWrap(True)
        layout.addWidget(self.staff)
        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.openProject.emit)
        layout.addWidget(self.open_button, alignment=Qt.AlignLeft)
        self.lock_button = QPushButton("Lock project")
        self.lock_button.clicked.connect(self.toggleLock.emit)
        layout.addWidget(self.lock_button, alignment=Qt.AlignLeft)
        self.changes_button = QPushButton("Project details")
        self.changes_button.clicked.connect(self.showChanges.emit)
        layout.addWidget(self.changes_button, alignment=Qt.AlignLeft)
        invite_row = QHBoxLayout()
        self.email = QLineEdit()
        self.email.setPlaceholderText("Email")
        self.role = QComboBox()
        self.role.addItems(["helper", "admin"])
        invite = QPushButton("Invite")
        invite.clicked.connect(self.invite.emit)
        self.invite_button = invite
        invite_row.addWidget(self.email, 1)
        invite_row.addWidget(self.role)
        invite_row.addWidget(invite)
        layout.addLayout(invite_row)
        self.review = QWidget()
        review_layout = QVBoxLayout(self.review)
        review_layout.setContentsMargins(0, 0, 0, 0)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        review_layout.addWidget(self.notice)
        self.branches = QListWidget()
        self.branches.setObjectName("branches")
        review_layout.addWidget(self.branches)
        branch_row = QHBoxLayout()
        load = QPushButton("Load these masks")
        load.clicked.connect(self.loadBranch.emit)
        discard = QPushButton("Discard")
        discard.clicked.connect(self.discardBranch.emit)
        branch_row.addWidget(load)
        branch_row.addWidget(discard)
        branch_row.addStretch(1)
        review_layout.addLayout(branch_row)
        layout.addWidget(self.review)
        layout.addStretch(1)

    def current_branch(self) -> str | None:
        item = self.branches.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)
