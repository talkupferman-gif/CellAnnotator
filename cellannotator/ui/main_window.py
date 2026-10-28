from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellannotator.images import color_name, default_layer_colors, page_count, render_display, split_layers
from cellannotator.local_server import ensure_local_server
from cellannotator.masks import delete_labels, merge_labels, split_label
from cellannotator.meta import load_meta, meta_paths
from cellannotator.project import Project
from cellannotator.segment import ModelRunner
from cellannotator.segio import seg_pair
from cellannotator.settings import Settings
from cellannotator.sync_client import SyncClient, SyncError
from cellannotator.ui.canvas import Canvas
from cellannotator.ui.dialogs import (
    BatchDialog,
    ConflictDialog,
    HistoryDialog,
    LoginDialog,
    ProjectListDialog,
    LayerColorDialog,
    ask_kind,
    confirm,
    info,
    read_zip,
    warn,
)
from cellannotator.ui.home import DetailPage, HomePage
from cellannotator.zstack import fill_links


class ModelThread(QThread):
    one = Signal(str, object)
    failed = Signal(str)
    finished_all = Signal()

    def __init__(self, jobs, model_name, weights, diameter, channels):
        super().__init__()
        self.jobs = jobs
        self.model_name = model_name
        self.weights = weights
        self.diameter = diameter
        self.channels = channels
        self.cancel = False

    def run(self) -> None:
        runner = ModelRunner()
        for relpath, image in self.jobs:
            if self.cancel:
                break
            try:
                masks = runner.run(
                    image,
                    model_name=self.model_name,
                    weights=self.weights,
                    diameter=self.diameter,
                    channels=self.channels,
                )
                self.one.emit(relpath, masks)
            except Exception as exc:  # noqa: BLE001 - show the model error in the window
                self.failed.emit(str(exc))
                break
        self.finished_all.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CellAnnotator")
        self.resize(1400, 900)
        self.settings = Settings.load()
        ensure_local_server(self.settings.server_url)
        self.project: Project | None = None
        self.current_rel: str | None = None
        self.active_link_id: str | None = None
        self.scores: dict[str, int | None] = {}
        self.weights_path: str | None = None
        self._thread: ModelThread | None = None
        self._showing_score = False
        self._asked_login = False
        self.member_role: str | None = None
        self._remote: dict | None = None
        self._build()
        self._account_label()
        self._show_home()

    def _build(self) -> None:
        self.canvas = Canvas()
        self.canvas.edited.connect(self._on_edited)
        self.canvas.labelsTouched.connect(self._on_labels)
        self.canvas.status.connect(self.statusBar().showMessage)
        self.setStyleSheet(_STYLE)
        self._build_model_dialog()
        self._build_image_dock()
        self._build_menus()
        self._build_toolbar()

        hint = QLabel(
            "Right-click and close the loop. Draw fills a cell. Erase cuts inside the loop. "
            "X shows the image alone. Z shows outlines. The menu has images, ranking, colors, models, and versions."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(hint)
        self.work = QWidget()
        self.work.setLayout(layout)
        self.home = HomePage()
        self.home.signIn.connect(self.login)
        self.home.chosen.connect(self._show_project)
        self.detail = DetailPage()
        self.detail.back.connect(self._show_home)
        self.detail.openProject.connect(self._open_from_detail)
        self.detail.saveDescription.connect(self._save_description)
        self.detail.toggleLock.connect(self._toggle_lock)
        self.detail.invite.connect(self._invite_from_detail)
        self.detail.showChanges.connect(self._show_changes)
        self.detail.loadBranch.connect(self._load_branch)
        self.detail.discardBranch.connect(self._discard_branch)
        self.pages = QStackedWidget()
        self.pages.addWidget(self.home)
        self.pages.addWidget(self.detail)
        self.pages.addWidget(self.work)
        self.setCentralWidget(self.pages)
        self.statusBar().showMessage("Sign in to see your projects")

    def _build_toolbar(self) -> None:
        self.draw_button = QToolButton()
        self.draw_button.setText("Draw")
        self.draw_button.setCheckable(True)
        self.draw_button.setChecked(True)
        self.erase_button = QToolButton()
        self.erase_button.setText("Erase")
        self.erase_button.setObjectName("erase")
        self.erase_button.setCheckable(True)
        modes = QButtonGroup(self)
        modes.setExclusive(True)
        modes.addButton(self.draw_button)
        modes.addButton(self.erase_button)
        self.draw_button.clicked.connect(lambda: self.canvas.set_mode("draw"))
        self.erase_button.clicked.connect(lambda: self.canvas.set_mode("erase"))
        bar = QToolBar("Annotate")
        bar.setObjectName("annotate")
        bar.setMovable(False)
        bar.setFloatable(False)
        for widget in (
            self.draw_button,
            self.erase_button,
            _action("Merge", self.merge),
            _action("Delete", self.delete_selected),
            _action("Go back", self.undo),
            _action("Redo", self.redo),
        ):
            bar.addWidget(widget)
        self.addToolBar(bar)
        bar.hide()
        self.toolbar = bar

    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("My projects", self._show_home)
        file_menu.addAction("Open folder", self.open_folder)
        save_action = file_menu.addAction("Save", self.save)
        save_action.setShortcut(QKeySequence.Save)
        file_menu.addAction("Open shared", self.open_shared)

        account = menu.addMenu("Account")
        self.account_action = account.addAction("Not logged in")
        self.account_action.setEnabled(False)
        account.addSeparator()
        account.addAction("Sign in", self.login)

        version = menu.addMenu("Version")
        version.addAction("Publish", self.publish)
        version.addAction("Commit", self.commit)
        version.addAction("Pull", self.pull)
        version.addAction("History", self.show_history)
        version.addAction("Invite", self.invite)

        image = menu.addMenu("Image")
        self.images_action = image.addAction("Images and ranking")
        self.images_action.setCheckable(True)
        self.images_action.setChecked(False)
        self.images_action.toggled.connect(self.images_dock.setVisible)
        image.addAction("Layer colors", self.edit_layer_colors)
        image.addAction("Fit", self.canvas.fit)
        channel_menu = image.addMenu("Channel")
        channel_group = QActionGroup(self)
        channel_group.setExclusive(True)
        self.channel_actions = {}
        for label, data in (("Layers", "layers"), ("Gray", "gray"), ("Red", "r"), ("Green", "g"), ("Blue", "b")):
            action = channel_menu.addAction(label)
            action.setCheckable(True)
            action.setData(data)
            action.setChecked(data == "layers")
            channel_group.addAction(action)
            self.channel_actions[data] = action
        channel_group.triggered.connect(self._change_channel)

        model_menu = menu.addMenu("Model")
        model_menu.addAction("Run model…", self.show_model_dialog)
        model_menu.addAction("Run on this image", self.run_current)
        model_menu.addAction("Run batch", self.run_batch)

        view = menu.addMenu("View")
        self.masks_action = view.addAction("Masks")
        self.masks_action.setCheckable(True)
        self.masks_action.setChecked(True)
        self.masks_action.setShortcut(QKeySequence("X"))
        self.masks_action.toggled.connect(self._set_masks_visible)
        self.outlines_action = view.addAction("Outlines")
        self.outlines_action.setCheckable(True)
        self.outlines_action.setChecked(False)
        self.outlines_action.setShortcut(QKeySequence("Z"))
        self.outlines_action.toggled.connect(self._set_outlines)
        self.fill_action_view = view.addAction("Mask fill")
        self.fill_action_view.setCheckable(True)
        self.fill_action_view.setChecked(True)
        self.fill_action_view.toggled.connect(self._set_fill)

        cells = menu.addMenu("Cells")
        cells.addAction("Split", self.split_cells)
        self.same_cell_action = cells.addAction("Same cell", self.same_cell)
        self.finish_action = cells.addAction("Finish linking", self.finish_linking)
        self.fill_action = cells.addAction("Fill", self.fill)
        undo_action = cells.addAction("Go back", self.undo)
        undo_action.setShortcut(QKeySequence.Undo)
        redo_action = cells.addAction("Redo", self.redo)
        redo_action.setShortcut(QKeySequence.Redo)

    def _build_image_dock(self) -> None:
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_list)
        self.score = QComboBox()
        self.score.addItem("Not set")
        for value in range(1, 11):
            self.score.addItem(str(value))
        self.score.currentIndexChanged.connect(self._on_score)
        self.sort = QComboBox()
        self.sort.addItems(["Filename order", "Score, least segmented first"])
        self.sort.currentIndexChanged.connect(lambda _index: self._refresh_list())
        self.link_label = QLabel("No cell link in progress")
        self.link_label.setWordWrap(True)
        side = QVBoxLayout()
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(8)
        title = QLabel("Images")
        title.setObjectName("section")
        side.addWidget(title)
        side.addWidget(self.list, 1)
        side.addWidget(QLabel("Ready: 1 needs work, 10 is done"))
        side.addWidget(self.score)
        side.addWidget(QLabel("Sort"))
        side.addWidget(self.sort)
        side.addWidget(self.link_label)
        side_widget = QWidget()
        side_widget.setObjectName("side")
        side_widget.setLayout(side)
        self.images_dock = QDockWidget("Images", self)
        self.images_dock.setMinimumWidth(280)
        self.images_dock.setWidget(side_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.images_dock)
        self.images_dock.hide()
        self.images_dock.visibilityChanged.connect(self._sync_images_action)

    def _build_model_dialog(self) -> None:
        self.model_dialog = QDialog(self)
        self.model_dialog.setWindowTitle("Run model")
        self.model_dialog.setMinimumWidth(420)
        self.model = QComboBox()
        self.model.addItems(["cyto3", "nuclei", "Custom weights"])
        self.weights_button = QPushButton("Weights file")
        self.weights_button.clicked.connect(self.pick_weights)
        self.weights_name = QLabel("No weights file")
        self.weights_name.setObjectName("hint")
        self.auto_diameter = QPushButton("Auto size")
        self.auto_diameter.setCheckable(True)
        self.auto_diameter.setChecked(True)
        self.diameter = QSpinBox()
        self.diameter.setRange(1, 2000)
        self.diameter.setValue(30)
        self.diameter.setEnabled(False)
        self.auto_diameter.toggled.connect(lambda on: self.diameter.setEnabled(not on))
        self.cyto = QComboBox()
        self.nucleus = QComboBox()
        note = QLabel(
            "cyto3 and nuclei are the built-in models. "
            "For one of your own, choose Custom weights and pick the file. "
            "The main channel is the stain to segment. The nuclei channel is optional. "
            "Each choice shows the color of that channel in this image."
        )
        note.setWordWrap(True)
        run = QPushButton("Run on this image")
        run.setObjectName("primary")
        run.clicked.connect(self.run_current)
        batch = QPushButton("Run batch")
        batch.clicked.connect(self.run_batch)
        form = QVBoxLayout(self.model_dialog)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(8)
        form.addWidget(note)
        form.addWidget(QLabel("Model"))
        form.addWidget(self.model)
        form.addWidget(self.weights_button)
        form.addWidget(self.weights_name)
        row = QHBoxLayout()
        row.addWidget(self.auto_diameter)
        row.addWidget(self.diameter)
        form.addLayout(row)
        form.addWidget(QLabel("Main channel"))
        form.addWidget(self.cyto)
        form.addWidget(QLabel("Nuclei channel"))
        form.addWidget(self.nucleus)
        self._fill_channel_choices()
        buttons = QHBoxLayout()
        buttons.addWidget(run)
        buttons.addWidget(batch)
        form.addLayout(buttons)
        self.channel = QComboBox()
        for label, data in (("Layers", "layers"), ("Gray", "gray"), ("Red", "r"), ("Green", "g"), ("Blue", "b")):
            self.channel.addItem(label, data)

    def show_model_dialog(self) -> None:
        self._fill_channel_choices()
        self.model_dialog.show()
        self.model_dialog.raise_()
        self.model_dialog.activateWindow()

    def _layer_colors_for_model(self) -> list[str | None]:
        if self.project is None or self.current_rel is None:
            return default_layer_colors(3)
        planes = split_layers(self.project.array_of(self.current_rel))
        count = max(len(planes), 1)
        colors = list(self.project.layer_colors or default_layer_colors(count))
        if len(colors) < count:
            colors.extend(default_layer_colors(count)[len(colors) :])
        return colors[:count]

    def _fill_channel_choices(self) -> None:
        colors = self._layer_colors_for_model()
        previous_main = self.cyto.currentData()
        previous_nuclei = self.nucleus.currentData()
        self.cyto.clear()
        self.nucleus.clear()
        self.cyto.addItem("Gray", 0)
        self.nucleus.addItem("None", 0)
        for index, color in enumerate(colors, start=1):
            label = f"{index} · {color_name(color)}"
            icon = _color_swatch(color)
            self.cyto.addItem(icon, label, index)
            self.nucleus.addItem(icon, label, index)
        _select_data(self.cyto, 0 if previous_main is None else previous_main)
        _select_data(self.nucleus, 0 if previous_nuclei is None else previous_nuclei)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)

    def _sync_images_action(self, visible: bool) -> None:
        if not hasattr(self, "images_action"):
            return
        self.images_action.blockSignals(True)
        self.images_action.setChecked(visible)
        self.images_action.blockSignals(False)

    def _set_masks_visible(self, on: bool) -> None:
        self.canvas.masks_visible = on
        self.canvas.refresh()
        self.statusBar().showMessage("Image only" if not on else "Masks on")

    def _set_outlines(self, on: bool) -> None:
        self.canvas.show_outlines = on
        self.canvas.refresh()
        self.statusBar().showMessage("Outlines on" if on else "Fill only")

    def _set_fill(self, on: bool) -> None:
        self.canvas.show_fill = on
        self.canvas.refresh()

    def _account_label(self) -> None:
        if self.settings.email and self.settings.token:
            self.account_action.setText(self.settings.email)
        else:
            self.account_action.setText("Not logged in")

    def _show_home(self) -> None:
        self.toolbar.hide()
        self.images_dock.hide()
        self.pages.setCurrentWidget(self.home)
        if not self.settings.token:
            self.home.show_signed_out()
            self.statusBar().showMessage("Sign in to see your projects")
            return
        client = self._client()
        if client is None:
            self.home.show_signed_out()
            return
        try:
            projects = client.list_projects()
        except SyncError as exc:
            warn(self, str(exc))
            projects = []
        self.home.show_projects(projects)
        self.statusBar().showMessage("Choose a project")

    def _show_project(self, project_id: str) -> None:
        client = self._client()
        if client is None:
            return
        try:
            remote = client.get_project(project_id)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._remote = remote
        self.member_role = remote.get("my_role")
        self.detail.name.setText(remote["name"])
        self.detail.creator.setText(f"Created by {remote.get('owner_email') or 'unknown'}")
        self.detail.description.setPlainText(remote.get("description") or "")
        staff = remote.get("staff") or []
        self.detail.staff.setText(
            "\n".join(f"{person['email']}  ({person['role']})" for person in staff) or "No admins yet."
        )
        locked = remote.get("status") == "locked"
        can_edit = self.member_role in {"owner", "admin"}
        self.detail.description.setReadOnly(not can_edit)
        self.detail.save_description.setVisible(can_edit)
        self.detail.email.setVisible(can_edit)
        self.detail.role.setVisible(can_edit)
        self.detail.invite_button.setVisible(can_edit)
        self.detail.review.setVisible(can_edit)
        self.detail.lock_button.setVisible(self.member_role == "owner")
        self.detail.lock_button.setText("Unlock project" if locked else "Lock project")
        self.detail.changes_button.setVisible(self.member_role == "owner")
        self._fill_branches(client, project_id)
        self.pages.setCurrentWidget(self.detail)
        self.toolbar.hide()

    def _fill_branches(self, client: SyncClient, project_id: str) -> None:
        self.detail.branches.clear()
        self.detail.notice.clear()
        if self.member_role not in {"owner", "admin"}:
            self.detail.notice.hide()
            self.detail.branches.hide()
            return
        try:
            branches = client.list_branches(project_id)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.detail.branches.setVisible(bool(branches))
        if not branches:
            self.detail.notice.hide()
            return
        self.detail.notice.show()
        self.detail.notice.setText("A helper committed a change. Load the masks, edit them, then commit. Or discard them.")
        for branch in branches:
            paths = ", ".join(branch.get("paths") or [])
            item = QListWidgetItem(f"{branch['author_email']}  {paths}")
            item.setData(Qt.UserRole, branch["id"])
            self.detail.branches.addItem(item)
        self.detail.branches.setCurrentRow(0)

    def _enter_work(self) -> None:
        self.pages.setCurrentWidget(self.work)
        self.toolbar.show()

    def _open_from_detail(self) -> None:
        if not self._remote:
            return
        project_id = self._remote["id"]
        remembered = self.settings.folder_for(project_id)
        folder = remembered if remembered and Path(remembered).is_dir() else ""
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Image folder for this project")
        if not folder:
            return
        self.settings.remember_folder(project_id, folder)
        self._open_directory(self._remote.get("kind"), project_id, folder)
        if self.project is None or self.project.state.get("project_id") != project_id:
            return
        self._enter_work()
        self._notice_branches()

    def _notice_branches(self) -> None:
        if self.member_role not in {"owner", "admin"} or self._remote is None:
            return
        client = self._client()
        if client is None:
            return
        try:
            branches = client.list_branches(self._remote["id"])
        except SyncError:
            return
        if not branches:
            return
        lines = [f"{branch['author_email']}: {', '.join(branch.get('paths') or [])}" for branch in branches]
        info(self, "A helper committed a change.\n" + "\n".join(lines))

    def _save_description(self) -> None:
        if not self._remote or self.member_role not in {"owner", "admin"}:
            return
        client = self._client()
        if client is None:
            return
        text = self.detail.description.toPlainText().strip()
        try:
            client.set_description(self._remote["id"], text)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._remote["description"] = text
        self.statusBar().showMessage("Saved the description")

    def _toggle_lock(self) -> None:
        if not self._remote or self.member_role != "owner":
            return
        client = self._client()
        if client is None:
            return
        locked = self._remote.get("status") != "locked"
        try:
            result = client.set_locked(self._remote["id"], locked)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._remote["status"] = result["status"]
        self.detail.lock_button.setText("Unlock project" if result["status"] == "locked" else "Lock project")
        self.statusBar().showMessage("Project locked" if result["status"] == "locked" else "Project unlocked")

    def _invite_from_detail(self) -> None:
        if not self._remote:
            return
        email = self.detail.email.text().strip()
        if not email:
            return
        client = self._client()
        if client is None:
            return
        try:
            result = client.invite(self._remote["id"], email, self.detail.role.currentText())
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.detail.email.clear()
        info(self, "They have access." if result["status"] == "joined" else "Invite saved until they create an account.")

    def _show_changes(self) -> None:
        if not self._remote or self.member_role != "owner":
            return
        client = self._client()
        if client is None:
            return
        try:
            rows = client.changes(self._remote["id"])
        except SyncError as exc:
            warn(self, str(exc))
            return
        if not rows:
            info(self, "No commits yet.")
            return
        lines = []
        for row in rows:
            lines.append(f"{row['created_at']}  {row['author_email']}")
            images = row.get("images") or []
            if not images:
                lines.append("  No cell counts for this commit.")
            for image in images:
                lines.append(
                    f"  {image['path']}: {image['added']} added, {image['deleted']} deleted, {image['edited']} edited"
                )
        info(self, "\n".join(lines))

    def _load_branch(self) -> None:
        if not self._remote or self.member_role not in {"owner", "admin"}:
            return
        branch_id = self.detail.current_branch()
        if not branch_id:
            warn(self, "Choose a helper commit first.")
            return
        if self.project is None or self.project.state.get("project_id") != self._remote["id"]:
            self._open_from_detail()
        if self.project is None or self.project.state.get("project_id") != self._remote["id"]:
            return
        client = self._client()
        if client is None:
            return
        try:
            payload = client.branch_archive(self._remote["id"], branch_id)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._apply_snapshot(client, self._remote["id"], branch_id, mode="working", payload=payload)
        self.project.meta = load_meta(self.project.root)
        files = read_zip(payload)
        image = None
        for path in files:
            if path.endswith("_seg.npy"):
                image = self._image_for_seg(path)
                if image:
                    break
        if image:
            self._load(image)
        else:
            self._reload_current()
        self._enter_work()
        self.statusBar().showMessage("Loaded the helper masks into the working copy. Commit to share them.")

    def _image_for_seg(self, seg_rel: str) -> str | None:
        if self.project is None:
            return None
        for relpath in self.project.rels:
            committed, _working = seg_pair(self.project.image_path(relpath))
            if self.project.rel(committed) == seg_rel:
                return relpath
        return None

    def _discard_branch(self) -> None:
        if not self._remote or self.member_role not in {"owner", "admin"}:
            return
        branch_id = self.detail.current_branch()
        if not branch_id:
            warn(self, "Choose a helper commit first.")
            return
        if not confirm(self, "Discard this helper commit? Its masks will be removed from the shared project."):
            return
        client = self._client()
        if client is None:
            return
        try:
            client.discard_branch(self._remote["id"], branch_id)
            self._fill_branches(client, self._remote["id"])
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.statusBar().showMessage("Discarded the helper commit")

    def _client(self) -> SyncClient | None:
        if not self.settings.token:
            warn(self, "Log in first.")
            return None
        return SyncClient(self.settings.server_url, self.settings.token)

    def open_folder(self) -> None:
        self._open_directory(None, None)

    def _open_directory(self, kind: str | None, project_id: str | None, folder: str | None = None) -> None:
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Image folder")
        if not folder:
            return
        self._flush_current()
        root = Path(folder)
        project = Project(root)
        committed, working = meta_paths(root)
        if not committed.exists() and not working.exists():
            chosen = kind or ask_kind(self)
            if chosen is None:
                return
            project.meta.kind = chosen
            project.write_working_meta()
        elif kind and project.meta.kind != kind:
            warn(self, f"This folder was marked {project.meta.kind}. The shared project is {kind}.")
        if project_id:
            existing = project.state.get("project_id")
            if existing and existing != project_id:
                warn(self, "This folder belongs to a different project.")
                return
            project.state["project_id"] = project_id
            project.save_state()
        if not project.images:
            warn(self, "That folder has no images.")
            return
        self.project = project
        self.current_rel = None
        self.active_link_id = None
        self.statusBar().showMessage("Reading saved masks…")
        self.scores = {rel: project.load_session(rel).readiness for rel in project.rels}
        self._refresh_list()
        self._set_stack_tools()
        item = self.list.currentItem()
        if item is not None:
            self._load(item.data(Qt.UserRole))
        self.statusBar().showMessage(f"Opened {root}")
        self._enter_work()

    def _set_stack_tools(self) -> None:
        enabled = bool(self.project and self.project.meta.kind == "zstack")
        for action in (self.same_cell_action, self.finish_action, self.fill_action):
            action.setEnabled(enabled)

    def _ordered_rels(self) -> list[str]:
        rels = self.project.rels if self.project else []
        if self.sort.currentIndex() == 0:
            return rels
        return sorted(rels, key=lambda rel: (self.scores.get(rel) is not None, self.scores.get(rel) or 0, rel))

    def _refresh_list(self) -> None:
        if self.project is None:
            return
        self.list.blockSignals(True)
        self.list.clear()
        selected_row = 0
        for index, relpath in enumerate(self._ordered_rels()):
            score = self.scores.get(relpath)
            mark = "·" if not score else str(score)
            item = self.list.addItem(f"{mark:>2}  {relpath}")
            self.list.item(index).setData(Qt.UserRole, relpath)
            if relpath == self.current_rel:
                selected_row = index
            del item
        if self.list.count():
            self.list.setCurrentRow(selected_row)
        self.list.blockSignals(False)

    def _on_list(self, current, _previous) -> None:
        if current is None:
            return
        relpath = current.data(Qt.UserRole)
        if relpath == self.current_rel:
            return
        self._flush_current()
        self._load(relpath)

    def _load(self, relpath: str) -> None:
        assert self.project is not None
        self.current_rel = relpath
        array = self.project.array_of(relpath)
        session = self.project.load_session(relpath)
        self.canvas.set_image(session, self._rgb(array))
        self._apply_link_colors()
        self._showing_score = True
        self.score.setCurrentIndex(0 if session.readiness is None else session.readiness)
        self._showing_score = False
        self._show_link_status()
        self.canvas.setFocus()
        if session.warning:
            self.statusBar().showMessage(session.warning)
        path = self.project.image_path(relpath)
        if page_count(path) > 1 and len(split_layers(array)) == 1:
            self.statusBar().showMessage(f"{relpath}: this file has many pages. Showing the first page.")

    def _flush_current(self) -> None:
        if self.project is None or self.current_rel is None or self.canvas.session is None:
            return
        if self.canvas.session.dirty:
            self.project.write_working(self.current_rel, self.canvas.session)
            self.scores[self.current_rel] = self.canvas.session.readiness

    def save(self) -> None:
        if self.project is None:
            return
        self._flush_current()
        self.project.write_working_meta()
        self.statusBar().showMessage("Saved the working copy. Commit to publish it.")

    def _on_score(self, index: int) -> None:
        if self._showing_score or self.canvas.session is None:
            return
        self.canvas.session.readiness = None if index == 0 else index
        self.canvas.session.dirty = True
        if self.current_rel:
            self.scores[self.current_rel] = self.canvas.session.readiness
        self._refresh_list()

    def _rgb(self, array):
        colors = self.project.layer_colors if self.project else None
        return render_display(array, self.channel.currentData() or "layers", colors)

    def _change_channel(self, action: QAction | None = None) -> None:
        if isinstance(action, QAction):
            index = self.channel.findData(action.data())
            if index >= 0:
                self.channel.setCurrentIndex(index)
        if self.project is None or self.current_rel is None:
            return
        self.canvas.set_display(self._rgb(self.project.array_of(self.current_rel)))

    def edit_layer_colors(self) -> None:
        if self.project is None or self.current_rel is None:
            warn(self, "Open an image folder first.")
            return
        planes = split_layers(self.project.array_of(self.current_rel))
        colors = list(self.project.layer_colors or default_layer_colors(len(planes)))
        if len(colors) < len(planes):
            colors.extend(default_layer_colors(len(planes))[len(colors) :])
        dialog = LayerColorDialog(colors[: len(planes)], self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.project.save_layer_colors(dialog.colors())
        self.channel.setCurrentIndex(0)
        layers = self.channel_actions.get("layers")
        if layers is not None:
            layers.setChecked(True)
        self._change_channel()
        self.statusBar().showMessage("Saved layer colors for this folder")

    def _on_edited(self) -> None:
        self.statusBar().showMessage("Edited the working copy")

    def _on_labels(self, labels: set[int]) -> None:
        if self.project is None or self.current_rel is None or self.canvas.session is None:
            return
        if self.project.meta.kind != "zstack":
            return
        changed = False
        for label in labels:
            if not (self.canvas.session.masks == label).any():
                self.project.meta.remove_label(self.current_rel, label)
                changed = True
            elif self.project.meta.promote_filled(self.current_rel, label):
                changed = True
        if changed:
            self.project.write_working_meta()
            self._apply_link_colors()
            self._show_link_status()

    def _apply_link_colors(self) -> None:
        if self.project is None or self.current_rel is None:
            return
        keys = {}
        for link in self.project.meta.links:
            for bucket in (link.anchors, link.filled):
                label = bucket.get(self.current_rel)
                if label:
                    keys[int(label)] = link.id
        self.canvas.set_color_keys(keys)

    def _show_link_status(self) -> None:
        if self.project is None or not self.active_link_id:
            self.link_label.setText("No cell link in progress")
            return
        link = self.project.meta.link_by_id(self.active_link_id)
        if link is None:
            self.link_label.setText("No cell link in progress")
            return
        parts = [f"{rel} #{label}" for rel, label in link.anchors.items()]
        self.link_label.setText("Linking cell " + link.id + ": " + ", ".join(parts))

    def merge(self) -> None:
        session = self.canvas.session
        if session is None or self.project is None or self.current_rel is None:
            return
        if len(self.canvas.selected) < 2:
            warn(self, "Shift-click at least two cells, then Merge.")
            return
        session.checkpoint()
        updated, keep = merge_labels(session.masks, set(self.canvas.selected))
        if not keep:
            session.drop_checkpoint()
            return
        mapping = {label: keep for label in self.canvas.selected}
        session.masks = updated
        session.set_manual(keep, True)
        session.dirty = True
        self.project.meta.rewrite_label(self.current_rel, mapping)
        self.project.write_working_meta()
        self.canvas.selected = {keep}
        self.canvas.refresh()
        self._apply_link_colors()

    def split_cells(self) -> None:
        session = self.canvas.session
        if session is None or len(self.canvas.selected) != 1:
            warn(self, "Select one cell to split.")
            return
        label = next(iter(self.canvas.selected))
        session.checkpoint()
        updated, created = split_label(session.masks, label)
        if not created:
            session.drop_checkpoint()
            self.statusBar().showMessage("That cell is still one piece")
            return
        session.masks = updated
        session.set_manual(label, True)
        for new_label in created:
            session.set_manual(new_label, True)
        session.dirty = True
        self.canvas.refresh()
        self.statusBar().showMessage(f"Split off {len(created)} piece(s)")

    def delete_selected(self) -> None:
        session = self.canvas.session
        if session is None or self.project is None or self.current_rel is None or not self.canvas.selected:
            return
        if not confirm(self, "Delete the selected cells?"):
            return
        session.checkpoint()
        session.masks = delete_labels(session.masks, set(self.canvas.selected))
        for label in list(self.canvas.selected):
            self.project.meta.remove_label(self.current_rel, label)
        session.dirty = True
        self.canvas.selected.clear()
        self.project.write_working_meta()
        self.canvas.refresh()
        self._apply_link_colors()

    def undo(self) -> None:
        if self.canvas.session and self.canvas.session.undo():
            self.canvas.refresh()
            self._apply_link_colors()
            self.statusBar().showMessage("Went back one edit")

    def redo(self) -> None:
        if self.canvas.session and self.canvas.session.redo():
            self.canvas.refresh()
            self._apply_link_colors()
            self.statusBar().showMessage("Redid the edit")

    def same_cell(self) -> None:
        if self.project is None or self.project.meta.kind != "zstack" or self.current_rel is None:
            return
        if len(self.canvas.selected) != 1 or self.canvas.session is None:
            warn(self, "Select one cell, then Same cell.")
            return
        label = next(iter(self.canvas.selected))
        existing = self.project.meta.link_for_anchor(self.current_rel, label)
        if existing and existing.id != self.active_link_id:
            self.active_link_id = existing.id
            self._show_link_status()
            self.statusBar().showMessage("Select the same cell on another slice and press Same cell again")
            return
        for link in list(self.project.meta.links):
            if link.anchors.get(self.current_rel) == label and link.id != self.active_link_id:
                del link.anchors[self.current_rel]
        self.project.meta.links = [link for link in self.project.meta.links if link.anchors]
        if self.active_link_id is None:
            link = self.project.meta.new_link(self.current_rel, label)
            self.active_link_id = link.id
        else:
            link = self.project.meta.link_by_id(self.active_link_id)
            if link is None:
                link = self.project.meta.new_link(self.current_rel, label)
                self.active_link_id = link.id
            else:
                link.filled.pop(self.current_rel, None)
                link.anchors[self.current_rel] = label
        self.canvas.session.set_manual(label, True)
        self.canvas.session.dirty = True
        self.project.write_working_meta()
        self._apply_link_colors()
        self._show_link_status()

    def finish_linking(self) -> None:
        self.active_link_id = None
        self._show_link_status()

    def fill(self) -> None:
        if self.project is None or self.project.meta.kind != "zstack":
            return
        self._flush_current()
        sessions = {}
        for relpath in self.project.rels:
            if relpath == self.current_rel and self.canvas.session is not None:
                sessions[relpath] = self.canvas.session
            else:
                sessions[relpath] = self.project.load_session(relpath)
        warnings = fill_links(self.project.meta, sessions, self.project.rels)
        for relpath, session in sessions.items():
            if session.dirty:
                self.project.write_working(relpath, session)
        self.project.write_working_meta()
        self.canvas.refresh()
        self._apply_link_colors()
        if warnings:
            warn(self, "\n".join(warnings[:12]))
        else:
            self.statusBar().showMessage("Filled slices between linked cells")

    def pick_weights(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Cellpose weights")
        if path:
            self.weights_path = path
            self.model.setCurrentText("Custom weights")
            self.weights_name.setText(Path(path).name)
            self.statusBar().showMessage(Path(path).name)

    def _model_args(self):
        name = self.model.currentText()
        weights = self.weights_path if name == "Custom weights" else None
        if name == "Custom weights" and not weights:
            warn(self, "Choose a weights file first.")
            return None
        diameter = None if self.auto_diameter.isChecked() else float(self.diameter.value())
        channels = [int(self.cyto.currentData() or 0), int(self.nucleus.currentData() or 0)]
        model_name = "cyto3" if name == "Custom weights" else name
        return model_name, weights, diameter, channels

    def run_current(self) -> None:
        if self.project is None or self.current_rel is None:
            warn(self, "Open an image first. Use the menu, then Image, then Images and ranking.")
            return
        self._start_model([self.current_rel])

    def run_batch(self) -> None:
        if self.project is None:
            return
        dialog = BatchDialog(self.project.rels, self.current_rel, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.chosen()
        if not chosen:
            return
        self._start_model(chosen)

    def _start_model(self, rels: list[str]) -> None:
        if self.project is None or self._thread is not None:
            return
        args = self._model_args()
        if args is None:
            return
        jobs = [(relpath, self.project.array_of(relpath)) for relpath in rels]
        self._thread = ModelThread(jobs, *args)
        self._thread.one.connect(self._apply_model)
        self._thread.failed.connect(lambda message: warn(self, message))
        self._thread.finished_all.connect(self._model_finished)
        self.statusBar().showMessage("Running segmentation…")
        self._thread.start()

    def _apply_model(self, relpath: str, masks) -> None:
        if self.project is None:
            return
        if relpath == self.current_rel and self.canvas.session is not None:
            session = self.canvas.session
        else:
            session = self.project.load_session(relpath)
        masks = np.asarray(masks, dtype=np.int32)
        if tuple(masks.shape) != tuple(session.masks.shape):
            self.statusBar().showMessage(f"{relpath}: model output size does not match the image")
            return
        session.checkpoint()
        session.masks = masks
        session.ismanual = np.zeros(int(session.masks.max()), dtype=bool)
        session.dirty = True
        self.project.meta.drop_slice(relpath)
        self.project.write_working(relpath, session)
        self.project.write_working_meta()
        if relpath == self.current_rel:
            self.canvas.selected.clear()
            self.canvas.refresh()
            self._apply_link_colors()
        self.statusBar().showMessage(f"Model masks are in the working copy for {relpath}")

    def _model_finished(self) -> None:
        self._thread = None

    def login(self) -> None:
        dialog = LoginDialog(self.settings.server_url, self.settings.email, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.server_url = dialog.server.text().strip() or self.settings.server_url
        if not ensure_local_server(self.settings.server_url):
            warn(self, "Cannot reach the account server. Check the address, or leave it as http://127.0.0.1:8765.")
            return
        client = SyncClient(self.settings.server_url)
        try:
            if dialog.action == "register":
                result = client.register(dialog.email.text(), dialog.password.text())
            else:
                result = client.login(dialog.email.text(), dialog.password.text())
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.settings.token = result["token"]
        self.settings.email = result["email"]
        self.settings.save()
        self._account_label()
        self._show_home()
        self.statusBar().showMessage(f"Logged in as {self.settings.email}")

    def publish(self) -> None:
        client = self._client()
        if client is None or self.project is None:
            return
        name, ok = QInputDialog.getText(self, "Publish project", "Project name")
        if not ok or not name.strip():
            return
        description, ok = QInputDialog.getMultiLineText(self, "Publish project", "Description")
        if not ok:
            return
        self.statusBar().showMessage("Hashing images…")
        try:
            created = client.create_project(
                name.strip(),
                self.project.meta.kind,
                {"files": self.project.file_manifest()},
                description.strip(),
            )
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.project.state["project_id"] = created["id"]
        self.project.state["base_commit_id"] = None
        self.project.save_state()
        self.settings.remember_folder(created["id"], str(self.project.root))
        self.member_role = "owner"
        self.statusBar().showMessage("Project published. Commit when you want to share masks.")

    def commit(self) -> None:
        client = self._client()
        if client is None or self.project is None:
            return
        project_id = self.project.state.get("project_id")
        if not project_id:
            warn(self, "Publish the project before committing.")
            return
        self._flush_current()
        self.project.write_working_meta()
        files = self.project.iter_publishable()
        if not files:
            info(self, "There are no working-copy changes to commit.")
            return
        role = self._role_for(project_id)
        if role != "helper" and not confirm(self, "Commit these masks to the shared project?"):
            return
        parent = self.project.state.get("base_commit_id")
        try:
            result = self._commit_with_conflicts(client, project_id, parent, files)
        except SyncError as exc:
            warn(self, str(exc))
            return
        if result.get("branch"):
            self.statusBar().showMessage("Committed to your branch. An owner or admin will review it.")
            return
        self._apply_snapshot(client, project_id, result["id"], mode="committed")
        self.project.state["base_commit_id"] = result["id"]
        self.project.save_state()
        self.project.meta = load_meta(self.project.root)
        self._reload_current()
        self.statusBar().showMessage("Committed. The shared masks now match this working copy.")

    def _role_for(self, project_id: str) -> str | None:
        if self.member_role and self._remote and self._remote.get("id") == project_id:
            return self.member_role
        client = self._client()
        if client is None:
            return self.member_role
        try:
            remote = client.get_project(project_id)
        except SyncError:
            return self.member_role
        self._remote = remote
        self.member_role = remote.get("my_role")
        return self.member_role

    def _commit_with_conflicts(self, client, project_id, parent, files):
        try:
            return client.commit(project_id, parent, files)
        except SyncError as exc:
            conflicts = exc.payload.get("conflicts") or []
            if exc.status != 409 or not conflicts:
                if exc.status == 409:
                    raise SyncError("Pull the latest commit before committing.") from exc
                raise
            dialog = ConflictDialog(conflicts, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                raise SyncError("Commit cancelled") from exc
            return client.commit(project_id, parent, files, dialog.resolutions())

    def pull(self) -> None:
        client = self._client()
        if client is None or self.project is None:
            return
        project_id = self.project.state.get("project_id")
        if not project_id:
            warn(self, "This folder is not a published project.")
            return
        try:
            remote = client.get_project(project_id)
            head = remote.get("head_id")
            if not head:
                info(self, "Nothing has been committed yet.")
                return
            payload = client.archive(project_id, head)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._flush_current()
        skipped = self._apply_snapshot(client, project_id, head, mode="pull", payload=payload)
        if not skipped:
            self.project.state["base_commit_id"] = head
            self.project.save_state()
        self.project.meta = load_meta(self.project.root)
        self._reload_current()
        self._report_manifest(remote)
        if skipped:
            warn(self, "Left these local edits in place:\n" + "\n".join(skipped))
        else:
            self.statusBar().showMessage("Pulled the latest commit")

    def _apply_snapshot(self, client, project_id, commit_id, mode: str, payload: bytes | None = None) -> list[str]:
        assert self.project is not None
        data = payload if payload is not None else client.archive(project_id, commit_id)
        files = read_zip(data)
        skipped = []
        for relpath, body in files.items():
            working_rel = _working_name(relpath)
            working_path = self.project.root / working_rel
            if mode == "pull" and working_path.exists():
                skipped.append(relpath)
                continue
            if mode == "working":
                working_path.parent.mkdir(parents=True, exist_ok=True)
                working_path.write_bytes(body)
            else:
                self.project.write_committed_bytes(relpath, body)
        return skipped

    def _reload_current(self) -> None:
        if self.current_rel and self.project:
            self.scores[self.current_rel] = self.project.load_session(self.current_rel).readiness
            self._load(self.current_rel)
            self._refresh_list()

    def show_history(self) -> None:
        client = self._client()
        if client is None or self.project is None or not self.project.state.get("project_id"):
            return
        try:
            commits = client.list_commits(self.project.state["project_id"])
        except SyncError as exc:
            warn(self, str(exc))
            return
        if not commits:
            info(self, "No commits yet.")
            return
        dialog = HistoryDialog(commits, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.chosen:
            return
        try:
            self._apply_snapshot(client, self.project.state["project_id"], dialog.chosen, mode="working")
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.project.meta = load_meta(self.project.root)
        self._reload_current()
        self.statusBar().showMessage("Restored that commit into the working copy")

    def open_shared(self) -> None:
        client = self._client()
        if client is None:
            return
        try:
            projects = client.list_projects()
        except SyncError as exc:
            warn(self, str(exc))
            return
        if not projects:
            info(self, "You have no shared projects yet.")
            return
        dialog = ProjectListDialog(projects, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.project_id():
            return
        try:
            remote = client.get_project(dialog.project_id())
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._open_directory(remote["kind"], remote["id"])
        if self.project and self.project.state.get("project_id") == remote["id"] and remote.get("head_id"):
            self.pull()
        elif self.project and self.project.state.get("project_id") == remote["id"]:
            self._report_manifest(remote)

    def invite(self) -> None:
        client = self._client()
        if client is None or self.project is None or not self.project.state.get("project_id"):
            warn(self, "Publish the project before inviting someone.")
            return
        email, ok = QInputDialog.getText(self, "Invite", "Email")
        if not ok or not email.strip():
            return
        role, ok = QInputDialog.getItem(self, "Invite", "Role", ["helper", "admin"], 0, False)
        if not ok:
            return
        try:
            result = client.invite(self.project.state["project_id"], email.strip(), role)
        except SyncError as exc:
            warn(self, str(exc))
            return
        info(self, "They have access." if result["status"] == "joined" else "Invite saved until they create an account.")

    def _report_manifest(self, remote: dict) -> None:
        if self.project is None:
            return
        local = {row["path"]: row for row in self.project.file_manifest()}
        missing = []
        different = []
        for row in remote.get("manifest", {}).get("files", []):
            path = row["path"]
            if path not in local:
                missing.append(path)
            elif row.get("sha256") and row["sha256"] != local[path]["sha256"]:
                different.append(path)
        lines = []
        if missing:
            lines.append("Missing here:\n" + "\n".join(missing[:20]))
        if different:
            lines.append("Different file contents:\n" + "\n".join(different[:20]))
        if lines:
            warn(self, "\n\n".join(lines))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.canvas.session is not None and self.canvas.session.dirty:
            answer = QMessageBox.question(
                self,
                "Save working copy?",
                "Save the working copy before closing? The committed masks stay as they are.",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self.save()
        super().closeEvent(event)


def _color_swatch(color: str | None) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(color or "#334155"))
    return QIcon(pixmap)


def _select_data(box: QComboBox, value) -> None:
    index = box.findData(value)
    box.setCurrentIndex(index if index >= 0 else 0)


def _action(text: str, slot, primary: bool = False, name: str = "") -> QPushButton:
    widget = QPushButton(text)
    widget.clicked.connect(slot)
    if name:
        widget.setObjectName(name)
    elif primary:
        widget.setObjectName("primary")
    return widget


_STYLE = """
QMainWindow, QWidget { background: #0F172A; color: #F8FAFC; font-family: "Segoe UI"; font-size: 13px; }
QLabel { background: transparent; }
QMenuBar { background: #1E293B; color: #F8FAFC; spacing: 4px; }
QMenuBar::item { background: transparent; padding: 6px 12px; }
QMenuBar::item:selected { background: #334155; }
QMenu { background: #1E293B; color: #F8FAFC; border: 1px solid #475569; padding: 4px; }
QMenu::item { padding: 6px 24px; }
QMenu::item:selected { background: #334155; }
QMenu::item:disabled { color: #94A3B8; }
QToolBar { background: #1E293B; border: none; spacing: 8px; padding: 6px 8px; }
QDockWidget { color: #F8FAFC; }
QDockWidget::title { background: #1E293B; padding: 6px; }
QLabel#hint { padding: 6px 12px; }
QGroupBox {
    background: #1E293B;
    border: 1px solid #475569;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #94A3B8; }
QPushButton, QToolButton {
    background: #334155;
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 28px;
}
QPushButton:hover, QToolButton:hover { background: #3B4C63; }
QPushButton:checked, QToolButton:checked { background: #22C55E; color: #052E16; border-color: #22C55E; }
QToolButton#erase:checked { background: #EF4444; color: #FFFFFF; border-color: #EF4444; }
QPushButton#primary { background: #22C55E; color: #052E16; border: none; font-weight: 700; }
QPushButton#primary:hover { background: #4ADE80; }
QListWidget, QComboBox, QSpinBox, QLineEdit {
    background: #0F172A;
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 4px 6px;
}
QListWidget::item { padding: 6px 4px; }
QListWidget::item:selected { background: #1D4ED8; color: #F8FAFC; }
QLabel#hint, QLabel#account, QGroupBox::title { color: #94A3B8; }
QLabel#section { font-size: 16px; font-weight: 700; color: #F8FAFC; }
QWidget#side { background: #1E293B; border: 1px solid #475569; border-radius: 10px; }
QStatusBar { background: #1E293B; color: #94A3B8; }
"""


def _working_name(relpath: str) -> str:
    if relpath.endswith("_seg.npy"):
        return relpath[: -len(".npy")] + ".working.npy"
    if relpath == "annotation_meta.json":
        return "annotation_meta.working.json"
    return relpath + ".working"
