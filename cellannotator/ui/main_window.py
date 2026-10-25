from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellannotator.images import display_rgb, for_model, page_count
from cellannotator.masks import delete_labels, merge_labels, split_label
from cellannotator.meta import load_meta, meta_paths
from cellannotator.project import Project
from cellannotator.segment import ModelRunner
from cellannotator.settings import Settings
from cellannotator.sync_client import SyncClient, SyncError
from cellannotator.ui.canvas import Canvas
from cellannotator.ui.dialogs import (
    BatchDialog,
    ConflictDialog,
    HistoryDialog,
    LoginDialog,
    ProjectListDialog,
    ask_kind,
    info,
    read_zip,
    warn,
)
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
                    for_model(image),
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
        self.project: Project | None = None
        self.current_rel: str | None = None
        self.active_link_id: str | None = None
        self.scores: dict[str, int | None] = {}
        self.weights_path: str | None = None
        self._thread: ModelThread | None = None
        self._showing_score = False
        self._build()
        self._account_label()

    def _build(self) -> None:
        self.canvas = Canvas()
        self.canvas.edited.connect(self._on_edited)
        self.canvas.labelsTouched.connect(self._on_labels)
        self.canvas.status.connect(self.statusBar().showMessage)

        file_row = QHBoxLayout()
        for text, slot in (
            ("Open folder", self.open_folder),
            ("Open shared", self.open_shared),
            ("Save", self.save),
            ("Publish", self.publish),
            ("Commit", self.commit),
            ("Pull", self.pull),
            ("History", self.show_history),
            ("Invite", self.invite),
            ("Log in", self.login),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            file_row.addWidget(button)
        file_row.addStretch(1)
        self.account = QLabel("Not logged in")
        file_row.addWidget(self.account)

        tool_row = QHBoxLayout()
        self.draw_button = QToolButton()
        self.draw_button.setText("Draw")
        self.draw_button.setCheckable(True)
        self.draw_button.setChecked(True)
        self.erase_button = QToolButton()
        self.erase_button.setText("Erase")
        self.erase_button.setCheckable(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.draw_button)
        group.addButton(self.erase_button)
        self.draw_button.clicked.connect(lambda: self.canvas.set_mode("draw"))
        self.erase_button.clicked.connect(lambda: self.canvas.set_mode("erase"))
        tool_row.addWidget(self.draw_button)
        tool_row.addWidget(self.erase_button)
        tool_row.addWidget(QLabel("Brush"))
        self.brush = QSlider(Qt.Horizontal)
        self.brush.setRange(1, 80)
        self.brush.setValue(8)
        self.brush.setFixedWidth(120)
        self.brush.valueChanged.connect(self._set_brush)
        tool_row.addWidget(self.brush)
        for text, slot in (
            ("Merge", self.merge),
            ("Split", self.split_cells),
            ("Delete", self.delete_selected),
            ("Same cell", self.same_cell),
            ("Finish linking", self.finish_linking),
            ("Fill", self.fill),
            ("Fit", self.canvas.fit),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            tool_row.addWidget(button)
            if text in {"Same cell", "Finish linking", "Fill"}:
                button.setObjectName(text)
        tool_row.addStretch(1)

        model_row = QHBoxLayout()
        self.model = QComboBox()
        self.model.addItems(["cyto3", "nuclei", "Custom weights"])
        self.weights_button = QPushButton("Weights file")
        self.weights_button.clicked.connect(self.pick_weights)
        self.auto_diameter = QPushButton("Auto diameter")
        self.auto_diameter.setCheckable(True)
        self.auto_diameter.setChecked(True)
        self.diameter = QSpinBox()
        self.diameter.setRange(1, 2000)
        self.diameter.setValue(30)
        self.diameter.setEnabled(False)
        self.auto_diameter.toggled.connect(lambda on: self.diameter.setEnabled(not on))
        self.cyto = QSpinBox()
        self.nucleus = QSpinBox()
        for box in (self.cyto, self.nucleus):
            box.setRange(0, 3)
        model_row.addWidget(QLabel("Model"))
        model_row.addWidget(self.model)
        model_row.addWidget(self.weights_button)
        model_row.addWidget(self.auto_diameter)
        model_row.addWidget(self.diameter)
        model_row.addWidget(QLabel("Cyto"))
        model_row.addWidget(self.cyto)
        model_row.addWidget(QLabel("Nucleus"))
        model_row.addWidget(self.nucleus)
        run = QPushButton("Run")
        run.clicked.connect(self.run_current)
        batch = QPushButton("Run batch")
        batch.clicked.connect(self.run_batch)
        model_row.addWidget(run)
        model_row.addWidget(batch)
        self.channel = QComboBox()
        for label, data in (("Gray", "gray"), ("Red", "r"), ("Green", "g"), ("Blue", "b")):
            self.channel.addItem(label, data)
        self.channel.currentIndexChanged.connect(self._change_channel)
        model_row.addWidget(self.channel)
        self.fill_toggle = QPushButton("Mask fill")
        self.fill_toggle.setCheckable(True)
        self.fill_toggle.setChecked(True)
        self.outline_toggle = QPushButton("Borders")
        self.outline_toggle.setCheckable(True)
        self.outline_toggle.setChecked(True)
        self.fill_toggle.toggled.connect(self._toggle_fill)
        self.outline_toggle.toggled.connect(self._toggle_outlines)
        model_row.addWidget(self.fill_toggle)
        model_row.addWidget(self.outline_toggle)
        model_row.addStretch(1)

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
        side.addWidget(QLabel("Images"))
        side.addWidget(self.list, 1)
        side.addWidget(QLabel("Ready (1 needs work, 10 is done)"))
        side.addWidget(self.score)
        side.addWidget(QLabel("Sort"))
        side.addWidget(self.sort)
        side.addWidget(self.link_label)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMinimumWidth(280)

        splitter = QSplitter()
        splitter.addWidget(side_widget)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        hint = QLabel(
            "Right-click, then move the pointer. Draw becomes a cell when you return to the start. "
            "Erase rubs pixels out along that same path; right-click again to stop. "
            "Left-drag moves the image. The wheel zooms. Shift-click adds to the selection. "
            "Escape cancels a draw."
        )
        hint.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(file_row)
        layout.addLayout(tool_row)
        layout.addLayout(model_row)
        layout.addWidget(splitter, 1)
        layout.addWidget(hint)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Open a folder of images")

        save_action = QAction(self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save)
        self.addAction(save_action)
        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        self.addAction(undo_action)

    def _set_brush(self, value: int) -> None:
        self.canvas.brush_radius = value
        self.statusBar().showMessage(f"Brush {value}px")

    def _toggle_fill(self, on: bool) -> None:
        self.canvas.show_fill = on
        self.canvas.refresh()

    def _toggle_outlines(self, on: bool) -> None:
        self.canvas.show_outlines = on
        self.canvas.refresh()

    def _account_label(self) -> None:
        if self.settings.email and self.settings.token:
            self.account.setText(self.settings.email)
        else:
            self.account.setText("Not logged in")

    def _client(self) -> SyncClient | None:
        if not self.settings.token:
            warn(self, "Log in first.")
            return None
        return SyncClient(self.settings.server_url, self.settings.token)

    def open_folder(self) -> None:
        self._open_directory(None, None)

    def _open_directory(self, kind: str | None, project_id: str | None) -> None:
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

    def _set_stack_tools(self) -> None:
        enabled = bool(self.project and self.project.meta.kind == "zstack")
        for name in ("Same cell", "Finish linking", "Fill"):
            button = self.findChild(QPushButton, name)
            if button is not None:
                button.setEnabled(enabled)

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
        self.canvas.set_image(session, display_rgb(array, self.channel.currentData()))
        self._apply_link_colors()
        self._showing_score = True
        self.score.setCurrentIndex(0 if session.readiness is None else session.readiness)
        self._showing_score = False
        self._show_link_status()
        self.canvas.setFocus()
        if session.warning:
            self.statusBar().showMessage(session.warning)
        path = self.project.image_path(relpath)
        if page_count(path) > 1:
            self.statusBar().showMessage(f"{relpath}: this file has multiple pages. Showing the first page.")

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

    def _change_channel(self) -> None:
        if self.project is None or self.current_rel is None:
            return
        rgb = display_rgb(self.project.array_of(self.current_rel), self.channel.currentData())
        self.canvas.set_display(rgb)

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
            self.statusBar().showMessage(Path(path).name)

    def _model_args(self):
        name = self.model.currentText()
        weights = self.weights_path if name == "Custom weights" else None
        if name == "Custom weights" and not weights:
            warn(self, "Choose a weights file first.")
            return None
        diameter = None if self.auto_diameter.isChecked() else float(self.diameter.value())
        channels = [self.cyto.value(), self.nucleus.value()]
        model_name = "cyto3" if name == "Custom weights" else name
        return model_name, weights, diameter, channels

    def run_current(self) -> None:
        if self.current_rel is None:
            return
        self._start_model([self.current_rel])

    def run_batch(self) -> None:
        if self.project is None:
            return
        dialog = BatchDialog(self.project.rels, self.current_rel, self)
        if dialog.exec() != dialog.Accepted:
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
        if dialog.exec() != dialog.Accepted:
            return
        self.settings.server_url = dialog.server.text().strip() or self.settings.server_url
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
        self.statusBar().showMessage(f"Logged in as {self.settings.email}")

    def publish(self) -> None:
        client = self._client()
        if client is None or self.project is None:
            return
        name, ok = QInputDialog.getText(self, "Publish project", "Project name")
        if not ok or not name.strip():
            return
        self.statusBar().showMessage("Hashing images…")
        try:
            created = client.create_project(name.strip(), self.project.meta.kind, {"files": self.project.file_manifest()})
        except SyncError as exc:
            warn(self, str(exc))
            return
        self.project.state["project_id"] = created["id"]
        self.project.state["base_commit_id"] = None
        self.project.save_state()
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
        parent = self.project.state.get("base_commit_id")
        try:
            result = self._commit_with_conflicts(client, project_id, parent, files)
        except SyncError as exc:
            warn(self, str(exc))
            return
        self._apply_snapshot(client, project_id, result["id"], mode="committed")
        self.project.state["base_commit_id"] = result["id"]
        self.project.save_state()
        self.project.meta = load_meta(self.project.root)
        self._reload_current()
        self.statusBar().showMessage("Committed. The shared masks now match this working copy.")

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
            if dialog.exec() != dialog.Accepted:
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
        if dialog.exec() != dialog.Accepted or not dialog.chosen:
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
        if dialog.exec() != dialog.Accepted or not dialog.project_id():
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
        try:
            result = client.invite(self.project.state["project_id"], email.strip())
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


def _working_name(relpath: str) -> str:
    if relpath.endswith("_seg.npy"):
        return relpath[: -len(".npy")] + ".working.npy"
    if relpath == "annotation_meta.json":
        return "annotation_meta.working.json"
    return relpath + ".working"
