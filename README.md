# CellAnnotator

Desktop app for drawing cell masks on a local image folder. People share a project through a small server. Images stay on each computer. A commit publishes the masks.

The mask file next to an image is Cellpose's `*_seg.npy`. While you work, the app writes `*_seg.working.npy`. **Save** updates that working file. **Commit** is what replaces the shared `*_seg.npy`.

## Run the app

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m cellannotator
```

Segmentation also needs Cellpose, which is optional until you run a model:

```powershell
.\.venv\Scripts\python.exe -m pip install cellpose
```

## Run the server

On the lab machine:

```powershell
.\.venv\Scripts\python.exe -m cellannotator.server --host 0.0.0.0 --port 8765 --data server_data
```

In the app, Log in, set the server address, then Publish a folder or Open shared. The default address is `http://127.0.0.1:8765`.

## Drawing

Draw and Erase use the same gesture. Turn Erase on when the stroke should remove pixels instead of creating a cell.

- Right-click, or a two-finger tap, then move the pointer without holding the button.
- Draw creates a cell only when the pointer returns to the start circle. Escape cancels it.
- Erase rubs pixels out along that path. Right-click again to stop. Escape stops and keeps what was already removed.
- The brush size is the slider, or `,` and `.` while the image has focus.
- If a cell is selected, only that cell is erased. Otherwise the cell under the start of the stroke is erased.
- Left-drag moves the image. The wheel zooms.
- Shift-click selects extra cells. Merge combines the selection. Split breaks a selected cell that has disconnected pieces. Delete removes the selected cells.
- A new cell does not cover an existing one.

## Z-stacks and scores

When you open a folder, mark it as unrelated images or a z-stack. A z-stack is separate image files in filename order.

Same cell links the selected mask to the cell you are building. Press it again on another slice to add that mask. Fill writes the slices between those masks. Editing a filled slice keeps your edit and does not redraw the neighbors.

Each image has a score from 1 to 10. 1 means little of it is segmented, and those images sort to the top. 10 means it is done.

## Models

Run uses Cellpose `cyto3`, `nuclei`, or a weights file you pick. Run batch does the same for several images. The result replaces the working masks. The committed file changes only when you commit. Training is not in this version; `ModelRunner.run` is the place to add it later.

## Sharing

A commit is the whole project. If two people change different images, both changes are kept. If they change the same image, the commit asks Use mine or Use theirs for that image. History can load an older commit into the working copy without moving the shared latest.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Windows installer

`packaging/build_windows.ps1` builds an app folder with PyInstaller. Cellpose stays installed in the environment you run the model from; the first build of the installer does not bundle it.
