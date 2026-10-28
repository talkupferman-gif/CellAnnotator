# CellAnnotator

Desktop app for drawing cell masks by hand. Images stay on each computer. A small server stores accounts, published projects, and committed masks, so several people can work from the same local image folder.

The app opens on a gallery of published projects. Open is what shows the image and the drawing tools.

## Install and start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m cellannotator
```

Python 3.11 or newer. The install includes Cellpose 3, which is what Run model uses. Cellpose 4 weight files are refused: the saved lab models are Cellpose 3 files.

On this computer the app starts its own account server at `http://127.0.0.1:8765` when that address is free. Accounts and the local project database live under `%APPDATA%\CellAnnotator`. Passwords are at least 8 characters. The server process lasts as long as the app is open.

## A project

Sign in from the home page. The gallery lists only projects that were published. A folder that was never published does not appear there.

Click a project for its name, description, creator, and the owner and admins. Helpers can be invited and can commit, and they are not listed on that page. Open asks for the image folder the first time on this computer and remembers the path. File → My projects returns to the gallery.

To share a folder you already have open: Version → Publish. You name it and write the description. Then Version → Commit when the masks should become the shared version.

Each image keeps two mask files beside it, in Cellpose `*_seg.npy` form:

- `*_seg.working.npy` is the copy you are editing. File → Save writes only that file.
- `*_seg.npy` is the shared version. It changes when an owner or admin commit lands.

A score from 1 to 10 is stored inside the mask file. 1 means barely segmented and sorts to the top of Image → Images and ranking. 10 means done. Z-stack links live in `annotation_meta.json` next to the images. The app remembers the project in `.cellannotator` inside the folder.

## People

The person who publishes a project is its only owner. The owner and admins invite by email and choose admin or helper. They cannot create a second owner.

| | Owner | Admin | Helper |
| --- | --- | --- | --- |
| Edit masks and open the images | yes | yes | yes |
| Invite, and edit the description | yes | yes | |
| Commit straight onto the shared version | yes | yes | |
| Lock, unlock, and see the change history | yes | | |

A locked project stays in the gallery. Only the owner can unlock it. While it is locked, nobody can commit, including the owner. Open still shows the images and the current masks.

## Drawing

The toolbar is Draw, Erase, Merge, Delete, Go back, and Redo. Everything else is in the menu.

Draw and Erase use the same gesture. Right-click, or a two-finger tap, then move the pointer without holding the button. The stroke counts only when the pointer returns to where it started. Escape cancels it. Enter does not finish a stroke.

- Draw creates a cell from the inside of that loop. Existing cells keep their pixels.
- Erase removes pixels inside the loop. If cells are selected, only those cells lose pixels. If nothing is selected, every cell inside the loop is cut.
- Left-drag pans. The wheel zooms. Image → Fit frames the image.
- Click selects a cell. Shift-click adds to the selection, then Merge combines those cells.
- Cells → Split keeps one cell until you press it, then breaks disconnected pieces. The largest piece keeps the number.
- Delete asks before it removes the selected cells.
- Go back and Redo cover the recent mask edits on this image.

X hides the masks so you see the image alone. Z shows outlines. The usual view is a colored fill with no outline. Image → Layer colors sets a color for each channel in the folder and can hide one. Those colors are reused when you pick channels for a model.

## Z-stacks

When you open a folder, mark it as unrelated images or a z-stack. A z-stack is a folder of separate images in filename order.

Select one mask on a slice and press Cells → Same cell, then do that on another slice. That links the two masks. It does not fill the gap. Cells → Fill writes the slices between those anchors into each slice's mask file. Editing a filled slice keeps the edit, marks it hand-drawn, and does not refill the neighbors.

## Models

Model → Run model… runs Cellpose `cyto3`, `nuclei`, or a weights file you choose. The result replaces the working masks on this image, or on a batch. The shared file changes only when someone commits.

For a custom model, set the main channel (the stain to segment) and the nuclei channel (optional). Each choice shows that channel's color in the open image. Gray and None use the whole image. Training is not in this version.

## Commits

An owner or admin commit updates the shared version after a confirmation. The project details, visible to the owner, list who committed, when, and for each image how many cells were added, deleted, or kept their number but changed shape.

A helper commit does not replace that shared version. It is stored as its own branch. When an owner or admin opens the project, the app says a helper committed and which images changed. They can load those masks into the working copy, edit them, and commit them to the shared version, or discard the branch. Discard removes that branch's masks from the server. It does not delete the image files, and it leaves the helper's local working copy as it is.

If two people change different images, both changes are kept. If they change the same image, the commit asks Use mine or Use theirs for that file. Version → History can load an older snapshot into the working copy. The shared latest stays where it is until someone commits again.

Version → Pull brings in the shared masks. A file you have edited locally is left alone, so a later commit can still merge it.

## A lab server

One always-on computer can run only the server. Desktops then point Sign in at that address. The annotation window stays on each person's computer, and the images stay in their folders.

```powershell
.\.venv\Scripts\python.exe -m cellannotator.server --host 0.0.0.0 --port 8765 --data server_data
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Windows installer

`packaging/build_windows.ps1` builds an app folder with PyInstaller. That build does not bundle Cellpose. Run models from an environment where Cellpose 3 is installed.
