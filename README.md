# Anthropomorph Keypoint Labeler

A PyQt5 desktop app for labeling anatomical keypoints on rasterised anthropomorph figures from a GeoJSON dataset.

## What it does

- loads anthropomorph features from `full_annotations.geojson`
- keeps all anthropomorphs with completeness filtering disabled
- converts polygon coordinates to local cartesian space
- projects shapes to 2D
- rasterises each figure to a `448 x 448` mask
- lets you annotate key anatomical points in a guided sequence
- saves labels to `keypoint_labels.json`
- resumes on the first figure whose `shape_id` is not already in the saved labels file

## Current defaults

- raster size: `448 x 448`
- output labels file: `keypoint_labels.json`
- expected input dataset filename: `full_annotations.geojson`

The app looks for `full_annotations.geojson` next to the script or next to the built executable. If it is not found there, it also checks the current working directory.

## Controls

- `Left-click`: place current keypoint and advance
- `Right-click`: remove nearest keypoint
- `G`: skip the current stage or limb section
- `N` or `Right Arrow`: save and go to next figure
- `Left Arrow`: save and go to previous figure
- `U`: undo last placed or removed point
- `Delete` or `Backspace`: clear current figure labels
- `S`: save labels
- `X`: toggle horizontal flip
- `Y`: toggle vertical flip

## Annotation stages

The guided keypoint order is:

1. head
2. neck
3. torso mid
4. pelvis
5. leg 1 knee
6. leg 1 foot
7. leg 2 knee
8. leg 2 foot
9. arm 1 elbow
10. arm 1 hand
11. arm 2 elbow
12. arm 2 hand

## Requirements

- Python 3.11 recommended
- the packages listed in `requirements.txt`

## Local setup

Create a virtual environment if you want to keep dependencies isolated.

Install dependencies:

```/dev/null/sh#L1-2
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Run the app:

```/dev/null/sh#L1-2
python3 keypoint_labeler.py
```

## Input data

Place `full_annotations.geojson` in the project directory before running the app locally.

If you build the Windows executable, place `full_annotations.geojson` next to `keypoint_labeler.exe` unless it is already included in the build artifact.

## Output data

The app writes:

- `keypoint_labels.json`

This file is written next to the script during local runs, or next to the executable in packaged runs.

## Windows executable build with GitHub Actions

This repository includes a GitHub Actions workflow at `.github/workflows/build-windows-exe.yml`.

It will:

- run on GitHub-hosted Windows
- install Python dependencies
- build `keypoint_labeler.exe` with `PyInstaller`
- upload the built files as an artifact named `keypoint_labeler-windows`

### How to use it

1. Push this repository to GitHub.
2. Open the repository on GitHub.
3. Go to the **Actions** tab.
4. Run the **Build Windows executable** workflow, or trigger it by pushing a commit.
5. Download the `keypoint_labeler-windows` artifact from the workflow run.

## Local Windows build

If you are on Windows, you can also build locally with:

```/dev/null/bat#L1-1
build_exe.bat
```

The built executable will be created at `dist\keypoint_labeler.exe`.

## Project files

- `keypoint_labeler.py` — main PyQt5 application
- `utils.py` — geometry, conversion, and helper functions
- `requirements.txt` — Python dependencies
- `build_exe.bat` — local Windows build script
- `.github/workflows/build-windows-exe.yml` — GitHub Actions workflow for Windows builds

## Notes

- this is a native desktop app, so packaged outputs are platform-specific
- GitHub Actions is used here to build a real Windows `.exe` without needing a local Windows machine
- the source code remains cross-platform even though build artifacts are OS-specific
