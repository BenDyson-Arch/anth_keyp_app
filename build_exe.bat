@echo off
setlocal

echo Building keypoint_labeler.exe with PyInstaller...
py -m PyInstaller --noconfirm --clean --onefile --windowed --name keypoint_labeler --collect-all skimage --collect-all shapely --collect-all matplotlib keypoint_labeler.py

echo.
echo Build complete.
echo The executable will be at dist\keypoint_labeler.exe
echo Place full_annotations.geojson next to the exe before running it.
