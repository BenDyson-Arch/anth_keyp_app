"""keypoint_labeler.py -- PyQt5 tool for labeling anatomical keypoints on
rasterised anthropomorphic figures from the Injalak polygon dataset.

Controls
--------
  Left-click  : place current keypoint, auto-advances to next stage
  Right-click : remove the keypoint nearest the cursor
  Tab         : skip current limb group (leg1 / leg2 / arm1 / arm2)
  Space / ->  : save & advance to next figure
  <-          : save & go to previous figure
  U           : undo last placed point (restores previous stage)
  Delete      : clear all keypoints on current figure
  S           : force save
  X / Y       : toggle flip along X / Y axis

Output
------
  keypoint_labels.json  -- { shape_id: { kp_name: [x, y], ... }, ... }
  Coordinates are in RASTER_SIZE x RASTER_SIZE pixel space (448 x 448).

Requires: PyQt5  (conda install pyqt)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from skimage.draw import polygon as sk_polygon

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import (
        QBrush,
        QColor,
        QFont,
        QImage,
        QPainter,
        QPen,
        QPixmap,
    )
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    sys.exit("PyQt5 not found.  Install:  conda install pyqt")

from utils import (
    convert_geojson_to_cartesian,
    extract_cats_from_geojson,
    pca_reduce_to_2d,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _runtime_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_input_path(filename):
    app_path = os.path.join(APP_DIR, filename)
    cwd_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(app_path):
        return app_path
    if os.path.exists(cwd_path):
        return cwd_path
    return app_path


APP_DIR = _runtime_dir()
GEOJSON_FILENAME = "full_annotations.geojson"
ANTHRO_CATEGORIES = [
    "anthropomorph",
    "anthropomorph unknown",
    "anthropomorph male",
    "anthropomorph female",
    "anthropomorph other",
]

RASTER_SIZE = 448
DISPLAY_SCALE = 2  # display at 896 x 896 px
BORDER = 10
DOT_RADIUS = 6  # keypoint dot radius in display pixels
SAVE_PATH = os.path.join(APP_DIR, "keypoint_labels.json")

# Guided annotation stages: (storage_key, group, display_label)
STAGES = [
    ("head", "head", "Head"),
    ("neck", "torso", "Neck"),
    ("torso_mid", "torso", "Torso mid"),
    ("pelvis", "torso", "Pelvis"),
    ("leg1_knee", "leg1", "Leg 1 - knee"),
    ("leg1_foot", "leg1", "Leg 1 - foot"),
    ("leg2_knee", "leg2", "Leg 2 - knee"),
    ("leg2_foot", "leg2", "Leg 2 - foot"),
    ("arm1_shoulder", "arm1", "Arm 1 - shoulder"),
    ("arm1_hand", "arm1", "Arm 1 - hand"),
    ("arm2_shoulder", "arm2", "Arm 2 - shoulder"),
    ("arm2_hand", "arm2", "Arm 2 - hand"),
]

# Groups that can be skipped -- value is the first stage index of the next group
SKIP_MAP = {
    "leg1": 6,  # -> leg2_knee
    "leg2": 8,  # -> arm1_shoulder
    "arm1": 10,  # -> arm2_shoulder
    "arm2": 12,  # -> end (save & next figure)
}

# Skeleton connections drawn as lines between placed keypoints
CONNECTIONS = [
    ("head", "neck"),
    ("neck", "torso_mid"),
    ("torso_mid", "pelvis"),
    ("pelvis", "leg1_knee"),
    ("leg1_knee", "leg1_foot"),
    ("pelvis", "leg2_knee"),
    ("leg2_knee", "leg2_foot"),
    ("neck", "arm1_shoulder"),
    ("arm1_shoulder", "arm1_hand"),
    ("neck", "arm2_shoulder"),
    ("arm2_shoulder", "arm2_hand"),
]

KP_COLOURS = {
    "head": "#FF4444",
    "neck": "#FF9900",
    "torso_mid": "#CC88FF",
    "pelvis": "#FF44CC",
    "leg1_knee": "#44AAFF",
    "leg1_foot": "#88FF88",
    "leg2_knee": "#2255FF",
    "leg2_foot": "#228833",
    "arm1_shoulder": "#FFDD00",
    "arm1_hand": "#0088FF",
    "arm2_shoulder": "#AAEE00",
    "arm2_hand": "#4444FF",
}

# ---------------------------------------------------------------------------
# Rasterisation (SVD-upright orientation)
# ---------------------------------------------------------------------------


def _orient_upright(pts):
    """Return SVD-rotated, head-at-top copy of pts (N x 2)."""
    c = pts.mean(axis=0)
    p = pts - c
    _, _, Vt = np.linalg.svd(p, full_matrices=False)
    vx, vy = float(Vt[0, 0]), float(Vt[0, 1])
    R = np.array([[vy, -vx], [vx, vy]])
    p = p @ R.T
    # Resolve 180-deg flip: narrower end -> top (head)
    hi_thresh = np.percentile(p[:, 1], 75)
    lo_thresh = np.percentile(p[:, 1], 25)
    top_pts = p[p[:, 1] >= hi_thresh]
    bot_pts = p[p[:, 1] <= lo_thresh]
    top_w = top_pts[:, 0].std() if len(top_pts) > 1 else 0.0
    bot_w = bot_pts[:, 0].std() if len(bot_pts) > 1 else 0.0
    if bot_w < top_w:
        p = -p
    return p


def rasterise_upright(pts, size=RASTER_SIZE):
    """Return filled uint8 (size x size) mask, SVD-upright oriented."""
    p = _orient_upright(pts)
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = max((hi - lo).max(), 1e-9)
    s = (size - 2 * BORDER) / span
    p = p * s
    lo, hi = p.min(axis=0), p.max(axis=0)
    col = p[:, 0] - lo[0] + (size - (hi[0] - lo[0])) / 2.0
    row = p[:, 1] - lo[1] + (size - (hi[1] - lo[1])) / 2.0
    row = (size - 1.0) - row  # flip: +Y -> top
    mask = np.zeros((size, size), dtype=np.uint8)
    rr, cc = sk_polygon(row, col, shape=(size, size))
    mask[rr, cc] = 255
    return mask


def render_pixmap(
    mask, kp_dict, stage=0, scale=DISPLAY_SCALE, flip_x=False, flip_y=False
):
    """Compose mask + skeleton lines + keypoint dots into a QPixmap."""
    h, w = mask.shape
    m = mask
    if flip_x:
        m = np.fliplr(m)
    if flip_y:
        m = np.flipud(m)
    m = np.ascontiguousarray(m)

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[m > 0] = (210, 210, 210)
    rgb[m == 0] = (28, 28, 28)
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg).scaled(
        w * scale,
        h * scale,
        Qt.KeepAspectRatio,
        Qt.FastTransformation,
    )

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    def _disp(x, y):
        dx = (RASTER_SIZE - 1 - x) if flip_x else x
        dy = (RASTER_SIZE - 1 - y) if flip_y else y
        return int(dx * scale), int(dy * scale)

    # --- skeleton lines (draw first, under dots) ---
    painter.setPen(QPen(QColor("#888888"), 2, Qt.SolidLine))
    for a, b in CONNECTIONS:
        if a in kp_dict and b in kp_dict:
            ax, ay = _disp(*kp_dict[a])
            bx, by = _disp(*kp_dict[b])
            painter.drawLine(ax, ay, bx, by)

    # --- keypoint dots ---
    r = DOT_RADIUS
    font = QFont("Arial", 7)
    painter.setFont(font)
    for name, (x, y) in kp_dict.items():
        px, py = _disp(x, y)
        colour = QColor(KP_COLOURS.get(name, "#FFFFFF"))
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(colour))
        painter.drawEllipse(px - r, py - r, r * 2, r * 2)
        painter.setPen(QPen(Qt.white, 1))
        painter.drawText(px + r + 2, py + r // 2 + 4, name)

    # --- crosshair for current stage (if not yet placed) ---
    if stage < len(STAGES):
        cur_key = STAGES[stage][0]
        if cur_key not in kp_dict:
            c = QColor(KP_COLOURS.get(cur_key, "#FFFFFF"))
            c.setAlpha(160)
            pen = QPen(c, 1, Qt.DashLine)
            painter.setPen(pen)
            mid = (RASTER_SIZE // 2) * scale
            painter.drawLine(mid, 0, mid, h * scale)
            painter.drawLine(0, mid, w * scale, mid)

    painter.end()
    return pixmap


# ---------------------------------------------------------------------------
# Canvas widget (handles mouse events)
# ---------------------------------------------------------------------------


class ImageCanvas(QLabel):
    def __init__(self, left_cb, right_cb, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setFixedSize(RASTER_SIZE * DISPLAY_SCALE, RASTER_SIZE * DISPLAY_SCALE)
        self._left_cb = left_cb
        self._right_cb = right_cb

    def mousePressEvent(self, event):
        x = max(0, min(RASTER_SIZE - 1, event.x() // DISPLAY_SCALE))
        y = max(0, min(RASTER_SIZE - 1, event.y() // DISPLAY_SCALE))
        if event.button() == Qt.LeftButton:
            self._left_cb(x, y)
        elif event.button() == Qt.RightButton:
            self._right_cb(x, y)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LabelerWindow(QMainWindow):
    def __init__(self, figures, masks):
        super().__init__()
        self.figures = figures
        self.masks = masks
        self.idx = 0
        self.stage = 0  # index into STAGES
        self.labels = {}  # shape_id -> {kp_key: [x, y]}
        self._undo = []  # (shape_id, prev_stage, kp_key, prev_value_or_None)
        self.flip_x = False
        self.flip_y = False

        if os.path.exists(SAVE_PATH):
            with open(SAVE_PATH) as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} existing labels from {SAVE_PATH}")

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Anthropomorph Keypoint Labeler")
        self.setMinimumSize(
            max(960, RASTER_SIZE * DISPLAY_SCALE + 320),
            max(780, RASTER_SIZE * DISPLAY_SCALE + 80),
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Left: canvas + controls
        left = QVBoxLayout()
        self.canvas = ImageCanvas(self._on_left_click, self._on_right_click)
        left.addWidget(self.canvas)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev  [<-]")
        self.btn_next = QPushButton("Next >  [Space]")
        self.btn_skip_fig = QPushButton("Skip fig (no save)")
        self.btn_clear = QPushButton("Clear All  [Del]")
        for b in (self.btn_prev, self.btn_next, self.btn_skip_fig, self.btn_clear):
            nav.addWidget(b)
        left.addLayout(nav)

        flip_row = QHBoxLayout()
        self.btn_flip_x = QPushButton("Flip X  [X]")
        self.btn_flip_y = QPushButton("Flip Y  [Y]")
        self.btn_flip_x.setCheckable(True)
        self.btn_flip_y.setCheckable(True)
        self.btn_flip_x.setStyleSheet("QPushButton:checked { background: #885500; }")
        self.btn_flip_y.setStyleSheet("QPushButton:checked { background: #885500; }")
        flip_row.addWidget(self.btn_flip_x)
        flip_row.addWidget(self.btn_flip_y)
        left.addLayout(flip_row)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 11px;")
        left.addWidget(self.info_label)
        layout.addLayout(left)

        # Right: stage list + actions
        right = QVBoxLayout()
        right.setSpacing(6)

        hdr = QLabel("<b>Annotation stages</b>")
        hdr.setStyleSheet("font-size: 12px;")
        right.addWidget(hdr)

        hint = QLabel("L-click: place   R-click: remove\nClick a stage to jump to it")
        hint.setStyleSheet("font-size: 10px; color: #888888;")
        right.addWidget(hint)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        right.addWidget(line)

        self.kp_list = QListWidget()
        self.kp_list.setFixedWidth(210)
        mono = QFont("Monospace", 10)
        for key, group, label in STAGES:
            item = QListWidgetItem("  " + label)
            item.setForeground(QColor(KP_COLOURS[key]))
            item.setFont(mono)
            self.kp_list.addItem(item)
        self.kp_list.currentRowChanged.connect(self._on_stage_click)
        right.addWidget(self.kp_list)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        right.addWidget(line2)

        self.btn_skip_group = QPushButton("Skip group  [Tab]")
        self.btn_skip_group.setEnabled(False)
        right.addWidget(self.btn_skip_group)

        self.btn_undo = QPushButton("Undo  [U]")
        right.addWidget(self.btn_undo)

        self.btn_save = QPushButton("Save  [S]")
        right.addWidget(self.btn_save)

        right.addStretch()

        self.saved_label = QLabel("Saved: 0 figures")
        self.saved_label.setStyleSheet("font-size: 11px; color: #44AA44;")
        right.addWidget(self.saved_label)

        layout.addLayout(right)

        self.btn_prev.clicked.connect(lambda: self._navigate(-1))
        self.btn_next.clicked.connect(lambda: self._navigate(+1))
        self.btn_skip_fig.clicked.connect(lambda: self._navigate(+1, save=False))
        self.btn_clear.clicked.connect(self._clear_current)
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_save.clicked.connect(self._save)
        self.btn_skip_group.clicked.connect(self._skip_group)
        self.btn_flip_x.clicked.connect(self._toggle_flip_x)
        self.btn_flip_y.clicked.connect(self._toggle_flip_y)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _shape_id(self):
        return self.figures[self.idx]["shape_id"]

    def _kp(self):
        return self.labels.setdefault(self._shape_id(), {})

    def _stage_key(self):
        return STAGES[self.stage][0] if self.stage < len(STAGES) else None

    def _stage_group(self):
        return STAGES[self.stage][1] if self.stage < len(STAGES) else None

    # ------------------------------------------------------------------
    # Refresh display
    # ------------------------------------------------------------------

    def _refresh(self):
        mask = self.masks[self.idx]
        kp = self._kp()
        self.canvas.setPixmap(
            render_pixmap(
                mask, kp, stage=self.stage, flip_x=self.flip_x, flip_y=self.flip_y
            )
        )
        self._refresh_list()
        sid = self._shape_id()
        name = self.figures[self.idx].get("name", "")
        stage_lbl = STAGES[self.stage][2] if self.stage < len(STAGES) else "Done"
        self.info_label.setText(
            f"Figure {self.idx + 1}/{len(self.figures)}   id: {sid[:12]}   "
            f"cat: {name}   kp: {len(kp)}   next: {stage_lbl}"
        )
        self.saved_label.setText(f"Saved: {len(self.labels)} figures labeled")

    def _refresh_list(self):
        kp = self._kp()
        group = self._stage_group()
        for i, (key, grp, label) in enumerate(STAGES):
            item = self.kp_list.item(i)
            tick = "v " if key in kp else "  "
            item.setText(tick + label)
            item.setBackground(
                QColor(60, 55, 10) if i == self.stage else QColor(0, 0, 0, 0)
            )
        self.kp_list.blockSignals(True)
        self.kp_list.setCurrentRow(min(self.stage, len(STAGES) - 1))
        self.kp_list.blockSignals(False)
        # Skip button label
        group_names = {
            "leg1": "Leg 1",
            "leg2": "Leg 2",
            "arm1": "Arm 1",
            "arm2": "Arm 2",
        }
        if group in SKIP_MAP:
            self.btn_skip_group.setText(f"Skip {group_names[group]}  [Tab]")
            self.btn_skip_group.setEnabled(True)
        else:
            self.btn_skip_group.setText("Skip group  [Tab]")
            self.btn_skip_group.setEnabled(False)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_stage_click(self, row):
        if 0 <= row < len(STAGES):
            self.stage = row
            self._refresh()

    def _on_left_click(self, x, y):
        if self.stage >= len(STAGES):
            return
        sx = (RASTER_SIZE - 1 - x) if self.flip_x else x
        sy = (RASTER_SIZE - 1 - y) if self.flip_y else y
        key = self._stage_key()
        kp = self._kp()
        self._undo.append((self._shape_id(), self.stage, key, kp.get(key)))
        kp[key] = [sx, sy]
        self.stage += 1
        if self.stage >= len(STAGES):
            self._save()
            self._navigate(+1)
            return
        self._refresh()

    def _on_right_click(self, x, y):
        kp = self._kp()
        if not kp:
            return
        sx = (RASTER_SIZE - 1 - x) if self.flip_x else x
        sy = (RASTER_SIZE - 1 - y) if self.flip_y else y
        name, pos = min(
            kp.items(),
            key=lambda kv: (kv[1][0] - sx) ** 2 + (kv[1][1] - sy) ** 2,
        )
        self._undo.append((self._shape_id(), self.stage, name, pos))
        del kp[name]
        self._refresh()

    def _undo_last(self):
        if not self._undo:
            return
        sid, prev_stage, kp_key, prev = self._undo.pop()
        if sid != self._shape_id():
            return
        kp = self.labels.setdefault(sid, {})
        if prev is None:
            kp.pop(kp_key, None)
        else:
            kp[kp_key] = prev
        self.stage = prev_stage
        self._refresh()

    def _clear_current(self):
        self.labels.pop(self._shape_id(), None)
        self._undo.clear()
        self.stage = 0
        self._refresh()

    def _skip_group(self):
        group = self._stage_group()
        if group in SKIP_MAP:
            self.stage = SKIP_MAP[group]
            if self.stage >= len(STAGES):
                self._save()
                self._navigate(+1)
                return
        self._refresh()

    def _toggle_flip_x(self):
        self.flip_x = self.btn_flip_x.isChecked()
        self._refresh()

    def _toggle_flip_y(self):
        self.flip_y = self.btn_flip_y.isChecked()
        self._refresh()

    def _navigate(self, delta, save=True):
        if save:
            self._save()
        self.idx = (self.idx + delta) % len(self.figures)
        self.stage = 0
        self.flip_x = False
        self.flip_y = False
        self._undo.clear()
        self.btn_flip_x.setChecked(False)
        self.btn_flip_y.setChecked(False)
        self._refresh()

    def _save(self):
        with open(SAVE_PATH, "w") as f:
            json.dump(self.labels, f, indent=2)
        self._refresh()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Right, Qt.Key_Space):
            self._navigate(+1)
        elif k == Qt.Key_Left:
            self._navigate(-1)
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self._clear_current()
        elif k == Qt.Key_U:
            self._undo_last()
        elif k == Qt.Key_S:
            self._save()
        elif k == Qt.Key_Tab:
            self._skip_group()
        elif k == Qt.Key_X:
            self.flip_x = not self.flip_x
            self.btn_flip_x.setChecked(self.flip_x)
            self._refresh()
        elif k == Qt.Key_Y:
            self.flip_y = not self.flip_y
            self.btn_flip_y.setChecked(self.flip_y)
            self._refresh()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._save()
        event.accept()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_figures():
    geojson_path = _resolve_input_path(GEOJSON_FILENAME)
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(
            f"Could not find {GEOJSON_FILENAME} in {APP_DIR} or {os.getcwd()}"
        )

    print("Loading GeoJSON...")
    print(f"  Source: {geojson_path}")
    features = extract_cats_from_geojson(geojson_path, ANTHRO_CATEGORIES)
    print(f"  {len(features)} features loaded")

    features = convert_geojson_to_cartesian(features)
    features = pca_reduce_to_2d(features)

    print("Using all anthropomorph figures (completeness filter disabled)...")
    print(f"  Retained {len(features)} / {len(features)} figures")

    print("Rasterising...")
    masks = [rasterise_upright(f["cartesian_2d"]) for f in features]
    return features, masks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    figures, masks = load_figures()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LabelerWindow(figures, masks)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
