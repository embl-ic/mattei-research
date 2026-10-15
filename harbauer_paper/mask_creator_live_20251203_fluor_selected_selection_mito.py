#!/usr/bin/env python3

# =============================================================================
# COPYRIGHT, ATTRIBUTION, AND LICENSE NOTICE
# =============================================================================
#
# Copyright (c) 2026 Mattei Team, Imaging Centre and Molecular Systems Biology Unit,
# European Molecular Biology Laboratory (EMBL).
#
# SOFTWARE DEVELOPMENT:
# This script was developed by the Mattei Team at the Imaging Centre and Molecular Systems Biology Unit,
# European Molecular Biology Laboratory (EMBL), with the assistance of
# Artificial Intelligence (AI) tools for code development, optimization,
# and documentation.
#
# OPEN-SOURCE AVAILABILITY:
# This software is released as open-source software under the MIT License
# to support scientific research, reproducibility, and collaboration.
#
# Users are permitted to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of this software, subject to the terms
# of the MIT License provided below.
#
# ATTRIBUTION AND CITATION:
# When using, modifying, adapting, or redistributing this software,
# users must retain the original copyright and license notices as
# required by the MIT License.
#
# When this software contributes to scientific publications,
# presentations, or other research outputs, users are kindly requested
# to acknowledge the original developers as:
#
# "Mattei Team, Imaging Centre and Molecular Systems Biology Unit,
#  European Molecular Biology Laboratory (EMBL)."
#
# Scientific citation and acknowledgment are strongly encouraged
# to recognize the contributions of the original developers.
#
# =============================================================================
# MIT LICENSE
# =============================================================================
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# =============================================================================
from qtpy.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QLabel, QCheckBox,QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem
)
from qtpy.QtCore import Qt, QTimer, Signal
from napari.qt.threading import thread_worker
import math
import os
import numpy as np
import mrcfile
import napari
from scipy.ndimage import label
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from pathlib import Path
import re
#----------------------Tomo Mask Generator Widget---------------------
def make_tilted_slab_mask(
    nx: int = 682,
    ny: int = 682,
    nz: int = 292,
    thickness: int = 20,
    center_z: float = None,
    tilt_xz_deg: float = 0.0,  # rotation about Y (tilt in XZ)
    tilt_yz_deg: float = 0.0,  # rotation about X (tilt in YZ)
    xy_margin: int = None,
    dtype=np.uint8,
):
    """
    Build a big rectangular slab (very large in X/Y, finite 'thickness' along its normal),
    tilt it in XZ and YZ, center it at (Xmid, Ymid, center_z), and rasterize into a (nz,ny,nx) mask.
    Array shape is (Z,Y,X) = (nz, ny, nx).
    """
    if center_z is None:
        center_z = (nz - 1) / 2.0

    if xy_margin is None:
        # Big enough to always cover; tweak if you like.
        xy_margin = max(nx, ny)  # one extra volume width on each side

    # Half-extents of the slab in its own coordinate frame.
    hx = max(nx, ny, nz) + xy_margin
    hy = max(nx, ny, nz) + xy_margin
    hz = thickness / 2.0

    # Precompute 2D grids for X and Y (to keep memory small)
    x_coords = np.arange(nx, dtype=np.float32)
    y_coords = np.arange(ny, dtype=np.float32)
    Xv, Yv = np.meshgrid(x_coords, y_coords, indexing='xy')  # shape (ny, nx)

    X0 = Xv - (nx - 1) / 2.0   # center in X at mid-volume
    Y0 = Yv - (ny - 1) / 2.0   # center in Y at mid-volume

    # Angles and their sines/cosines (radians)
    beta  = math.radians(tilt_xz_deg)  # about Y
    alpha = math.radians(tilt_yz_deg)  # about X

    cosb, sinb = math.cos(beta), math.sin(beta)
    cosa, sina = math.cos(alpha), math.sin(alpha)

    mask = np.zeros((nz, ny, nx), dtype=dtype)

    # Loop over slices to keep memory reasonable
    for z in range(nz):
        z0 = float(z) - float(center_z)

        # Compute the three local coordinates for this z-slice
        Xp =  cosb * X0 - sinb * z0
        Yp = (sinb * sina) * X0 + cosa * Y0 + (cosb * sina) * z0
        Zp = (sinb * cosa) * X0 - sina * Y0 + (cosb * cosa) * z0

        slab = (np.abs(Zp) <= hz) & (np.abs(Xp) <= hx) & (np.abs(Yp) <= hy)
        mask[z, :, :] = slab.astype(dtype)
    return mask


# ---- Qt Dock Widget ----------------------------------------------------------
class TiltedSlabWidget(QWidget):
    mask_updated = Signal(object)
    """Qt dock widget for generating a tilted slab mask (always shown as Labels)."""
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.setWindowTitle("Tilted slab mask (Labels)")
        self._last_mask = None
        form = QFormLayout()

        # Volume size
        form.addRow(self._row_label("Volume size"))
        self.nx = QSpinBox(); self.nx.setRange(1, 10000); self.nx.setValue(682)
        self.ny = QSpinBox(); self.ny.setRange(1, 10000); self.ny.setValue(682)
        self.nz = QSpinBox(); self.nz.setRange(1, 10000); self.nz.setValue(292)
        form.addRow("nx", self.nx)
        form.addRow("ny", self.ny)
        form.addRow("nz", self.nz)

        # Slab geometry
        form.addRow(self._row_label("Slab geometry"))
        self.thickness = QSpinBox(); self.thickness.setRange(1, 10000); self.thickness.setValue(120)
        self.thickness.setSingleStep(1)
        form.addRow("thickness (px)", self.thickness)

        # center_z (always explicit; no checkbox)
        self.center_z = QDoubleSpinBox()
        self.center_z.setDecimals(3)
        self.center_z.setRange(-1e6, 1e6)
        self.center_z.setSingleStep(1)
        self.center_z.setValue((self.nz.value() - 1) / 2.0)
        form.addRow("center_z", self.center_z)

        # Tilts
        self.tilt_xz = QDoubleSpinBox(); self.tilt_xz.setRange(-180.0, 180.0); self.tilt_xz.setDecimals(3); self.tilt_xz.setValue(0.0)
        self.tilt_xz.setSingleStep(0.1)
        self.tilt_yz = QDoubleSpinBox(); self.tilt_yz.setRange(-180.0, 180.0); self.tilt_yz.setDecimals(3); self.tilt_yz.setValue(0.0)
        self.tilt_yz.setSingleStep(0.1)
        form.addRow("tilt_yz_deg (about X)", self.tilt_yz)
        form.addRow("tilt_xz_deg (about Y)", self.tilt_xz)

        # xy_margin (explicit)
        self.xy_margin = QSpinBox(); self.xy_margin.setRange(0, 100000); self.xy_margin.setValue(max(self.nx.value(), self.ny.value()))
        form.addRow("xy_margin", self.xy_margin)

        # Output
        form.addRow(self._row_label("Output"))
        # dtype is for internal mask creation; display will always be Labels
        self.dtype = QComboBox(); self.dtype.addItems(["uint8", "uint16", "bool"])
        form.addRow("compute dtype", self.dtype)

        self.layer_name = QComboBox(); self.layer_name.setEditable(True); self.layer_name.setEditText("Tilted slab mask")
        self._refresh_layer_name_choices()
        form.addRow("layer name", self.layer_name)

        # Live controls
        self.live = QCheckBox("Live update"); self.live.setChecked(True)
        self.update_delay = QSpinBox(); self.update_delay.setRange(0, 2000); self.update_delay.setValue(150)
        form.addRow("", self.live)
        form.addRow("Update delay (ms)", self.update_delay)

        # Generate button
        self.btn_generate = QPushButton("Generate mask")
        self.btn_generate.clicked.connect(self._start_generation_now)

        # USe mask button
        self._use_mask_state = False  # mirror of the checkbox state that outlives Qt object

        self.use_mask = QCheckBox("Use mask in pipeline")
        self.use_mask.setChecked(self._use_mask_state)
        self.use_mask.toggled.connect(self._on_use_mask_toggled)
        form.addRow("", self.use_mask)

        # Layout
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(self.btn_generate, alignment=Qt.AlignRight)

        # Debounce + job tracking
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._start_generation_now)
        self._current_job_id = 0

        # Wire live updates
        for w in [self.nx, self.ny, self.nz, self.thickness, self.center_z,
                  self.tilt_xz, self.tilt_yz, self.xy_margin]:
            w.valueChanged.connect(self._on_param_changed)
        self.dtype.currentTextChanged.connect(self._on_param_changed)

    # ---------- helpers ----------
    def _on_use_mask_toggled(self, checked: bool):
        self._use_mask_state = checked

    def get_mask(self):
        return self._last_mask if getattr(self, "_use_mask_state", False) else None

    def _row_label(self, text: str) -> QLabel:
        lab = QLabel(f"<b>{text}</b>")
        lab.setTextFormat(Qt.RichText)
        return lab

    def _refresh_layer_name_choices(self):
        suggested = "Tilted slab mask"
        names = [lyr.name for lyr in self.viewer.layers]
        if suggested not in names:
            self.layer_name.addItem(suggested)
        for n in names:
            self.layer_name.addItem(n)
        self.layer_name.setEditText(suggested)

    def _find_layer_by_name(self, name: str):
        for lyr in self.viewer.layers:
            if lyr.name == name:
                return lyr
        return None

    def _collect_params(self):
        dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "bool": np.bool_}
        return dict(
            nx=int(self.nx.value()),
            ny=int(self.ny.value()),
            nz=int(self.nz.value()),
            thickness=int(self.thickness.value()),
            center_z=float(self.center_z.value()),
            tilt_xz_deg=float(self.tilt_xz.value()),
            tilt_yz_deg=float(self.tilt_yz.value()),
            xy_margin=int(self.xy_margin.value()),
            dtype=dtype_map[self.dtype.currentText()],
        )

    # ---------- live update wiring ----------
    def _on_param_changed(self, *args):
        if self.live.isChecked():
            self._debounce.start(self.update_delay.value())

    def _start_generation_now(self):
        params = self._collect_params()
        self._current_job_id += 1
        job_id = self._current_job_id

        @thread_worker
        def _compute():
            return make_tilted_slab_mask(**params)

        worker = _compute()
        worker.returned.connect(lambda mask: self._apply_mask(mask, job_id))
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def _on_worker_error(self, exc):
        print("Error generating mask:", exc)

    # ---------- display always as Labels ----------
    def _show_as_labels(self, name: str, mask: np.ndarray):
        """
        Ensure display is a Labels layer:
        - convert to integer (uint8 for boolean, else uint32)
        - remove same-named non-Labels layer if present
        - update existing Labels layer in place when shape matches
        """
        if mask.dtype == np.bool_:
            mask_disp = mask.astype(np.uint8)
        else:
            mask_disp = mask.astype(np.uint32)

        existing = self._find_layer_by_name(name)
        if existing is not None and getattr(existing, "_type_string", "") != "labels":
            # Replace non-labels layer with labels
            try:
                self.viewer.layers.remove(existing)
            except Exception:
                pass
            existing = None

        if existing is not None and getattr(existing, "data", None) is not None and np.shape(existing.data) == mask_disp.shape:
            existing.data = mask_disp
            try:
                self.viewer.layers.selection = {existing}
            except Exception:
                pass
        else:
            lyr = self.viewer.add_labels(mask_disp, name=name)
            try:
                self.viewer.layers.selection = {lyr}
            except Exception:
                pass

    def _apply_mask(self, mask: np.ndarray, job_id: int):
        # Skip stale results
        if job_id != self._current_job_id:
            return

        self._last_mask = mask
        self.mask_updated.emit(mask if self._use_mask_state else None)

        name = (self.layer_name.currentText() or "Tilted slab mask").strip()
        self._show_as_labels(name, mask)

        if mask.ndim == 3:
            try:
                self.viewer.dims.set_point(0, mask.shape[0] // 2)
            except Exception:
                pass





#-----------------------End Of Tomo Mask Generator--------------------

#-----------------------Label Selection Widget--------------------
def _format_id_count(label_id: int, count: int) -> str:
    # e.g. "ID 7 — 153,204 px"
    return f"ID {label_id} — {count:,} px"


class SelectLabelsWidget(QWidget):
    """
    Napari dock widget to select subset of labels from an input Labels layer.

    - Choose a Labels layer (default: 'sorted_labels')
    - See list of label IDs (+ pixel counts), filter via search
    - Check the IDs you want to keep
    - Press 'Create filtered labels' to create a new Labels layer keeping only those IDs
      (others set to 0)

    Pipeline access:
      - widget.get_filtered_labels()  -> np.ndarray or None (if toggle off or nothing created yet)
      - widget.filtered_labels_updated (Signal(object)) -> emits np.ndarray or None on creation
    """
    filtered_labels_updated = Signal(object)  # np.ndarray OR None

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.setWindowTitle("Select & Filter Labels")

        # -------- internal state (Qt-safe mirrors) --------
        self._use_filtered_state = True
        self._last_filtered = None
        self._all_ids = np.array([], dtype=np.int64)
        self._all_counts = np.array([], dtype=np.int64)

        # -------- UI --------
        form = QFormLayout()

        # Layer picker
        self.layer_combo = QComboBox()
        self._refresh_layer_combo()
        form.addRow("Labels layer", self.layer_combo)

        # Optional: target layer name for output
        self.output_name = QLineEdit("filtered_labels")
        form.addRow("Output layer name", self.output_name)

        # Search/filter box
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter IDs (e.g., '12', '1-5', '7,10,15-18')")
        form.addRow("Search / ranges", self.search)

        # IDs list (checkable)
        self.list_ids = QListWidget()
        self.list_ids.setSelectionMode(QListWidget.ExtendedSelection)
        form.addRow(QLabel("<b>Available label IDs</b>"))
        form.addRow(self.list_ids)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_check_all = QPushButton("Check all")
        self.btn_uncheck_all = QPushButton("Uncheck all")
        self.btn_invert = QPushButton("Invert")
        btn_row.addWidget(self.btn_check_all)
        btn_row.addWidget(self.btn_uncheck_all)
        btn_row.addWidget(self.btn_invert)
        form.addRow(btn_row)

        # Live debounce + sampling (optional)
        live_row = QHBoxLayout()
        self.live_counts = QCheckBox("Auto-refresh IDs")
        self.live_counts.setChecked(True)
        self.refresh_delay = QSpinBox()
        self.refresh_delay.setRange(0, 2000)
        self.refresh_delay.setValue(150)
        live_row.addWidget(self.live_counts)
        live_row.addWidget(QLabel("Delay (ms)"))
        live_row.addWidget(self.refresh_delay)
        form.addRow(live_row)

        # Use in pipeline toggle (controls get_filtered_labels() & signal payload)
        self.use_in_pipeline = QCheckBox("Use filtered labels in pipeline")
        self.use_in_pipeline.setChecked(self._use_filtered_state)
        self.use_in_pipeline.toggled.connect(self._on_use_toggled)
        form.addRow("", self.use_in_pipeline)

        # Create button
        self.btn_create = QPushButton("Create filtered labels")
        form.addRow(self.btn_create)

        # Layout
        root = QVBoxLayout(self)
        root.addLayout(form)

        # -------- timers & connections --------
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._compute_unique_in_background)

        # Wiring
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        self.search.textChanged.connect(self._apply_filter_to_list)
        self.btn_check_all.clicked.connect(lambda: self._set_all_checked(True))
        self.btn_uncheck_all.clicked.connect(lambda: self._set_all_checked(False))
        self.btn_invert.clicked.connect(self._invert_checks)
        self.btn_create.clicked.connect(self._create_filtered_labels)

        # Initial populate
        self._schedule_refresh()

    # -------------------- Public API --------------------
    def get_filtered_labels(self):
        """Return latest filtered labels (np.ndarray) or None if disabled/none yet."""
        return self._last_filtered if getattr(self, "_use_filtered_state", False) else None

    # -------------------- Internal helpers --------------------
    def _on_use_toggled(self, checked: bool):
        self._use_filtered_state = checked

    def _refresh_layer_combo(self):
        self.layer_combo.clear()
        names = []
        for lyr in self.viewer.layers:
            # only Labels layers
            if getattr(lyr, "_type_string", "") == "labels":
                names.append(lyr.name)
        # Prefer "sorted_labels" if present
        if "sorted_labels" in names:
            names.remove("sorted_labels")
            names.insert(0, "sorted_labels")
        self.layer_combo.addItems(names)

    def _get_selected_labels_layer(self):
        name = self.layer_combo.currentText().strip()
        if not name:
            return None
        for lyr in self.viewer.layers:
            if lyr.name == name:
                return lyr
        return None

    def _schedule_refresh(self):
        if self.live_counts.isChecked():
            self._debounce.start(self.refresh_delay.value())
        else:
            self._compute_unique_in_background()

    def _on_layer_changed(self, _):
        self._schedule_refresh()

    def _compute_unique_in_background(self):
        layer = self._get_selected_labels_layer()
        if layer is None:
            self._set_id_list([], [])
            return

        data = layer.data  # numpy integer array
        # Compute unique IDs (excluding 0) in a worker
        @thread_worker
        def _compute():
            ids, counts = np.unique(data, return_counts=True)
            # Drop background 0
            m = ids != 0
            return ids[m].astype(np.int64, copy=False), counts[m].astype(np.int64, copy=False)

        worker = _compute()
        worker.returned.connect(lambda tup: self._on_unique_ready(*tup))
        worker.errored.connect(lambda e: print("Unique compute error:", e))
        worker.start()

    def _on_unique_ready(self, ids: np.ndarray, counts: np.ndarray):
        self._all_ids = ids
        self._all_counts = counts
        self._populate_list(ids, counts)
        self._apply_filter_to_list()  # respect current search text

    def _set_id_list(self, ids, counts):
        self._all_ids = np.asarray(ids, dtype=np.int64)
        self._all_counts = np.asarray(counts, dtype=np.int64)
        self._populate_list(self._all_ids, self._all_counts)

    def _populate_list(self, ids: np.ndarray, counts: np.ndarray):
        self.list_ids.blockSignals(True)
        self.list_ids.clear()
        for _id, cnt in zip(ids, counts):
            item = QListWidgetItem(_format_id_count(int(_id), int(cnt)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, int(_id))  # store the raw label id
            self.list_ids.addItem(item)
        self.list_ids.blockSignals(False)

    def _apply_filter_to_list(self):
        """Filter visible items by search string; also tick ranges if typed (optional UX)."""
        text = self.search.text().strip()
        # Parse ranges like "1-5,7,10-12"
        targets = self._parse_ranges(text) if text else None

        for i in range(self.list_ids.count()):
            item = self.list_ids.item(i)
            label_id = item.data(Qt.UserRole)
            visible = True
            if text:
                # visible if it matches text OR in parsed targets
                visible = (str(label_id).find(text) != -1) or (targets and label_id in targets)
            item.setHidden(not visible)

        # Optionally, auto-check ranges the user typed:
        if targets:
            for i in range(self.list_ids.count()):
                item = self.list_ids.item(i)
                if item.data(Qt.UserRole) in targets:
                    item.setCheckState(Qt.Checked)

    @staticmethod
    def _parse_ranges(s: str):
        """
        Turn "1-3,7,10-12" into a set{1,2,3,7,10,11,12}.
        Ignores malformed tokens.
        """
        result = set()
        for tok in re.split(r"[,\s]+", s):
            if not tok:
                continue
            if "-" in tok:
                try:
                    a, b = tok.split("-")
                    a, b = int(a), int(b)
                    if a <= b:
                        result.update(range(a, b + 1))
                    else:
                        result.update(range(b, a + 1))
                except Exception:
                    pass
            else:
                try:
                    result.add(int(tok))
                except Exception:
                    pass
        return result

    def _set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list_ids.count()):
            self.list_ids.item(i).setCheckState(state)

    def _invert_checks(self):
        for i in range(self.list_ids.count()):
            item = self.list_ids.item(i)
            item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def _gather_checked_ids(self):
        ids = []
        for i in range(self.list_ids.count()):
            item = self.list_ids.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(int(item.data(Qt.UserRole)))
        return np.array(sorted(ids), dtype=np.int64)

    def _create_filtered_labels(self):
        layer = self._get_selected_labels_layer()
        if layer is None:
            print("No Labels layer selected.")
            self._emit_filtered(None)
            return

        keep_ids = self._gather_checked_ids()
        if keep_ids.size == 0:
            print("No IDs checked → emitting None and not creating a layer.")
            self._emit_filtered(None)
            return

        data = layer.data
        # Build filtered labels: keep selected IDs, others -> 0
        # Fast with np.isin
        mask = np.isin(data, keep_ids)
        filtered = np.where(mask, data, 0).astype(np.uint32, copy=False)

        # Create or update target layer
        out_name = (self.output_name.text().strip() or "filtered_labels")
        existing = self._find_layer_by_name(out_name)
        if existing is not None and getattr(existing, "_type_string", "") != "labels":
            # replace non-labels
            try: self.viewer.layers.remove(existing)
            except Exception: pass
            existing = None

        if existing is not None and getattr(existing, "data", None) is not None and np.shape(existing.data) == filtered.shape:
            existing.data = filtered
        else:
            self.viewer.add_labels(filtered, name=out_name)

        # Store & emit for pipeline
        self._last_filtered = filtered  # store the actual array
        self._emit_filtered(filtered)

    def _emit_filtered(self, arr):
        payload = arr if self._use_filtered_state else None
        self.filtered_labels_updated.emit(payload)

    def _find_layer_by_name(self, name: str):
        for lyr in self.viewer.layers:
            if lyr.name == name:
                return lyr
        return None

#-----------------End Of Label Selection Widget--------------------


def make_tilted_slab_mask_old(
    nx: int = 682,
    ny: int = 682,
    nz: int = 292,
    thickness: int = 20,
    center_z: float = None,
    tilt_xz_deg: float = 0.0,  # rotation about Y (tilt in XZ)
    tilt_yz_deg: float = 0.0,  # rotation about X (tilt in YZ)
    xy_margin: int = None,
    dtype=np.uint8,
):
    """
    Build a big rectangular slab (very large in X/Y, finite 'thickness' along its normal),
    tilt it in XZ and YZ, center it at (Xmid, Ymid, center_z), and rasterize into a (nz,ny,nx) mask.

    Notes on conventions:
      - Array shape for MRC is (Z,Y,X) = (nz, ny, nx).
      - We treat 'height' as center_z (in pixel index units, 0..nz-1).
      - Positive tilt_xz_deg rotates about +Y (right-handed), i.e., tilts the slab in the XZ plane.
      - Positive tilt_yz_deg rotates about +X (right-handed), i.e., tilts the slab in the YZ plane.
      - The slab is huge in X and Y: half-widths default to max(nx, ny, nz) + xy_margin.
    """
    if center_z is None:
        center_z = (nz - 1) / 2.0

    if xy_margin is None:
        # Big enough to always cover; tweak if you like.
        xy_margin = max(nx, ny)  # one extra volume width on each side

    # Half-extents of the slab in its own coordinate frame.
    hx = max(nx, ny, nz) + xy_margin
    hy = max(nx, ny, nz) + xy_margin
    hz = thickness / 2.0

    # Precompute 2D grids for X and Y (to keep memory small)
    x_coords = np.arange(nx, dtype=np.float32)
    y_coords = np.arange(ny, dtype=np.float32)

    Xv, Yv = np.meshgrid(x_coords, y_coords, indexing='xy')  # shape (ny, nx)

    X0 = Xv - (nx - 1) / 2.0   # center in X at mid-volume
    Y0 = Yv - (ny - 1) / 2.0   # center in Y at mid-volume

    # Angles and their sines/cosines (radians)
    beta  = math.radians(tilt_xz_deg)  # about Y
    alpha = math.radians(tilt_yz_deg)  # about X

    cosb, sinb = math.cos(beta), math.sin(beta)
    cosa, sina = math.cos(alpha), math.sin(alpha)

    # We want local slab coords u = R^T * v, where:
    #   R = R_y(beta) @ R_x(alpha)
    # So R^T =
    # [[ cosb,      0,        -sinb ],
    #  [ sinb*sina, cosa,     cosb*sina ],
    #  [ sinb*cosa, -sina,    cosb*cosa ]]
    #
    # Given v = [x0, y0, z0]^T, compute:
    # x' =  cosb * x0             - sinb * z0
    # y' = (sinb*sina) * x0 + cosa * y0 + (cosb*sina) * z0
    # z' = (sinb*cosa) * x0 - sina * y0 + (cosb*cosa) * z0

    mask = np.zeros((nz, ny, nx), dtype=dtype)

    # Loop over slices to keep memory reasonable
    for z in range(nz):
        z0 = float(z) - float(center_z)

        # Compute the three local coordinates for this z-slice
        Xp =  cosb * X0 - sinb * z0
        Yp = (sinb * sina) * X0 + cosa * Y0 + (cosb * sina) * z0
        Zp = (sinb * cosa) * X0 - sina * Y0 + (cosb * cosa) * z0

        slab = (np.abs(Zp) <= hz) & (np.abs(Xp) <= hx) & (np.abs(Yp) <= hy)
        mask[z, :, :] = slab.astype(dtype)
    return mask

def save_mask_mrc(mask, out_path):
    # Write MRC: mrcfile expects shape (Z, Y, X)
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(mask)

def load_tomogram(tomogram_path):
    with mrcfile.open(tomogram_path, permissive=True) as mrc:
        tomo = mrc.data.copy()
    return tomo

def load_segmentation(segmentation_path):
    with mrcfile.open(segmentation_path, permissive=True) as mrc:
        data = mrc.data.copy()
    return data

def component_printer(labeled_sorted):
    max_val = np.max(labeled_sorted)
    i = 1
    while i < max_val + 1:
        print(f"component {i}: size: {np.sum(labeled_sorted == i)}")
        i += 1

def component_sorter(segmentation):
    binary_data = (segmentation > 0).astype(np.int32)
    labeled, num_features = label(binary_data)
    sizes = np.bincount(labeled.ravel())
    K = len(sizes) - 1
    if K <= 0:
        raise ValueError("No foreground components found.")
    # Indices 0..K-1 correspond to labels 1..K
    idx = np.arange(K)
    # Stable descending sort by size via negative sizes + mergesort (stable)
    order_idx = np.argsort(-sizes[1:], kind="mergesort")  # values in 0..K-1
    order_labels = order_idx + 1  # convert to labels 1..K
    # --- Step 6: Build mapping (0 stays 0; largest -> 1; ...; smallest -> K) ---
    mapping = np.zeros(K + 1, dtype=np.int32)  # length = max_label + 1
    mapping[order_labels] = np.arange(1, K + 1, dtype=np.int32)
    # Apply mapping
    labeled_sorted = mapping[labeled]
    return labeled_sorted


def label_saver(labeled_sorted,
                output_path):
    with mrcfile.new(output_path, overwrite=True) as mrc_out:
        mrc_out.set_data(labeled_sorted.astype(np.int8))

def threshold_finder(data):
    diffs = [data[i] - data[i + 1] for i in range(len(data) - 1)]
    # Find the position of the largest gap
    max_gap_index = diffs.index(max(diffs))
    threshold = (data[max_gap_index] + data[max_gap_index + 1]) / 2
    print("Threshold:", threshold)

def plot_component_sizes(segmentation,title = None):
    """
    Plot histogram/bar chart of component sizes in a relabeled segmentation.
    Background (label 0) is ignored.
    """
    # Count pixels per label
    counts = np.bincount(segmentation.ravel())
    if len(counts) <= 1:
        print("No components found (only background).")
        return
    # Skip background count (index 0)
    labels = np.arange(1, len(counts))
    sizes = counts[1:]
    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(labels, sizes)
    plt.xlabel("Component label")
    plt.xticks(labels)
    plt.ylabel("Size (pixels)")
    if title is None:
        plt.title("Component sizes")
    else:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def loaded_segmentation_sorter(loaded_segmentation,
                               threshold_size):
    labeled_sorted = component_sorter(loaded_segmentation)
    labeled_sorted = remove_small_objects_skimage(segmentation=labeled_sorted,
                                                  size=threshold_size)
    return labeled_sorted

def remove_small_objects_skimage(segmentation, size):
    # skimage's remove_small_objects supports labeled images if `mask is None`
    cleaned = remove_small_objects(segmentation, min_size=size, connectivity=1)
    relabeled, _, _ = relabel_sequential(cleaned)  # makes labels contiguous
    return relabeled

def get_roots_stripping_suffix(directory, suffix = "_mito.mrc"):
    roots = []
    for f in Path(directory).iterdir():
        if f.is_file() and f.name.endswith(suffix):
            root = f.name.removesuffix(suffix)
            roots.append(root)
    return roots

def fuse_to_one(segmentation: np.ndarray, labels_to_fuse) -> np.ndarray:
    labels_to_fuse = np.asarray(list(labels_to_fuse), dtype=segmentation.dtype)
    mask = np.isin(segmentation, labels_to_fuse)
    return mask.astype(np.uint8)  # 1 for fused, 0 otherwise

def input_labels(prompt="Enter labels separated by space (to keep. They will be merged. (k) if no fusion is needed): "):
    raw = input(prompt)
    if raw.strip().lower() == 'k':
        return ["k"]
    else:
        return [int(x) for x in raw.split()]

def sort_roots(name_core, roots):
    """
    Sort roots like 'corr_L_pace_<NN>_ts_<XXX>' numerically by NN and XXX.
    """
    def key_fn(root):
        # Extract the two numbers (NN and XXX)
        m = re.search(rf"{name_core}_(\d+)_ts_(\d+)", root)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (float('inf'), float('inf'))  # fallback if pattern missing

    return sorted(roots, key=key_fn)

def napari_viewer_old(tomogram,
                  segmentation,
                  tomo_mask=None,
                  tomogram_name="tomogram",
                  segmentation_name="labels_sorted",
                  tomo_mask_name="tomo_mask",
                  tomo_visible = True,
                  segmentation_visible = True,
                  tomo_mask_visible = True):
    viewer = napari.Viewer()
    viewer.add_image(tomogram, name=tomogram_name, visible=tomo_visible)
    viewer.add_labels(segmentation, name=segmentation_name, visible=segmentation_visible)
    if tomo_mask is not None:
        viewer.add_labels(tomo_mask, name=tomo_mask_name, visible=tomo_mask_visible)
    viewer.dims.ndisplay = 2
    napari.run()


def napari_viewer_tomo_mask_widget(tomo_loaded,
                                   labeled_sorted):
    viewer = napari.Viewer()
    viewer.add_image(tomo_loaded, name='tomo', visible=True)
    viewer.add_labels(labeled_sorted, name='seg', visible=True)
    w = TiltedSlabWidget(viewer)
    viewer.window.add_dock_widget(w, area="right")
    napari.run()
    mask = w.get_mask()
    return mask

def napari_viewer_selection_widget(tomo_loaded,
                                   labeled_sorted):
    viewer = napari.Viewer()
    viewer.add_image(tomo_loaded, name='tomo', visible=True)
    viewer.add_labels(labeled_sorted, name="sorted_labels")
    w = SelectLabelsWidget(viewer)
    viewer.window.add_dock_widget(w, area="right")
    napari.run()
    filtered_labeled_sorted = w.get_filtered_labels()  # returns np.ndarray or None (respects toggle)
    return filtered_labeled_sorted

def binarize_labels(labels: np.ndarray) -> np.ndarray:
    return (labels > 0).astype(np.uint8)

def main_function_widgets(membrain_segmentation,
                      isonet_tomogram,
                      tomo_root,
                      threshold_size=10000,
                      binarize = True,
                      via_tomo_mask_generation = True,
                      generate_tomo_mask_only = False):

    tomo = load_tomogram(tomogram_path=isonet_tomogram)
    segmentation = load_segmentation(segmentation_path=membrain_segmentation)
    filtered_labeled_sorted = None

    #1. Tomo Mask Creation
    if via_tomo_mask_generation:
        labeled_sorted = loaded_segmentation_sorter(loaded_segmentation=segmentation,
                                                    threshold_size=threshold_size)
        plot_component_sizes(labeled_sorted, tomo_root)
        tomo_mask = napari_viewer_tomo_mask_widget(tomo,labeled_sorted)
        if tomo_mask is not None:
            labeled_sorted = tomo_mask*labeled_sorted
            labeled_sorted = loaded_segmentation_sorter(loaded_segmentation=labeled_sorted,
                                                        threshold_size=threshold_size)
        else:
            print("No tomo mask needed")
    else:
        labeled_sorted = loaded_segmentation_sorter(loaded_segmentation=segmentation,
                                                    threshold_size=threshold_size)
        tomo_mask = None
    # 2. Selection part
    if not generate_tomo_mask_only:
        filtered_labeled_sorted = napari_viewer_selection_widget(tomo,
                                                                 labeled_sorted)
    if filtered_labeled_sorted is not None:
        if binarize:
            filtered_labeled_sorted = binarize_labels(filtered_labeled_sorted)
    else:
        print("No labels layer selected.")
    return tomo_mask, labeled_sorted, filtered_labeled_sorted

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_corr_roots(name_core, directory):
    """
    Return unique roots like 'corr_L_pace_<NN>_ts_<XXX>' for files whose names
    start with that pattern, ignoring any suffixes/extensions.
    """
    # look-ahead: next char is '_' or '.' or end-of-string
    pat = re.compile(rf"^({name_core}_(\d+)_ts_(\d+))(?=(?:[_\.]|$))", re.IGNORECASE)

    roots, seen = [], set()
    for p in Path(directory).iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            root = m.group(1)
            if root not in seen:
                seen.add(root)
                roots.append(root)
    return roots
if __name__ == "__main__":
    dataset_id = 5
    tomo_id = 10
    via_tomo_mask_generation = True
    generate_tomo_mask_only = False

    if dataset_id == 5:
        name_core = 'mmm'
        path_to_reconstruction_dir = "/run/user/1000/gvfs/smb-share:server=mattei.embl.de,share=icem/bekbayev/selected_fluor"

    organelle_suffix = "_mito"
    organelle_folder_name = 'mito'
    existing_mito_folder = os.path.join(path_to_reconstruction_dir, 'masks', organelle_folder_name)

    if dataset_id == 5:
        if tomo_id == 1:
            tomo_roots_list = ['mmm142_ts_006']
            threshold_value = '-3.0'
        if tomo_id == 2:
            tomo_roots_list = ['mmm145_ts_002']
            threshold_value = '-2.0'
        if tomo_id == 3:
            tomo_roots_list = ['mmm156_ts_004']
            threshold_value = '-3.0'
        if tomo_id == 4:
            tomo_roots_list = ['mmm159_ts_005']
            threshold_value = '-3.0'
        if tomo_id == 5:
            tomo_roots_list = ['mmm164_ts_005']
            threshold_value = '-2.0'
        if tomo_id == 6:
            tomo_roots_list = ['mmm165_ts_007']
            threshold_value = '-2.0'
        if tomo_id == 7:
            tomo_roots_list = ['mmm167_ts_003']
            threshold_value = '-3.0'
        if tomo_id == 8:
            tomo_roots_list = ['mmm167_ts_004']
            threshold_value = '-2.0'
        if tomo_id == 9:
            tomo_roots_list = ['mmm175_ts_007']
            threshold_value = '-3.0'
        if tomo_id == 10:
            tomo_roots_list = ['mmm177_ts_003']
            threshold_value = '0.0'
        membrain_suffix = f'_3round.mrc'
        tomo_suffix = '_14.34Apx_corrected.mrc'

    membrain_dir = os.path.join(path_to_reconstruction_dir, 'masks', '3round')
    isonet_tomo_dir = os.path.join(path_to_reconstruction_dir, 'isonet/corrected_tomos')

    results_folder = os.path.join(path_to_reconstruction_dir, 'masks')

    ensure_dir_exists(results_folder)
    organelle_folder = os.path.join(results_folder, organelle_folder_name)
    ensure_dir_exists(organelle_folder)

    if dataset_id == 4:
        already_processed = get_roots_stripping_suffix(os.path.join(path_to_reconstruction_dir, 'masks'),
                                                       "_tomomoask.mrc")

    for tomo_root in tomo_roots_list:
        seg_path = os.path.join(membrain_dir, tomo_root + membrain_suffix)
        tomo_file = os.path.join(isonet_tomo_dir, tomo_root + tomo_suffix)


        tomo_mask, labeled_sorted, fused_labels = main_function_widgets(seg_path,
                                                                        tomo_file,
                                                                        tomo_root,
                                                                        via_tomo_mask_generation=via_tomo_mask_generation,
                                                                        generate_tomo_mask_only = generate_tomo_mask_only)
        ###outputs
        tomo_mask_out = os.path.join(results_folder, tomo_root  + '_tomomoask' + '.mrc')
        labeled_sorted_out = os.path.join(results_folder, tomo_root + '_sorted_label' + '.mrc')
        fused_labels_out = os.path.join(organelle_folder, tomo_root + organelle_suffix + '.mrc')
        ###outputs end

        # WRITING
        if tomo_mask is not None:
            save_mask_mrc(tomo_mask, tomo_mask_out)
            print(f"Wrote {tomo_mask_out}")
        else:
            print(f"{tomo_root}: Tomo mask is not needed")
        label_saver(labeled_sorted, labeled_sorted_out)
        print(f"Wrote {labeled_sorted_out}")

        if fused_labels is not None:
            label_saver(fused_labels, fused_labels_out)
            print(f"Wrote {fused_labels_out}")
        else:
            print(f"{tomo_root}: No fused labels needed")

