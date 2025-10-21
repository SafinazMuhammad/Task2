import sys
import os
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QFileDialog,
    QSlider, QStatusBar, QGroupBox, QLabel, QComboBox, QCheckBox, QMessageBox,
    QRadioButton, QButtonGroup, QSpinBox, QSizePolicy, QTabWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import measure
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, map_coordinates
import pydicom


# ==============================
# Utility helpers
# ==============================
def safe_percentiles(arr, p_low=1, p_high=99):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return 0.0, 1.0
    lo = np.percentile(a, p_low)
    hi = np.percentile(a, p_high)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def rotation_matrix_from_tilts(axial_deg, sagittal_deg, coronal_deg):
    """Build rotation: Rz(axial) @ Rx(sagittal) @ Ry(coronal)."""
    ax = np.deg2rad(axial_deg)
    sg = np.deg2rad(sagittal_deg)
    cr = np.deg2rad(coronal_deg)
    Rz = np.array([[ np.cos(ax), -np.sin(ax), 0],
                   [ np.sin(ax),  np.cos(ax), 0],
                   [ 0,           0,          1]])
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(sg), -np.sin(sg)],
                   [0,  np.sin(sg),  np.cos(sg)]])
    Ry = np.array([[ np.cos(cr), 0, np.sin(cr)],
                   [ 0,          1, 0         ],
                   [-np.sin(cr), 0, np.cos(cr)]])
    return Rz @ Rx @ Ry


# normalize helper
def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    return v / n


# ==============================
# Main widget
# ==============================
class DICOMMultiPlanarViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.dicom_data = None        # (Z, Y, X)
        self.mask_data = None         # same shape as dicom_data
        self.show_mask = False
        self._mesh_cache = None
        # voxel sizes in mm (z, y, x)
        self._voxel_sizes = (1.0, 1.0, 1.0)

        # crosshair indices [z, x, y]
        self.current_slices = [0, 0, 0]  # [axial(z), sagittal(x), coronal(y)]

        # ROI
        self.roi_mode = False
        self.roi_start = None
        self.roi_active_view = None
        self.roi_coords = None
        self.roi_slices = None

        # Ruler
        self.ruler_mode = False
        self.ruler_points = {}  # per view list of up to 2 points

        # Contrast/brightness
        self.contrast_mode = 'normal'
        self._base_lo = 0.0
        self._base_hi = 1.0

        # Oblique (independent tilts)
        self.fourth_window_mode = 'oblique'
        self.tilt_axial = 0.0     # around Z
        self.tilt_sagittal = 0.0  # around X
        self.tilt_coronal = 0.0   # around Y
        self.oblique_offset = 0.0  # along plane normal (in voxels)

        # Sectioning
        self.sectioning_position = 0.5  # 0..1

        # crosshair dragging
        self._dragging_crosshair_view = None

        # ===== NEW: oblique lines (Shift+Drag) per view =====
        # store as ((x1,y1),(x2,y2)) in view pixel coords
        self._oblique_lines = {'axial': None, 'sagittal': None, 'coronal': None}
        self._last_oblique_source = None  # which view provided the last valid line

        self.initUI()

    # ---------------------------
    # UI
    # ---------------------------
    def initUI(self):
        self.setWindowTitle("Multi-Planar DICOM Viewer — MPR (true cut) + Oblique line + 2D Sectioning")
        self.setGeometry(50, 50, 1800, 1000)

        main_layout = QHBoxLayout(self)

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 0)

        viewport_panel = self.create_viewport_panel()
        main_layout.addWidget(viewport_panel, 1)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        panel.setMaximumWidth(440)

        tabs = QTabWidget()

        tabs.addTab(self.create_data_roi_tab(), "Data & ROI")
        tabs.addTab(self.create_visualization_tab(), "Visualization")
        tabs.addTab(self.create_3d_sectioning_tab(), "3D & Sectioning")

        layout.addWidget(tabs)

        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        return panel

    def create_data_roi_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel('<b>═══ Data Loading ═══</b>'))
        self.load_dicom_btn = QPushButton('📂 Load DICOM/NIfTI Data')
        self.load_dicom_btn.clicked.connect(self.load_dicom)
        layout.addWidget(self.load_dicom_btn)

        self.load_mask_btn = QPushButton('📂 Load Mask (Optional)')
        self.load_mask_btn.clicked.connect(self.load_mask)
        self.load_mask_btn.setEnabled(False)
        layout.addWidget(self.load_mask_btn)

        self.mask_checkbox = QCheckBox('👁 Show Mask Overlay')
        self.mask_checkbox.setEnabled(False)
        self.mask_checkbox.stateChanged.connect(self.toggle_mask)
        layout.addWidget(self.mask_checkbox)

        layout.addWidget(QLabel('<b>═══ Auto ROI From Mask ═══</b>'))
        row = QHBoxLayout()
        row.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.setEnabled(False)
        row.addWidget(self.label_combo)
        self.auto_roi_btn = QPushButton("🎯 Auto ROI")
        self.auto_roi_btn.setEnabled(False)
        self.auto_roi_btn.clicked.connect(self.auto_roi_from_mask)
        row.addWidget(self.auto_roi_btn)
        layout.addLayout(row)

        layout.addWidget(QLabel('<b>═══ Manual ROI Selection ═══</b>'))
        layout.addWidget(QLabel('Select view to draw ROI:'))
        self.roi_view_group = QButtonGroup()
        self.roi_axial_radio = QRadioButton('Axial View')
        self.roi_sagittal_radio = QRadioButton('Sagittal View')
        self.roi_coronal_radio = QRadioButton('Coronal View')
        self.roi_axial_radio.setChecked(True)
        for r in (self.roi_axial_radio, self.roi_sagittal_radio, self.roi_coronal_radio):
            r.setEnabled(False)
            layout.addWidget(r)
            self.roi_view_group.addButton(r)

        self.roi_btn = QPushButton('✏️ Draw ROI Rectangle')
        self.roi_btn.setEnabled(False)
        self.roi_btn.clicked.connect(self.toggle_roi_mode)
        layout.addWidget(self.roi_btn)

        zrow = QHBoxLayout()
        self.zoom_in_btn = QPushButton("🔍+ Zoom In")
        self.zoom_in_btn.setEnabled(False)
        self.zoom_in_btn.clicked.connect(self.roi_zoom_in)
        zrow.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QPushButton("🔍- Zoom Out")
        self.zoom_out_btn.setEnabled(False)
        self.zoom_out_btn.clicked.connect(self.roi_zoom_out)
        zrow.addWidget(self.zoom_out_btn)
        layout.addLayout(zrow)

        layout.addWidget(QLabel('<b>═══ Slice Ranges ═══</b>'))
        # axial
        row = QHBoxLayout()
        row.addWidget(QLabel('Axial:'))
        self.axial_start_spin = QSpinBox(); self.axial_end_spin = QSpinBox()
        self.axial_start_spin.setPrefix('S: '); self.axial_end_spin.setPrefix('E: ')
        self.axial_start_spin.valueChanged.connect(self.update_manual_roi)
        self.axial_end_spin.valueChanged.connect(self.update_manual_roi)
        row.addWidget(self.axial_start_spin); row.addWidget(self.axial_end_spin)
        layout.addLayout(row)
        # sagittal
        row = QHBoxLayout()
        row.addWidget(QLabel('Sagittal:'))
        self.sagittal_start_spin = QSpinBox(); self.sagittal_end_spin = QSpinBox()
        self.sagittal_start_spin.setPrefix('S: '); self.sagittal_end_spin.setPrefix('E: ')
        self.sagittal_start_spin.valueChanged.connect(self.update_manual_roi)
        self.sagittal_end_spin.valueChanged.connect(self.update_manual_roi)
        row.addWidget(self.sagittal_start_spin); row.addWidget(self.sagittal_end_spin)
        layout.addLayout(row)
        # coronal
        row = QHBoxLayout()
        row.addWidget(QLabel('Coronal:'))
        self.coronal_start_spin = QSpinBox(); self.coronal_end_spin = QSpinBox()
        self.coronal_start_spin.setPrefix('S: '); self.coronal_end_spin.setPrefix('E: ')
        self.coronal_start_spin.valueChanged.connect(self.update_manual_roi)
        self.coronal_end_spin.valueChanged.connect(self.update_manual_roi)
        row.addWidget(self.coronal_start_spin); row.addWidget(self.coronal_end_spin)
        layout.addLayout(row)

        self.roi_info_label = QLabel('ROI: Not selected')
        self.roi_info_label.setWordWrap(True)
        self.roi_info_label.setStyleSheet(
            "background-color:#d32f2f;color:white;padding:8px;border-radius:4px;font-weight:bold;")
        layout.addWidget(self.roi_info_label)

        layout.addWidget(QLabel('<b>═══ Export Options ═══</b>'))
        self.export_all_btn = QPushButton('💾 Export All Views')
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(lambda: self.export_roi('all'))
        layout.addWidget(self.export_all_btn)

        row = QHBoxLayout()
        self.export_axial_btn = QPushButton('Ax')
        self.export_sagittal_btn = QPushButton('Sag')
        self.export_coronal_btn = QPushButton('Cor')
        for b, v in [(self.export_axial_btn,'axial'),(self.export_sagittal_btn,'sagittal'),(self.export_coronal_btn,'coronal')]:
            b.setEnabled(False)
            b.clicked.connect(lambda _, vv=v: self.export_roi(vv))
            row.addWidget(b)
        layout.addLayout(row)

        self.reset_btn = QPushButton('🔄 Reset View')
        self.reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(self.reset_btn)

        layout.addStretch()
        return tab

    def create_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel('<b>═══ Measurement Tool ═══</b>'))
        self.ruler_btn = QPushButton('📏 Ruler Tool (Click 2 points)')
        self.ruler_btn.setCheckable(True)
        self.ruler_btn.setEnabled(False)
        self.ruler_btn.clicked.connect(self.toggle_ruler_mode)
        layout.addWidget(self.ruler_btn)

        self.ruler_info_label = QLabel('Distance: N/A')
        self.ruler_info_label.setStyleSheet("background:#ffffff;color:#000;padding:6px;border:1px solid #ddd;")
        layout.addWidget(self.ruler_info_label)
        self.clear_ruler_btn = QPushButton('🗑️ Clear Measurements')
        self.clear_ruler_btn.setEnabled(False)
        self.clear_ruler_btn.clicked.connect(self.clear_ruler)
        layout.addWidget(self.clear_ruler_btn)

        layout.addWidget(QLabel('<b>═══ Contrast & Focus ═══</b>'))
        self.contrast_group = QButtonGroup()
        self.normal_contrast_radio = QRadioButton('🔘 Normal')
        self.bone_contrast_radio = QRadioButton('🦴 Bone focus')
        self.organ_contrast_radio = QRadioButton('🫀 Organ focus')
        self.normal_contrast_radio.setChecked(True)
        for r in (self.normal_contrast_radio, self.bone_contrast_radio, self.organ_contrast_radio):
            r.setEnabled(False)
            layout.addWidget(r)
            self.contrast_group.addButton(r)
        self.normal_contrast_radio.toggled.connect(lambda: self.set_contrast_mode('normal'))
        self.bone_contrast_radio.toggled.connect(lambda: self.set_contrast_mode('bone'))
        self.organ_contrast_radio.toggled.connect(lambda: self.set_contrast_mode('organ'))

        layout.addWidget(QLabel('Brightness (level):'))
        self.brightness_slider = QSlider(Qt.Horizontal); self.brightness_slider.setRange(-100, 100); self.brightness_slider.setEnabled(False)
        self.brightness_slider.valueChanged.connect(self.update_brightness_contrast)
        layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QLabel('0'); layout.addWidget(self.brightness_value_label)

        layout.addWidget(QLabel('Contrast (window):'))
        self.contrast_slider = QSlider(Qt.Horizontal); self.contrast_slider.setRange(-100, 100); self.contrast_slider.setEnabled(False)
        self.contrast_slider.valueChanged.connect(self.update_brightness_contrast)
        layout.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel('0'); layout.addWidget(self.contrast_value_label)

        reset_bc_btn = QPushButton('↺ Reset Brightness/Contrast')
        reset_bc_btn.clicked.connect(self.reset_brightness_contrast)
        layout.addWidget(reset_bc_btn)

        layout.addWidget(QLabel('<b>═══ Oblique Tilt (independent) ═══</b>'))

        # Sliders for oblique tilts (−180°..+180°)
        def make_angle_row(title):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            s = QSlider(Qt.Horizontal); s.setRange(-180, 180); s.setSingleStep(1); s.setEnabled(False)
            val = QLabel('0°'); val.setMinimumWidth(40)
            row.addWidget(s); row.addWidget(val)
            return row, s, val

        row, self.tilt_axial_slider, self.tilt_axial_label = make_angle_row('Axial tilt (Z):')
        layout.addLayout(row)
        row, self.tilt_sagittal_slider, self.tilt_sagittal_label = make_angle_row('Sagittal tilt (X):')
        layout.addLayout(row)
        row, self.tilt_coronal_slider, self.tilt_coronal_label = make_angle_row('Coronal tilt (Y):')
        layout.addLayout(row)

        # Connect sliders
        self.tilt_axial_slider.valueChanged.connect(self.update_oblique_tilts_from_sliders)
        self.tilt_sagittal_slider.valueChanged.connect(self.update_oblique_tilts_from_sliders)
        self.tilt_coronal_slider.valueChanged.connect(self.update_oblique_tilts_from_sliders)

        reset_oblique_btn = QPushButton('↺ Reset Tilts')
        reset_oblique_btn.clicked.connect(self.reset_oblique_tilts)
        layout.addWidget(reset_oblique_btn)

        layout.addStretch()
        return tab

    def create_3d_sectioning_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel('<b>═══ 4th Window Mode ═══</b>'))
        self.window_mode_group = QButtonGroup()
        self.oblique_mode_radio = QRadioButton('🔀 Oblique View'); self.oblique_mode_radio.setChecked(True)
        self.render3d_mode_radio = QRadioButton('🧊 3D Outline/Sectioning')
        for r in (self.oblique_mode_radio, self.render3d_mode_radio):
            layout.addWidget(r); self.window_mode_group.addButton(r)

        self.oblique_mode_radio.toggled.connect(
            lambda checked: self.set_fourth_window_mode('oblique') if checked else None
        )
        self.render3d_mode_radio.toggled.connect(
            lambda checked: self.set_fourth_window_mode('3d') if checked else None
        )

        layout.addWidget(QLabel('<b>═══ 3D Full Mesh Viewer ═══</b>'))
        self.btn_render_3d = QPushButton("🎨 Show Full 3D Mesh"); self.btn_render_3d.setEnabled(False); self.btn_render_3d.clicked.connect(self.render_3d_outline)
        layout.addWidget(self.btn_render_3d)

        layout.addWidget(QLabel('Camera Presets:'))
        row = QHBoxLayout()
        self.btn_view_ax = QPushButton("Axial")
        self.btn_view_sag = QPushButton("Sagittal")
        self.btn_view_cor = QPushButton("Coronal")
        for b in (self.btn_view_ax, self.btn_view_sag, self.btn_view_cor):
            b.setEnabled(False); row.addWidget(b)
        self.btn_view_ax.clicked.connect(lambda: self.set_3d_camera('axial'))
        self.btn_view_sag.clicked.connect(lambda: self.set_3d_camera('sagittal'))
        self.btn_view_cor.clicked.connect(lambda: self.set_3d_camera('coronal'))
        layout.addLayout(row)

        layout.addWidget(QLabel('<b>═══ 2D Outline (section) ═══</b>'))
        self.section_group = QButtonGroup()
        self.section_axial_radio = QRadioButton('👁 Axial Outline'); self.section_axial_radio.setChecked(True)
        self.section_sagittal_radio = QRadioButton('👁 Sagittal Outline')
        self.section_coronal_radio = QRadioButton('👁 Coronal Outline')
        for r in (self.section_axial_radio, self.section_sagittal_radio, self.section_coronal_radio):
            r.setEnabled(False); layout.addWidget(r); self.section_group.addButton(r)
            r.toggled.connect(self.auto_show_2d_outline)

        layout.addWidget(QLabel('Section Position (0..1):'))
        self.section_position_slider = QSlider(Qt.Horizontal); self.section_position_slider.setRange(0, 100); self.section_position_slider.setValue(50); self.section_position_slider.setEnabled(False)
        self.section_position_slider.valueChanged.connect(self.update_section_position)
        layout.addWidget(self.section_position_slider)
        self.section_pos_label = QLabel('Position: 0.50'); self.section_pos_label.setStyleSheet("color:#d32f2f;font-weight:bold;")
        layout.addWidget(self.section_pos_label)

        self.apply_section_btn = QPushButton('✂ SHOW 2D OUTLINE (Auto)')
        self.apply_section_btn.setEnabled(False)
        self.apply_section_btn.clicked.connect(self.perform_sectioning)
        layout.addWidget(self.apply_section_btn)

        self.section_info_label = QLabel('ℹ Selecting a plane shows the 2D outline of the mesh.')
        self.section_info_label.setWordWrap(True)
        self.section_info_label.setStyleSheet("background:#e3f2fd;padding:8px;border-radius:4px;")
        layout.addWidget(self.section_info_label)

        layout.addStretch()
        return tab

    def create_viewport_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0,0,0,0); layout.setSpacing(6)

        self.axial_fig, self.axial_ax = plt.subplots(figsize=(6,6))
        self.sagittal_fig, self.sagittal_ax = plt.subplots(figsize=(6,6))
        self.coronal_fig, self.coronal_ax = plt.subplots(figsize=(6,6))

        self.axial_canvas = FigureCanvas(self.axial_fig)
        self.sagittal_canvas = FigureCanvas(self.sagittal_fig)
        self.coronal_canvas = FigureCanvas(self.coronal_fig)

        self.fourth_fig = plt.figure(figsize=(6,6))
        self.fourth_ax = self.fourth_fig.add_subplot(111)
        self.fourth_canvas = FigureCanvas(self.fourth_fig)

        for c in (self.axial_canvas, self.sagittal_canvas, self.coronal_canvas, self.fourth_canvas):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            c.setMinimumSize(450, 400)

        # Mouse events on main views
        for canvas, view_name in [(self.axial_canvas, 'axial'),
                                  (self.sagittal_canvas, 'sagittal'),
                                  (self.coronal_canvas, 'coronal')]:
            canvas.mpl_connect('button_press_event', lambda e, v=view_name: self.on_mouse_press(e, v))
            canvas.mpl_connect('button_release_event', lambda e, v=view_name: self.on_mouse_release(e, v))
            canvas.mpl_connect('motion_notify_event', lambda e, v=view_name: self.on_mouse_move(e, v))
            canvas.mpl_connect('button_press_event', lambda e, v=view_name: self.on_view_click(e, v))

        # Grid 2x2
        grid = QGridLayout()
        grid.setContentsMargins(0,0,0,0)
        grid.setHorizontalSpacing(6); grid.setVerticalSpacing(6)
        grid.addWidget(self.create_view_group("Axial View", self.axial_canvas, 0), 0, 0)
        grid.addWidget(self.create_view_group("Sagittal View", self.sagittal_canvas, 1), 0, 1)
        grid.addWidget(self.create_view_group("Coronal View", self.coronal_canvas, 2), 1, 0)

        # 4th Window + Oblique slider
        self.fourth_group = QGroupBox("Oblique / 3D View")
        self.fourth_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fourth_layout = QVBoxLayout(); fourth_layout.setContentsMargins(6,6,6,6)
        fourth_layout.addWidget(self.fourth_canvas)

        # Oblique slice offset slider (visible/enabled only in Oblique mode)
        ob_row = QHBoxLayout()
        self.oblique_offset_label = QLabel("Oblique offset: 0.0 vox")
        self.oblique_offset_slider = QSlider(Qt.Horizontal)
        self.oblique_offset_slider.setRange(-300, 300)   # +/- 300 voxels along normal
        self.oblique_offset_slider.setValue(0)
        self.oblique_offset_slider.setEnabled(False)
        self.oblique_offset_slider.valueChanged.connect(self._on_oblique_offset_changed)
        ob_row.addWidget(self.oblique_offset_label); ob_row.addWidget(self.oblique_offset_slider)
        fourth_layout.addLayout(ob_row)

        self.fourth_group.setLayout(fourth_layout)
        grid.addWidget(self.fourth_group, 1, 1)

        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)

        layout.addLayout(grid)
        return panel

    def create_view_group(self, title, canvas, slider_idx):
        group = QGroupBox(title)
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(group); lay.setContentsMargins(6,6,6,6); lay.setSpacing(6)
        lay.addWidget(canvas)
        slider = QSlider(Qt.Horizontal)
        slider.valueChanged.connect(lambda v: self.update_slice(slider_idx, v))
        lay.addWidget(slider)
        if slider_idx == 0: self.axial_slider = slider
        elif slider_idx == 1: self.sagittal_slider = slider
        else: self.coronal_slider = slider
        return group

    # ---------------------------
    # Load data
    # ---------------------------
    def load_dicom(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Select Data Type")
        msg.setText("What type of data do you want to load?")
        dicom_btn = msg.addButton("DICOM Folder", QMessageBox.ActionRole)
        nifti_btn = msg.addButton("NIfTI File", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec()
        if msg.clickedButton() == dicom_btn:
            self.load_dicom_folder()
        elif msg.clickedButton() == nifti_btn:
            self.load_nifti_file()

    def load_nifti_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "",
                                                   "NIfTI files (*.nii *.nii.gz);;All files (*)")
        if not file_path: return
        try:
            import SimpleITK as sitk
            image = sitk.ReadImage(file_path)
            arr = sitk.GetArrayFromImage(image)  # (z,y,x)
            self.dicom_data = arr.astype(np.float32)
            sx, sy, sz = image.GetSpacing()  # (x,y,z)
            self._voxel_sizes = (float(sz), float(sy), float(sx))
            self.initialize_after_load()
        except Exception as e:
            self.status_bar.showMessage(f"Error loading NIfTI: {str(e)}")

    def load_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder: return
        try:
            files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
            if not files:
                self.status_bar.showMessage("No DICOM files found"); return
            ds_list = []
            for f in files:
                try:
                    ds = pydicom.dcmread(f)
                    ds_list.append(ds)
                except Exception:
                    pass
            if not ds_list:
                self.status_bar.showMessage("No valid DICOM slices")
                return
            try:
                ds_list.sort(key=lambda d: int(getattr(d, 'InstanceNumber', 0)))
            except Exception:
                pass
            slices = []
            for ds in ds_list:
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds.pixel_array.astype(np.float32))
            self.dicom_data = np.stack(slices, axis=0)  # (z,y,x)
            try:
                py, px = [float(v) for v in ds_list[0].PixelSpacing]
                pz = float(getattr(ds_list[0], 'SliceThickness', 1.0))
                self._voxel_sizes = (pz, py, px)
            except Exception:
                self._voxel_sizes = (1.0, 1.0, 1.0)
            self.initialize_after_load()
        except Exception as e:
            self.status_bar.showMessage(f"Error loading DICOM: {str(e)}")

    def initialize_after_load(self):
        Z, Y, X = self.dicom_data.shape
        self.axial_slider.setMaximum(Z-1)
        self.sagittal_slider.setMaximum(X-1)
        self.coronal_slider.setMaximum(Y-1)

        for spin, mx in [(self.axial_start_spin, Z-1), (self.axial_end_spin, Z-1),
                         (self.sagittal_start_spin, X-1), (self.sagittal_end_spin, X-1),
                         (self.coronal_start_spin, Y-1), (self.coronal_end_spin, Y-1)]:
            spin.setMaximum(mx)

        self.axial_start_spin.setValue(0); self.axial_end_spin.setValue(Z-1)
        self.sagittal_start_spin.setValue(0); self.sagittal_end_spin.setValue(X-1)
        self.coronal_start_spin.setValue(0); self.coronal_end_spin.setValue(Y-1)

        self.current_slices = [Z//2, X//2, Y//2]
        self.axial_slider.setValue(self.current_slices[0])
        self.sagittal_slider.setValue(self.current_slices[1])
        self.coronal_slider.setValue(self.current_slices[2])

        # baseline contrast stats
        self._base_lo, self._base_hi = safe_percentiles(self.dicom_data, 1, 99)

        self.update_all_views()

        # enable controls
        self.load_mask_btn.setEnabled(True)
        self.roi_btn.setEnabled(True)
        for r in (self.roi_axial_radio, self.roi_sagittal_radio, self.roi_coronal_radio):
            r.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.ruler_btn.setEnabled(True)
        self.clear_ruler_btn.setEnabled(True)
        for r in (self.normal_contrast_radio, self.bone_contrast_radio, self.organ_contrast_radio):
            r.setEnabled(True)
        self.brightness_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        for s in (self.tilt_axial_slider, self.tilt_sagittal_slider, self.tilt_coronal_slider):
            s.setEnabled(True)

        # Oblique offset slider only enabled in Oblique mode
        self._update_oblique_offset_enabled()

        self.status_bar.showMessage(
            f"Loaded: {Z} slices, {Y}x{X}px | spacing (z,y,x)={self._voxel_sizes} mm")

    # ---------------------------
    # Mask
    # ---------------------------
    def load_mask(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Select Mask Type")
        msg.setText("What type of mask data?")
        dicom_btn = msg.addButton("DICOM Folder", QMessageBox.ActionRole)
        nifti_btn = msg.addButton("NIfTI File", QMessageBox.ActionRole)
        numpy_btn = msg.addButton("NumPy File", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec()
        if msg.clickedButton() == dicom_btn:
            self.load_mask_dicom_folder()
        elif msg.clickedButton() == nifti_btn:
            self.load_mask_nifti()
        elif msg.clickedButton() == numpy_btn:
            self.load_mask_numpy()

    def load_mask_numpy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open NumPy Mask", "", "NumPy files (*.npy)")
        if not path: return
        try:
            m = np.load(path)
            if self.dicom_data is None or m.shape != self.dicom_data.shape:
                self.status_bar.showMessage("Mask shape mismatch!")
                return
            self.mask_data = m
            self.mask_checkbox.setEnabled(True)
            self._enable_auto_roi_controls()
            self.update_all_views()
            self.status_bar.showMessage("Loaded NumPy mask")
        except Exception as e:
            self.status_bar.showMessage(f"Error: {e}")

    def load_mask_nifti(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI Mask", "", "NIfTI files (*.nii *.nii.gz)")
        if not path: return
        try:
            import SimpleITK as sitk
            m = sitk.GetArrayFromImage(sitk.ReadImage(path))
            if self.dicom_data is None or m.shape != self.dicom_data.shape:
                self.status_bar.showMessage("Mask shape mismatch!")
                return
            self.mask_data = m
            self.mask_checkbox.setEnabled(True)
            self._enable_auto_roi_controls()
            self.update_all_views()
            self.status_bar.showMessage("Loaded NIfTI mask")
        except Exception as e:
            self.status_bar.showMessage(f"Error: {e}")

    def load_mask_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Mask Folder")
        if not folder: return
        try:
            files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
            files.sort()
            slices = []
            for f in files:
                try:
                    ds = pydicom.dcmread(f)
                    if hasattr(ds, 'pixel_array'):
                        slices.append(ds.pixel_array)
                except Exception:
                    pass
            if not slices:
                self.status_bar.showMessage("No valid DICOM mask data"); return
            m = np.stack(slices, axis=0)
            if self.dicom_data is None or m.shape != self.dicom_data.shape:
                self.status_bar.showMessage("Mask shape mismatch!"); return
            self.mask_data = m
            self.mask_checkbox.setEnabled(True)
            self._enable_auto_roi_controls()
            self.update_all_views()
            self.status_bar.showMessage("Loaded DICOM mask")
        except Exception as e:
            self.status_bar.showMessage(f"Error: {e}")

    def _enable_auto_roi_controls(self):
        if self.mask_data is None: return
        labels = np.unique(self.mask_data); labels = labels[labels!=0]
        self.label_combo.clear()
        for v in labels: self.label_combo.addItem(str(int(v)))
        have = len(labels)>0
        self.label_combo.setEnabled(have); self.auto_roi_btn.setEnabled(have)
        self.zoom_in_btn.setEnabled(have)
        for b in (self.btn_render_3d, self.btn_view_ax, self.btn_view_sag, self.btn_view_cor):
            b.setEnabled(True)
        for r in (self.section_axial_radio, self.section_sagittal_radio, self.section_coronal_radio):
            r.setEnabled(True)
        self.section_position_slider.setEnabled(True)
        self.apply_section_btn.setEnabled(True)
        self._mesh_cache = None

    def toggle_mask(self):
        self.show_mask = self.mask_checkbox.isChecked()
        self.update_all_views()

    # ---------------------------
    # ROI + mouse on main views
    # ---------------------------
    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            if self.roi_axial_radio.isChecked(): self.roi_active_view = 'axial'
            elif self.roi_sagittal_radio.isChecked(): self.roi_active_view = 'sagittal'
            else: self.roi_active_view = 'coronal'
            self.roi_btn.setText(f'Drawing on {self.roi_active_view.capitalize()}...')
            self.status_bar.showMessage(f"Draw ROI on {self.roi_active_view.capitalize()}")
        else:
            self.roi_btn.setText('✏️ Draw ROI Rectangle')
            self.status_bar.showMessage("ROI mode off")

    def on_mouse_press(self, event, view_name):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # ======= NEW: Shift+Drag to set oblique line in any view =======
        if event.button == 1 and (event.key == 'shift' or (event.guiEvent is not None and event.guiEvent.modifiers() & Qt.ShiftModifier)):
            # start line
            self._oblique_lines[view_name] = [(event.xdata, event.ydata), (event.xdata, event.ydata)]
            self._last_oblique_source = view_name
            self.update_view(view_name)  # show live line
            return

        # Start ROI rectangle
        if self.roi_mode and view_name == self.roi_active_view:
            self.roi_start = (event.xdata, event.ydata)
            return

        # Ruler points
        if self.ruler_mode:
            if view_name not in self.ruler_points: self.ruler_points[view_name] = []
            self.ruler_points[view_name].append((event.xdata, event.ydata))
            if len(self.ruler_points[view_name]) == 2:
                self.calculate_distance(view_name)
                self.ruler_mode = False
                self.ruler_btn.setChecked(False)
            self.update_view(view_name)
            return

        # begin crosshair drag (plain left button)
        if event.button == 1:
            self._dragging_crosshair_view = view_name
            self._update_crosshair_from_event(view_name, event.xdata, event.ydata)

    def on_mouse_release(self, event, view_name):
        # finish oblique line?
        if self._oblique_lines.get(view_name) is not None and isinstance(self._oblique_lines[view_name], list):
            # finalize to tuple
            p0, p1 = self._oblique_lines[view_name]
            self._oblique_lines[view_name] = (p0, p1)
            self._last_oblique_source = view_name
            # re-render oblique if in oblique mode
            if self.fourth_window_mode == 'oblique':
                self.render_oblique_view()
            self.update_view(view_name)
            self._dragging_crosshair_view = None
            return

        # Finish ROI rectangle
        if self.roi_mode and view_name == self.roi_active_view and self.roi_start is not None:
            if event.inaxes is None:
                return
            x1, y1 = self.roi_start
            x2, y2 = event.xdata, event.ydata
            x, y = min(x1,x2), min(y1,y2)
            w, h = abs(x2-x1), abs(y2-y1)
            if self.roi_coords is None: self.roi_coords = {}
            self.roi_coords[view_name] = (int(x), int(y), int(w), int(h))
            self.detect_roi_slices_and_sync(view_name)
            self.roi_mode = False
            self.roi_btn.setText('✏️ Draw ROI Rectangle')
            for b in (self.export_all_btn, self.export_axial_btn, self.export_sagittal_btn, self.export_coronal_btn, self.zoom_in_btn):
                b.setEnabled(True)
            self.update_all_views(); self.update_roi_info()
            return

        # end crosshair dragging
        self._dragging_crosshair_view = None

    def on_mouse_move(self, event, view_name):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # live oblique line while dragging with Shift
        if self._oblique_lines.get(view_name) is not None and isinstance(self._oblique_lines[view_name], list):
            self._oblique_lines[view_name][1] = (event.xdata, event.ydata)
            self.update_view(view_name)
            if self.fourth_window_mode == 'oblique':
                self.render_oblique_view()
            return

        # ROI preview
        if self.roi_mode and view_name == self.roi_active_view and self.roi_start is not None:
            self.update_view(view_name, preview_roi=(self.roi_start, (event.xdata, event.ydata)))
            return

        # live crosshair dragging
        if self._dragging_crosshair_view == view_name:
            self._update_crosshair_from_event(view_name, event.xdata, event.ydata)

    def _update_crosshair_from_event(self, view_name, x, y):
        if self.dicom_data is None:
            return
        Z, Y, X = self.dicom_data.shape
        if view_name == 'axial':
            self.sagittal_slider.setValue(int(np.clip(x, 0, X-1)))
            self.coronal_slider.setValue(int(np.clip(y, 0, Y-1)))
        elif view_name == 'sagittal':
            self.coronal_slider.setValue(int(np.clip(x, 0, Y-1)))
            self.axial_slider.setValue(int(np.clip(Z-1-y, 0, Z-1)))
        elif view_name == 'coronal':
            self.sagittal_slider.setValue(int(np.clip(x, 0, X-1)))
            self.axial_slider.setValue(int(np.clip(Z-1-y, 0, Z-1)))

    def on_view_click(self, event, view_name):
        # simple click jumps crosshair too
        if event.inaxes is None or event.button != 1 or self.roi_mode or self.ruler_mode or self.dicom_data is None:
            return
        # ignore if shift (handled as line)
        if (event.key == 'shift') or (event.guiEvent is not None and event.guiEvent.modifiers() & Qt.ShiftModifier):
            return
        self._update_crosshair_from_event(view_name, event.xdata, event.ydata)

    def detect_roi_slices_and_sync(self, view_name):
        if self.roi_coords is None or view_name not in self.roi_coords or self.mask_data is None:
            return
        x, y, w, h = self.roi_coords[view_name]
        yy = slice(y, min(y+h, self.dicom_data.shape[1]))
        xx = slice(x, min(x+w, self.dicom_data.shape[2]))
        if view_name == 'axial':
            idxs = [i for i in range(self.dicom_data.shape[0]) if np.any(self.mask_data[i, yy, xx] > 0)]
            if idxs:
                z0, z1 = min(idxs), max(idxs)
                if self.roi_slices is None: self.roi_slices = {}
                self.roi_slices['axial'] = (z0, z1)
                self.axial_start_spin.setValue(z0); self.axial_end_spin.setValue(z1)

    def auto_detect_other_view_ranges(self, *args, **kwargs):
        pass

    def update_manual_roi(self):
        if self.roi_slices is None: self.roi_slices = {}
        self.roi_slices['axial'] = (self.axial_start_spin.value(), self.axial_end_spin.value())
        self.roi_slices['sagittal'] = (self.sagittal_start_spin.value(), self.sagittal_end_spin.value())
        self.roi_slices['coronal'] = (self.coronal_start_spin.value(), self.coronal_end_spin.value())
        self.update_roi_info()

    def update_roi_info(self):
        if self.roi_coords is None and self.roi_slices is None:
            self.roi_info_label.setText('ROI: Not selected'); return
        txt = "ROI Info:\n"
        if self.roi_coords:
            for v, (x,y,w,h) in self.roi_coords.items():
                txt += f"{v.capitalize()}: ({x},{y}) {w}x{h}\n"
        if self.roi_slices:
            txt += "\nRanges:\n"
            for v,(s,e) in self.roi_slices.items():
                txt += f"{v.capitalize()}: {s}-{e}\n"
        self.roi_info_label.setText(txt)

    # ---------------------------
    # Slices & views
    # ---------------------------
    def update_slice(self, idx, value):
        if self.dicom_data is None: return
        self.current_slices[idx] = value
        self.update_all_views()

    def update_all_views(self):
        if self.dicom_data is None: return
        self.update_view('axial')
        self.update_view('sagittal')
        self.update_view('coronal')
        self.update_fourth_window()

    def apply_display_window(self, img):
        lo, hi = self._base_lo, self._base_hi
        imgf = np.asarray(img, dtype=np.float32)
        lvl_shift = (self.brightness_slider.value())/100.0
        wl = (lo+hi)/2.0 + (hi-lo)*0.5*lvl_shift
        c = self.contrast_slider.value()
        width_scale = 2.0 ** (-c/50.0)
        ww = (hi-lo) * width_scale
        low = wl - ww/2.0
        high = wl + ww/2.0
        if self.contrast_mode == 'bone':
            low = wl + 0.1*ww; high = wl + 0.5*ww
        elif self.contrast_mode == 'organ':
            low = wl - 0.4*ww; high = wl + 0.2*ww
        imgf = np.clip((imgf - low) / max(high-low, 1e-6), 0.0, 1.0)
        return imgf

    def update_view(self, view_name, preview_roi=None):
        if self.dicom_data is None: return

        if view_name == 'axial':
            ax, canvas = self.axial_ax, self.axial_canvas
            slice_data = self.dicom_data[self.current_slices[0], :, :]
            mask_slice = self.mask_data[self.current_slices[0], :, :] if self.mask_data is not None else None
            cross_x, cross_y = self.current_slices[1], self.current_slices[2]
        elif view_name == 'sagittal':
            ax, canvas = self.sagittal_ax, self.sagittal_canvas
            slice_data = np.flipud(self.dicom_data[:, :, self.current_slices[1]])
            mask_slice = np.flipud(self.mask_data[:, :, self.current_slices[1]]) if self.mask_data is not None else None
            cross_x = self.current_slices[2]
            cross_y = self.dicom_data.shape[0]-1-self.current_slices[0]
        else:  # coronal
            ax, canvas = self.coronal_ax, self.coronal_canvas
            slice_data = np.flipud(self.dicom_data[:, self.current_slices[2], :])
            mask_slice = np.flipud(self.mask_data[:, self.current_slices[2], :]) if self.mask_data is not None else None
            cross_x = self.current_slices[1]
            cross_y = self.dicom_data.shape[0]-1-self.current_slices[0]

        img_disp = self.apply_display_window(slice_data)

        ax.clear()
        ax.imshow(img_disp, cmap='gray', aspect='auto', vmin=0.0, vmax=1.0)

        if mask_slice is not None and self.show_mask:
            masked = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(masked, cmap='jet', alpha=0.45, aspect='auto')

        if self.roi_coords is not None and view_name in self.roi_coords:
            x, y, w, h = self.roi_coords[view_name]
            rect = patches.Rectangle((x,y), w,h, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)

        if preview_roi is not None:
            (x1,y1),(x2,y2) = preview_roi
            x,y = min(x1,x2), min(y1,y2)
            w,h = abs(x2-x1), abs(y2-y1)
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--')
            ax.add_patch(rect)

        # ruler points
        if view_name in self.ruler_points and len(self.ruler_points[view_name])>0:
            pts = self.ruler_points[view_name]
            for p in pts:
                ax.plot(p[0], p[1], 'ro', markersize=6)
            if len(pts)==2:
                p1,p2 = pts
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]], 'r-', linewidth=2)

        # crosshair (thin)
        ax.axvline(cross_x, color='red', linestyle='--', linewidth=1)
        ax.axhline(cross_y, color='red', linestyle='--', linewidth=1)
        ax.plot(cross_x, cross_y, 'ro', markersize=4)

        # ======= NEW: draw oblique line if present for this view =======
        line = self._oblique_lines.get(view_name)
        if line:
            (x0,y0),(x1,y1) = line
            ax.plot([x0,x1],[y0,y1], color='cyan', linewidth=2)

        ax.set_title(f"{view_name.capitalize()} - Slice {self.current_slices[['axial','sagittal','coronal'].index(view_name)]}")
        ax.axis('on'); canvas.draw()

    # ---------------------------
    # Export / Reset
    # ---------------------------
    def export_roi(self, export_type):
        if self.dicom_data is None:
            self.status_bar.showMessage("No data loaded"); return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save ROI Data", "", "NIfTI files (*.nii.gz *.nii)")
        if not save_path: return
        try:
            import SimpleITK as sitk
            if export_type == 'all':
                if self.roi_slices:
                    roi = self.dicom_data[
                        self.axial_start_spin.value():self.axial_end_spin.value()+1,
                        self.coronal_start_spin.value():self.coronal_end_spin.value()+1,
                        self.sagittal_start_spin.value():self.sagittal_end_spin.value()+1
                    ]
                else:
                    roi = self.dicom_data.copy()
            elif export_type == 'axial':
                s,e = self.axial_start_spin.value(), self.axial_end_spin.value()
                if self.roi_coords and 'axial' in self.roi_coords:
                    x,y,w,h = self.roi_coords['axial']
                    roi = self.dicom_data[s:e+1, y:y+h, x:x+w]
                else:
                    roi = self.dicom_data[s:e+1, :, :]
            elif export_type == 'sagittal':
                s,e = self.sagittal_start_spin.value(), self.sagittal_end_spin.value()
                roi = self.dicom_data[:, :, s:e+1]
            else:  # coronal
                s,e = self.coronal_start_spin.value(), self.coronal_end_spin.value()
                roi = self.dicom_data[:, s:e+1, :]
            img = sitk.GetImageFromArray(roi)
            sz, sy, sx = self._voxel_sizes
            img.SetSpacing((sx, sy, sz))
            sitk.WriteImage(img, save_path)
            self.status_bar.showMessage(f"Exported {export_type}: {roi.shape} → {save_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error: {e}")

    def reset_view(self):
        if self.dicom_data is None: return
        Z,Y,X = self.dicom_data.shape
        self.current_slices = [Z//2, X//2, Y//2]
        self.axial_slider.setValue(self.current_slices[0])
        self.sagittal_slider.setValue(self.current_slices[1])
        self.coronal_slider.setValue(self.current_slices[2])
        self.roi_coords = None; self.roi_slices=None
        self._oblique_lines = {'axial': None, 'sagittal': None, 'coronal': None}
        self._last_oblique_source = None
        self.roi_info_label.setText('ROI: Not selected')
        for b in (self.export_all_btn, self.export_axial_btn, self.export_sagittal_btn, self.export_coronal_btn, self.zoom_in_btn):
            b.setEnabled(False)
        self.brightness_slider.setValue(0); self.contrast_slider.setValue(0)
        self.reset_oblique_tilts()
        self.oblique_offset_slider.setValue(0)
        self.update_all_views()
        self.status_bar.showMessage("View reset")

    def auto_roi_from_mask(self):
        if self.mask_data is None or self.dicom_data is None:
            self.status_bar.showMessage("Load data and mask first"); return
        if self.label_combo.currentText().strip()=="":
            self.status_bar.showMessage("No label selected"); return
        label = int(self.label_combo.currentText())
        zz, yy, xx = np.where(self.mask_data == label)
        if len(zz)==0:
            self.status_bar.showMessage(f"Label {label} not found"); return
        z0,z1 = int(zz.min()), int(zz.max())
        y0,y1 = int(yy.min()), int(yy.max())
        x0,x1 = int(xx.min()), int(xx.max())
        w,h = x1-x0+1, y1-y0+1
        if self.roi_coords is None: self.roi_coords={}
        self.roi_coords['axial'] = (x0, y0, w, h)
        self.sagittal_slider.setValue((x0+x1)//2)
        self.coronal_slider.setValue((y0+y1)//2)
        self.axial_slider.setValue((z0+z1)//2)
        if self.roi_slices is None: self.roi_slices={}
        self.roi_slices['axial']=(z0,z1); self.roi_slices['sagittal']=(x0,x1); self.roi_slices['coronal']=(y0,y1)
        self.axial_start_spin.setValue(z0); self.axial_end_spin.setValue(z1)
        self.sagittal_start_spin.setValue(x0); self.sagittal_end_spin.setValue(x1)
        self.coronal_start_spin.setValue(y0); self.coronal_end_spin.setValue(y1)
        for b in (self.export_all_btn, self.export_axial_btn, self.export_sagittal_btn, self.export_coronal_btn, self.zoom_in_btn):
            b.setEnabled(True)
        self.update_all_views(); self.update_roi_info()
        self.status_bar.showMessage(f"Auto ROI from label {label}")

    def roi_zoom_in(self):
        if self.roi_slices is None and (self.roi_coords is None or 'axial' not in self.roi_coords):
            self.status_bar.showMessage("Draw/auto ROI first"); return
        if self.roi_slices:
            a = self.roi_slices.get('axial', (self.axial_start_spin.value(), self.axial_end_spin.value()))
            s = self.roi_slices.get('sagittal', (self.sagittal_start_spin.value(), self.sagittal_end_spin.value()))
            c = self.roi_slices.get('coronal', (self.coronal_start_spin.value(), self.coronal_end_spin.value()))
            self.axial_start_spin.setValue(a[0]); self.axial_end_spin.setValue(a[1])
            self.sagittal_start_spin.setValue(s[0]); self.sagittal_end_spin.setValue(s[1])
            self.coronal_start_spin.setValue(c[0]); self.coronal_end_spin.setValue(c[1])
        self.update_roi_info()
        self.status_bar.showMessage("Zoomed to ROI")

    def roi_zoom_out(self):
        if self.dicom_data is None: return
        Z,Y,X = self.dicom_data.shape
        self.axial_start_spin.setValue(0); self.axial_end_spin.setValue(Z-1)
        self.sagittal_start_spin.setValue(0); self.sagittal_end_spin.setValue(X-1)
        self.coronal_start_spin.setValue(0); self.coronal_end_spin.setValue(Y-1)
        self.update_roi_info()
        self.status_bar.showMessage("Zoomed to full volume")

    # ---------------------------
    # Ruler / contrast controls
    # ---------------------------
    def toggle_ruler_mode(self):
        self.ruler_mode = self.ruler_btn.isChecked()
        self.status_bar.showMessage("Click 2 points to measure" if self.ruler_mode else "Ruler off")

    def calculate_distance(self, view_name):
        if view_name not in self.ruler_points or len(self.ruler_points[view_name]) != 2:
            return
        p1, p2 = self.ruler_points[view_name]
        dx = float(p2[0]-p1[0]); dy = float(p2[1]-p1[1])
        pix_dist = np.hypot(dx, dy)

        vz, vy, vx = self._voxel_sizes
        if view_name == 'axial':
            sx, sy = vx, vy
        elif view_name == 'sagittal':
            sx, sy = vy, vz
        else:  # coronal
            sx, sy = vx, vz
        mm_dist = np.hypot(dx*sx, dy*sy)

        self.ruler_info_label.setText(f"Distance ({view_name}): {pix_dist:.2f} px  |  {mm_dist:.2f} mm")
        self.status_bar.showMessage(f"{pix_dist:.2f} px, {mm_dist:.2f} mm")

    def clear_ruler(self):
        self.ruler_points = {}
        self.ruler_info_label.setText('Distance: N/A')
        self.update_all_views()
        self.status_bar.showMessage("Ruler cleared")

    def set_contrast_mode(self, mode):
        self.contrast_mode = mode
        self.update_all_views()
        self.status_bar.showMessage(f"Contrast preset: {mode}")

    def update_brightness_contrast(self):
        self.brightness_value_label.setText(str(self.brightness_slider.value()))
        self.contrast_value_label.setText(str(self.contrast_slider.value()))
        self.update_all_views()

    def reset_brightness_contrast(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.update_all_views()

    # ---------------------------
    # Oblique (true cut)
    # ---------------------------
    def set_fourth_window_mode(self, mode):
        self.fourth_window_mode = mode
        self.fourth_group.setTitle("Oblique View" if mode == 'oblique' else "3D Outline / 2D Section")

        self.fourth_fig.clear()
        self.fourth_ax = self.fourth_fig.add_subplot(111)
        self.fourth_canvas.draw()

        self._update_oblique_offset_enabled()
        self.update_fourth_window()
        self.status_bar.showMessage(f"4th window mode: {mode}")

    def _update_oblique_offset_enabled(self):
        enabled = (self.fourth_window_mode == 'oblique') and (self.dicom_data is not None)
        self.oblique_offset_slider.setEnabled(enabled)

    def update_fourth_window(self):
        if self.fourth_window_mode == 'oblique':
            self.render_oblique_view()

    def update_oblique_tilts_from_sliders(self):
        self.tilt_axial = float(self.tilt_axial_slider.value())
        self.tilt_sagittal = float(self.tilt_sagittal_slider.value())
        self.tilt_coronal = float(self.tilt_coronal_slider.value())
        self.tilt_axial_label.setText(f"{int(self.tilt_axial)}°")
        self.tilt_sagittal_label.setText(f"{int(self.tilt_sagittal)}°")
        self.tilt_coronal_label.setText(f"{int(self.tilt_coronal)}°")
        if self.fourth_window_mode == 'oblique':
            self.render_oblique_view()

    def _on_oblique_offset_changed(self, value):
        self.oblique_offset = float(value)
        self.oblique_offset_label.setText(f"Oblique offset: {self.oblique_offset:.1f} vox")
        if self.fourth_window_mode == 'oblique':
            self.render_oblique_view()

    def reset_oblique_tilts(self):
        self.tilt_axial_slider.blockSignals(True); self.tilt_axial_slider.setValue(0); self.tilt_axial_slider.blockSignals(False)
        self.tilt_sagittal_slider.blockSignals(True); self.tilt_sagittal_slider.setValue(0); self.tilt_sagittal_slider.blockSignals(False)
        self.tilt_coronal_slider.blockSignals(True); self.tilt_coronal_slider.setValue(0); self.tilt_coronal_slider.blockSignals(False)
        self.update_oblique_tilts_from_sliders()

    def _oblique_from_line(self):
        """
        If a Shift-drawn line exists, return (center[z,y,x], u(3,), o(3,))
        u and o are orthonormal basis vectors for the cutting plane.
        """
        if self._last_oblique_source is None:
            return None

        src = self._last_oblique_source
        line = self._oblique_lines.get(src)
        if not line:  # None or invalid
            return None

        (x0,y0),(x1,y1) = line
        # center in 3D voxel coords from view + current crosshair slice index
        if src == 'axial':
            # image axes: (x horizontal, y vertical); plane z=const
            z = float(self.current_slices[0])
            center = np.array([z, (y0+y1)/2.0, (x0+x1)/2.0], dtype=np.float32)
            # direction in-plane:
            dx = (x1 - x0); dy = (y1 - y0)
            u = _norm([0.0, dy, dx])         # (z,y,x)
            o = _norm([1.0, 0.0, 0.0])       # out-of-plane axis: +z

        elif src == 'sagittal':
            # sagittal view shows flipud(dicom[:, :, x]); axes: ximg->y, yimg->-z
            x = float(self.current_slices[1])
            center = np.array([(self.dicom_data.shape[0]-1 - (y0+y1)/2.0),  # z
                               (x0+x1)/2.0,                                 # y
                               x], dtype=np.float32)                        # x (const)
            dx_img = (x1 - x0)
            dy_img = (y1 - y0)
            u = _norm([-dy_img, dx_img, 0.0])   # (z,y,x) from (ximg->y, yimg->-z)
            o = _norm([0.0, 0.0, 1.0])          # out-of-plane axis: +x

        else:  # 'coronal'
            # coronal view shows flipud(dicom[:, y, :]); axes: ximg->x, yimg->-z
            y = float(self.current_slices[2])
            center = np.array([(self.dicom_data.shape[0]-1 - (y0+y1)/2.0),  # z
                               y,                                           # y (const)
                               (x0+x1)/2.0], dtype=np.float32)              # x
            dx_img = (x1 - x0)
            dy_img = (y1 - y0)
            u = _norm([-dy_img, 0.0, dx_img])   # (z,y,x)
            o = _norm([0.0, 1.0, 0.0])          # out-of-plane axis: +y

        # ensure orthonormal frame (re-orthogonalize o against u)
        o = _norm(o - np.dot(o, u)*u)
        return center, u, o

    def render_oblique_view(self):
        """True MPR oblique re-slice of the VOLUME (data only)."""
        ax = self.fourth_ax
        ax.clear()
        if self.dicom_data is None:
            self.fourth_canvas.draw(); return

        Z,Y,X = self.dicom_data.shape

        # Try line-defined plane first
        lo = self._oblique_from_line()
        if lo is not None:
            center, u, o = lo
            # normal for offset
            n = _norm(np.cross(u, o))
            center = center + n * self.oblique_offset
            title_suffix = " (line)"
        else:
            # fallback to tilt sliders: build plane basis from rotation matrix
            R = rotation_matrix_from_tilts(self.tilt_axial, self.tilt_sagittal, self.tilt_coronal)
            u = _norm(R[:,0]); o = _norm(R[:,1]); n = _norm(R[:,2])
            center = np.array([self.current_slices[0], self.current_slices[2], self.current_slices[1]], dtype=np.float32)
            center = center + n * self.oblique_offset
            title_suffix = " (tilts)"

        # output sampling size
        out_size = int(min(512, max(Z,Y,X)))
        coords = np.indices((out_size, out_size), dtype=np.float32)
        gy = coords[0] - (out_size/2.0)
        gx = coords[1] - (out_size/2.0)
        gx = gx.ravel(); gy = gy.ravel()

        # sample points in volume space
        sample_pts = (center[:,None] + u[:,None]*gx[None,:] + o[:,None]*gy[None,:])

        sampled = map_coordinates(self.dicom_data, sample_pts, order=1, mode='constant', cval=float(self.dicom_data.min()))
        oblique = sampled.reshape((out_size, out_size))

        img_disp = self.apply_display_window(oblique)
        ax.imshow(img_disp, cmap='gray', origin='lower', aspect='equal', vmin=0.0, vmax=1.0)

        if lo is not None:
            ax.set_title(f"Oblique{title_suffix} | Off:{self.oblique_offset:.1f}", fontsize=11)
        else:
            ax.set_title(f"Oblique{title_suffix} | Ax:{self.tilt_axial:.0f}° Sag:{self.tilt_sagittal:.0f}° Cor:{self.tilt_coronal:.0f}° | Off:{self.oblique_offset:.1f}", fontsize=11)
        ax.axis('off')
        self.fourth_canvas.draw()

    # ---------------------------
    # 3D mesh + 2D outlines  (UNMODIFIED LOGIC)
    # ---------------------------
    def _compute_3d_mesh_from_mask(self):
        if self.mask_data is None:
            raise RuntimeError("Load mask first")
        if self._mesh_cache is not None:
            return self._mesh_cache
        mask = (self.mask_data > 0).astype(np.uint8)
        mask = binary_closing(mask, iterations=1).astype(np.uint8)
        mask = binary_fill_holes(mask).astype(np.uint8)
        smooth = gaussian_filter(mask.astype(float), sigma=1.0)
        verts, faces, _, _ = measure.marching_cubes(smooth, level=0.5, step_size=1, spacing=self._voxel_sizes)
        self._mesh_cache = (verts, faces)
        return self._mesh_cache

    def render_3d_outline(self):
        try:
            verts, faces = self._compute_3d_mesh_from_mask()
        except Exception as e:
            self.status_bar.showMessage(f"3D render error: {e}")
            return

        self.fourth_fig.clear()
        self.fourth_ax = self.fourth_fig.add_subplot(111, projection='3d')
        ax = self.fourth_ax

        ax.set_facecolor('#1a1a1a')
        self.fourth_fig.patch.set_facecolor('#1a1a1a')

        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces, linewidth=0.1, antialiased=True, shade=True,
            alpha=0.9, color="#cfcfcf"
        )

        ax.set_title("3D Outline (Full Mesh)", color='white', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False); ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=20, azim=35)

        self.fourth_canvas.draw()
        self.status_bar.showMessage("3D outline rendered")

    def set_3d_camera(self, plane):
        if not hasattr(self.fourth_ax, 'view_init'):
            self.status_bar.showMessage("Render 3D first"); return
        if plane=='axial':   elev,az = 90,-90
        elif plane=='sagittal': elev,az = 0,0
        elif plane=='coronal':  elev,az = 0,90
        else: elev,az = 20,35
        self.fourth_ax.view_init(elev=elev, azim=az)
        self.fourth_canvas.draw()
        self.status_bar.showMessage(f"3D camera: {plane}")

    def auto_show_2d_outline(self):
        if self.mask_data is None or self.fourth_window_mode != '3d':
            return
        plane = 'axial' if self.section_axial_radio.isChecked() else 'sagittal' if self.section_sagittal_radio.isChecked() else 'coronal'
        try:
            self.generate_2d_outline(plane)
        except Exception as e:
            self.status_bar.showMessage(f"Error generating outline: {e}")

    def generate_2d_outline(self, plane):
        # UNCHANGED LOGIC
        try:
            verts, faces = self._compute_3d_mesh_from_mask()
        except Exception as e:
            self.status_bar.showMessage(f"Mesh error: {e}")
            return

        pos = self.sectioning_position
        segs = self.compute_section(verts, faces, plane, pos)
        if not segs:
            self.fourth_fig.clear()
            self.fourth_ax = self.fourth_fig.add_subplot(111)
            ax = self.fourth_ax
            ax.set_facecolor('black'); self.fourth_fig.patch.set_facecolor('black')
            ax.set_title(f"No intersection at {pos:.2f}", color='white')
            ax.axis('off')
            self.fourth_canvas.draw()
            self.status_bar.showMessage(f"No intersection at {pos:.2f}")
            return

        self.fourth_fig.clear()
        self.fourth_ax = self.fourth_fig.add_subplot(111)
        ax = self.fourth_ax

        ax.set_facecolor('black')
        self.fourth_fig.patch.set_facecolor('black')

        for seg in segs:
            seg = np.array(seg)
            if plane == 'axial':
                ax.plot(seg[:, 0], seg[:, 1], color='white', linewidth=2.2, alpha=1.0)
            elif plane == 'sagittal':
                ax.plot(seg[:, 1], seg[:, 2], color='white', linewidth=2.2, alpha=1.0)
            elif plane == 'coronal':
                ax.plot(seg[:, 0], seg[:, 2], color='white', linewidth=2.2, alpha=1.0)

        ax.set_title(f"2D Outline — {plane.capitalize()} @ {pos:.2f}", color='white', fontsize=12, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        self.fourth_canvas.draw()
        self.status_bar.showMessage(f"2D outline: {plane} | {len(segs)} segments | pos={pos:.2f}")

    def update_section_position(self, v):
        self.sectioning_position = v/100.0
        self.section_pos_label.setText(f"Position: {self.sectioning_position:.2f}")
        if self.fourth_window_mode=='3d' and self.mask_data is not None:
            self.auto_show_2d_outline()

    def perform_sectioning(self):
        if self.mask_data is None:
            self.status_bar.showMessage("Load mask first"); return
        if self.fourth_window_mode!='3d':
            self.status_bar.showMessage("Switch to 3D mode first"); return
        self.auto_show_2d_outline()

    def compute_section(self, verts, faces, plane='axial', position=0.5):
        if plane=='axial':
            n = np.array([0,0,1]); p0 = verts.mean(axis=0)
            zmin,zmax = verts[:,2].min(), verts[:,2].max(); p0[2] = zmin + (zmax-zmin)*position
        elif plane=='sagittal':
            n = np.array([1,0,0]); p0 = verts.mean(axis=0)
            xmin,xmax = verts[:,0].min(), verts[:,0].max(); p0[0] = xmin + (xmax-xmin)*position
        else: # coronal
            n = np.array([0,1,0]); p0 = verts.mean(axis=0)
            ymin,ymax = verts[:,1].min(), verts[:,1].max(); p0[1] = ymin + (ymax-ymin)*position

        segs = []
        for tri in faces:
            tri_v = verts[tri]
            d = np.dot(tri_v - p0, n)
            if (d<0).any() and (d>0).any():
                pts=[]
                for i in range(3):
                    j=(i+1)%3
                    if d[i]*d[j] < 0:
                        t = d[i]/(d[i]-d[j]+1e-12)
                        pt = tri_v[i] + t*(tri_v[j]-tri_v[i])
                        pts.append(pt)
                if len(pts)==2:
                    segs.append(pts)
        return segs


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DICOMMultiPlanarViewer()
    viewer.show()
    sys.exit(app.exec())
