"""Qt6-compatible dialog for the PickShift plugin.

Only uses qgis.PyQt (never PyQt5/PyQt6 directly) and fully-qualified enums so the
same code runs unchanged on QGIS built against Qt5 or Qt6.
"""

import os

from qgis.core import Qgis, QgsApplication, QgsMapLayerProxyModel, QgsProject, QgsTask, QgsVectorLayer
from qgis.gui import QgsFieldComboBox, QgsFileWidget, QgsMapLayerComboBox, QgsProjectionSelectionWidget
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon, QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .pickshift_task import PickShiftTask


class PickShiftDialog(QDialog):

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._task = None
        self.setWindowTitle("PickShift - Monte-Carlo positional uncertainty")
        self.resize(880, 640)
        self._build_ui()
        self._connect_layer_signals()
        self._apply_style()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _apply_style(self):
        """Light, theme-safe polish: typography and spacing only, no hardcoded
        colors, so it looks correct in both light and dark QGIS themes."""
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                margin-top: 10px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                margin-top: 4px;
            }
            QTabBar::tab {
                min-width: 90px;
                padding: 6px 10px;
            }
            """
        )
        self.run_button.setDefault(True)
        run_font = self.run_button.font()
        run_font.setBold(True)
        self.run_button.setFont(run_font)

    def _build_ui(self):
        outer = QHBoxLayout(self)

        main_widget = QWidget(self)
        root = QVBoxLayout(main_widget)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        tabs = QTabWidget(main_widget)
        root.addWidget(tabs)

        tabs.addTab(self._build_inputs_tab(), "Inputs")
        tabs.addTab(self._build_params_tab(), "Parameters")
        tabs.addTab(self._build_outputs_tab(), "Outputs")

        self.progress_bar = QProgressBar(main_widget)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit(main_widget)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFixedHeight(140)
        root.addWidget(self.log_view)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run", main_widget)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.cancel_button = QPushButton("Cancel run", main_widget)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, main_widget)
        close_box.rejected.connect(self.reject)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch()
        button_row.addWidget(close_box)
        root.addLayout(button_row)

        outer.addWidget(main_widget, 1)
        outer.addWidget(self._build_about_panel(), 0)

    def _build_about_panel(self):
        """Right-hand sidebar: what the plugin does, who wrote it, and the
        scientific reference the Monte-Carlo method is based on."""
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(280)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        def add_label(html, spacing_after=6):
            label = QLabel(html, content)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setOpenExternalLinks(True)
            layout.addWidget(label)
            layout.addSpacing(spacing_after)
            return label

        def add_separator():
            line = QFrame(content)
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(line)
            layout.addSpacing(6)

        add_label("<h3 style='margin-bottom:2px;'>About</h3>")
        add_label(
            "Monte-Carlo estimation of the positional and area uncertainty of "
            "polygons, based on planimetric biases measured at Ground Control "
            "Points (GCPs)."
        )
        add_label(
            "The bias is interpolated (IDW) over the extent to produce an "
            "error surface (SVE), propagated to the polygon nodes, then "
            "simulated over N Monte-Carlo runs to quantify the area "
            "uncertainty of each polygon (total uncertainty and 95% "
            "confidence interval)."
        )

        add_separator()
        add_label(
            "<b>Inputs</b>"
            "<ol style='margin-top:4px; margin-left:-18px;'>"
            "<li><b>GCP layer/table</b> - reference (true) and measured "
            "(digitized) X/Y coordinate fields for your control points.</li>"
            "<li><b>Extent layer</b> - defines the working area for the "
            "interpolated error raster (SVE).</li>"
            "<li><b>Polygon layer</b> - the features to evaluate, with a "
            "unique id field.</li>"
            "</ol>"
            "Hover the &#9432; icons next to each field for details."
        )

        add_separator()
        add_label(
            "<b>QGIS plugin developed by:</b><br/>"
            "<a href='https://romainwenger.fr/'>Romain Wenger</a>"
        )

        add_separator()
        add_label(
            "<b>Paper:</b><br/>"
            "Jautzy, T., Freys, P., Chardon, V., Wenger, R., Rixhon, G., "
            "Schmitt, L., &amp; Herrault, P.-A. (2024). "
            "<i>PickShift: A user-friendly Python tool to assess the surficial "
            "uncertainties associated with polygons extracted from historical "
            "planimetric data.</i> SoftwareX, 27, 101866.<br/>"
            "<a href='https://doi.org/10.1016/j.softx.2024.101866'>"
            "doi.org/10.1016/j.softx.2024.101866</a>"
        )

        add_separator()
        add_label(
            "<b>Source code (original Python script):</b><br/>"
            "<a href='https://github.com/r-wenger/Pickshift'>"
            "github.com/r-wenger/Pickshift</a>"
        )

        layout.addStretch()
        layout.addWidget(self._build_logos_row(content))

        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _build_logos_row(parent):
        plugin_dir = os.path.dirname(__file__)
        row = QWidget(parent)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addStretch()

        icon_label = QLabel(row)
        icon_label.setPixmap(QIcon(os.path.join(plugin_dir, "icon.svg")).pixmap(48, 48))
        h.addWidget(icon_label)

        live_pixmap = QPixmap(os.path.join(plugin_dir, "logo_live.png"))
        if not live_pixmap.isNull():
            live_label = QLabel(row)
            live_label.setPixmap(
                live_pixmap.scaledToHeight(48, Qt.TransformationMode.SmoothTransformation)
            )
            h.addWidget(live_label)

        h.addStretch()
        return row

    @staticmethod
    def _field_label(text, tooltip, parent):
        """A QFormLayout row label with a small (i) info icon carrying a tooltip."""
        container = QWidget(parent)
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(QLabel(text, container))
        info = QLabel("ⓘ", container)  # circled "i"
        info.setStyleSheet("color: #2c6e9e; font-weight: bold;")
        info.setToolTip(tooltip)
        info.setCursor(Qt.CursorShape.WhatsThisCursor)
        h.addWidget(info)
        h.addStretch()
        return container

    def _build_inputs_tab(self):
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 10, 6, 8)
        layout.setSpacing(12)

        gcp_group = QGroupBox("Ground Control Points (planimetric bias)", widget)
        gcp_form = QFormLayout(gcp_group)
        gcp_form.setVerticalSpacing(8)
        gcp_form.setHorizontalSpacing(10)
        self.gcp_layer_combo = QgsMapLayerComboBox(gcp_group)
        self.gcp_layer_combo.setFilters(QgsMapLayerProxyModel.Filter.VectorLayer)
        self.gcp_xref_combo = QgsFieldComboBox(gcp_group)
        self.gcp_yref_combo = QgsFieldComboBox(gcp_group)
        self.gcp_xinit_combo = QgsFieldComboBox(gcp_group)
        self.gcp_yinit_combo = QgsFieldComboBox(gcp_group)
        gcp_form.addRow(self._field_label(
            "Layer / table", "Point layer or attribute table containing your Ground "
            "Control Points (GCPs). Only the 4 fields selected below are read - the "
            "layer's own geometry (if any) is ignored.", gcp_group), self.gcp_layer_combo)
        gcp_form.addRow(self._field_label(
            "Reference X field", "Attribute field holding the known/reference X "
            "coordinate of each GCP (Xref).", gcp_group), self.gcp_xref_combo)
        gcp_form.addRow(self._field_label(
            "Reference Y field", "Attribute field holding the known/reference Y "
            "coordinate of each GCP (Yref).", gcp_group), self.gcp_yref_combo)
        gcp_form.addRow(self._field_label(
            "Measured X field", "Attribute field holding the measured/digitized X "
            "coordinate of each GCP (Xinit), i.e. its position in your source data.",
            gcp_group), self.gcp_xinit_combo)
        gcp_form.addRow(self._field_label(
            "Measured Y field", "Attribute field holding the measured/digitized Y "
            "coordinate of each GCP (Yinit).", gcp_group), self.gcp_yinit_combo)
        layout.addWidget(gcp_group)

        extent_group = QGroupBox("Working extent", widget)
        extent_form = QFormLayout(extent_group)
        extent_form.setVerticalSpacing(8)
        extent_form.setHorizontalSpacing(10)
        self.extent_layer_combo = QgsMapLayerComboBox(extent_group)
        self.extent_layer_combo.setFilters(QgsMapLayerProxyModel.Filter.All)
        extent_form.addRow(self._field_label(
            "Layer defining the extent", "Any layer whose bounding box defines the "
            "working area over which the GCP bias is interpolated (IDW) into the SVE "
            "error raster. Typically your study area boundary.", extent_group),
            self.extent_layer_combo)
        layout.addWidget(extent_group)

        poly_group = QGroupBox("Polygons to evaluate", widget)
        poly_form = QFormLayout(poly_group)
        poly_form.setVerticalSpacing(8)
        poly_form.setHorizontalSpacing(10)
        self.poly_layer_combo = QgsMapLayerComboBox(poly_group)
        self.poly_layer_combo.setFilters(QgsMapLayerProxyModel.Filter.PolygonLayer)
        self.poly_id_combo = QgsFieldComboBox(poly_group)
        poly_form.addRow(self._field_label(
            "Polygon layer", "The polygon features whose positional/area uncertainty "
            "you want to estimate (e.g. features digitized from the same source as "
            "the GCPs).", poly_group), self.poly_layer_combo)
        poly_form.addRow(self._field_label(
            "Unique id field", "Integer field that uniquely identifies each polygon. "
            "Used to group Monte-Carlo results and match them back to the output "
            "layer.", poly_group), self.poly_id_combo)
        layout.addWidget(poly_group)

        layout.addStretch()
        return widget

    def _build_params_tab(self):
        widget = QWidget(self)
        form = QFormLayout(widget)
        form.setContentsMargins(10, 12, 10, 8)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(10)

        self.crs_widget = QgsProjectionSelectionWidget(widget)
        self.crs_widget.setCrs(QgsProject.instance().crs())
        form.addRow(self._field_label(
            "Output CRS", "Coordinate system assigned to all outputs. All inputs "
            "(GCP coordinates, extent, polygons) must already be expressed in this "
            "same CRS/units - no reprojection is performed.", widget), self.crs_widget)

        self.buffer_spin = QDoubleSpinBox(widget)
        self.buffer_spin.setRange(0.0001, 1_000_000)
        self.buffer_spin.setDecimals(4)
        self.buffer_spin.setValue(5.0)
        form.addRow(self._field_label(
            "Node error buffer", "Radius (map units) of the buffer built around each "
            "polygon vertex, used to sample the interpolated bias (SVE) around that "
            "vertex via zonal statistics.", widget), self.buffer_spin)

        self.runs_spin = QSpinBox(widget)
        self.runs_spin.setRange(1, 1_000_000)
        self.runs_spin.setValue(500)
        form.addRow(self._field_label(
            "Monte-Carlo runs", "Number of random simulations. More runs give more "
            "stable uncertainty estimates but take longer to compute - start with a "
            "small number (e.g. 50) to test your setup.", widget), self.runs_spin)

        self.resol_x_spin = QDoubleSpinBox(widget)
        self.resol_x_spin.setRange(0.0001, 1_000_000)
        self.resol_x_spin.setDecimals(4)
        self.resol_x_spin.setValue(1.0)
        form.addRow(self._field_label(
            "SVE raster resolution X", "Pixel width (map units) of the interpolated "
            "bias raster (SVE) generated by IDW over the extent. Finer resolution is "
            "more precise but produces a larger, slower raster.", widget),
            self.resol_x_spin)

        self.resol_y_spin = QDoubleSpinBox(widget)
        self.resol_y_spin.setRange(0.0001, 1_000_000)
        self.resol_y_spin.setDecimals(4)
        self.resol_y_spin.setValue(1.0)
        form.addRow(self._field_label(
            "SVE raster resolution Y", "Pixel height (map units) of the interpolated "
            "bias raster (SVE) generated by IDW over the extent.", widget),
            self.resol_y_spin)

        self.digit_error_spin = QDoubleSpinBox(widget)
        self.digit_error_spin.setRange(0.0, 1_000_000)
        self.digit_error_spin.setDecimals(4)
        self.digit_error_spin.setValue(0.5)
        form.addRow(self._field_label(
            "Digitizing error", "Fixed digitizing error term (map units) added to "
            "the simulated bias at each Monte-Carlo run, independent of the "
            "interpolated GCP bias.", widget), self.digit_error_spin)

        self.douglas_peucker_check = QCheckBox("Simplify simulated polygons (Douglas-Peucker)", widget)
        self.douglas_peucker_check.setToolTip(
            "Simplifies each simulated polygon to avoid topological errors when the "
            "interpolated error is large relative to the spacing between polygon "
            "vertices. The plugin will refuse to run and suggest a tolerance if this "
            "is needed but left disabled."
        )
        self.douglas_peucker_check.toggled.connect(lambda checked: self.tolerance_spin.setEnabled(checked))
        form.addRow(self.douglas_peucker_check)

        self.tolerance_spin = QDoubleSpinBox(widget)
        self.tolerance_spin.setRange(0.0, 1_000_000)
        self.tolerance_spin.setDecimals(4)
        self.tolerance_spin.setEnabled(False)
        form.addRow(self._field_label(
            "Simplification tolerance", "Maximum allowed deviation (map units) when "
            "simplifying simulated polygons (Douglas-Peucker).", widget),
            self.tolerance_spin)

        self.seed_spin = QSpinBox(widget)
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setValue(0)
        form.addRow(self._field_label(
            "Random seed (0 = random)", "Fixes the random number generator so "
            "re-running with the same inputs gives identical results. Leave at 0 "
            "for a different random outcome each run.", widget), self.seed_spin)

        note = QLabel(
            "Note: GCP fields, extent layer and polygon layer are assumed to already share "
            "the same coordinate system/units as the output CRS above (no reprojection is "
            "performed), matching the original PickShift script behaviour.",
            widget,
        )
        note.setWordWrap(True)
        form.addRow(note)

        return widget

    def _build_outputs_tab(self):
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 10, 6, 8)
        layout.setSpacing(12)

        out_group = QGroupBox("Output folder", widget)
        out_form = QFormLayout(out_group)
        out_form.setVerticalSpacing(8)
        out_form.setHorizontalSpacing(10)
        self.output_folder_widget = QgsFileWidget(out_group)
        self.output_folder_widget.setStorageMode(QgsFileWidget.StorageMode.GetDirectory)
        out_form.addRow(self._field_label(
            "Folder", "Directory where all output files (GPKG layers, rasters, "
            "CSVs) will be written. Created automatically if it doesn't exist.",
            out_group), self.output_folder_widget)
        layout.addWidget(out_group)

        flags_group = QGroupBox("Files to keep", widget)
        flags_layout = QVBoxLayout(flags_group)
        flags_layout.setSpacing(6)
        self.export_gcp_bias_check = QCheckBox("GCP_bias.gpkg (GCP bias points)", flags_group)
        self.export_gcp_bias_check.setToolTip(
            "Point layer with each GCP position and its computed planimetric bias "
            "(BiasX, BiasY, BiasXY)."
        )
        self.export_sve_check = QCheckBox("SVE_X.tif / SVE_Y.tif / SVE_XY.tif (interpolated bias rasters)", flags_group)
        self.export_sve_check.setToolTip(
            "Interpolated (IDW) bias rasters over the extent - the basis for the "
            "zonal statistics sampled at each polygon vertex."
        )
        self.export_point_csv_check = QCheckBox("point_sim_MC.csv (per-run simulated points)", flags_group)
        self.export_point_csv_check.setToolTip(
            "One row per simulated point per Monte-Carlo run (run, id, sampled "
            "bias, simulated X/Y). Can be a large file for many runs/polygons."
        )
        self.export_poly_csv_check = QCheckBox("poly_sim_MC.csv (per-run simulated polygon areas)", flags_group)
        self.export_poly_csv_check.setToolTip(
            "One row per simulated polygon per Monte-Carlo run (run, id, area) - "
            "the raw data behind the final uncertainty statistics."
        )
        self.export_gcp_bias_check.setChecked(True)
        self.export_sve_check.setChecked(True)
        for cb in (self.export_gcp_bias_check, self.export_sve_check,
                   self.export_point_csv_check, self.export_poly_csv_check):
            flags_layout.addWidget(cb)
        layout.addWidget(flags_group)

        self.add_to_map_check = QCheckBox("Add poly_MC.gpkg result layer to the map when finished", widget)
        self.add_to_map_check.setToolTip(
            "Loads the final result layer (original polygons + mean/min/max/std "
            "area, initial area, total and 95% uncertainty percentages) into the "
            "QGIS project when the run finishes."
        )
        self.add_to_map_check.setChecked(True)
        layout.addWidget(self.add_to_map_check)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------ #
    # Layer/field wiring
    # ------------------------------------------------------------------ #
    def _connect_layer_signals(self):
        self.gcp_layer_combo.layerChanged.connect(self.gcp_xref_combo.setLayer)
        self.gcp_layer_combo.layerChanged.connect(self.gcp_yref_combo.setLayer)
        self.gcp_layer_combo.layerChanged.connect(self.gcp_xinit_combo.setLayer)
        self.gcp_layer_combo.layerChanged.connect(self.gcp_yinit_combo.setLayer)
        self.poly_layer_combo.layerChanged.connect(self.poly_id_combo.setLayer)

        self.gcp_xref_combo.setLayer(self.gcp_layer_combo.currentLayer())
        self.gcp_yref_combo.setLayer(self.gcp_layer_combo.currentLayer())
        self.gcp_xinit_combo.setLayer(self.gcp_layer_combo.currentLayer())
        self.gcp_yinit_combo.setLayer(self.gcp_layer_combo.currentLayer())
        self.poly_id_combo.setLayer(self.poly_layer_combo.currentLayer())

    # ------------------------------------------------------------------ #
    # Validation + run
    # ------------------------------------------------------------------ #
    def _collect_params(self):
        gcp_layer = self.gcp_layer_combo.currentLayer()
        extent_layer = self.extent_layer_combo.currentLayer()
        poly_layer = self.poly_layer_combo.currentLayer()

        if gcp_layer is None or extent_layer is None or poly_layer is None:
            raise ValueError("Please select a GCP layer, an extent layer and a polygon layer.")

        for label, field in (
            ("reference X", self.gcp_xref_combo.currentField()),
            ("reference Y", self.gcp_yref_combo.currentField()),
            ("measured X", self.gcp_xinit_combo.currentField()),
            ("measured Y", self.gcp_yinit_combo.currentField()),
        ):
            if not field:
                raise ValueError(f"Please select the GCP {label} field.")

        id_field = self.poly_id_combo.currentField()
        if not id_field:
            raise ValueError("Please select the polygon unique id field.")

        output_folder = self.output_folder_widget.filePath()
        if not output_folder:
            raise ValueError("Please choose an output folder.")

        if self.douglas_peucker_check.isChecked() and self.tolerance_spin.value() <= 0:
            raise ValueError("Please set a simplification tolerance greater than 0.")

        seed = self.seed_spin.value()
        return {
            "gcp_layer": gcp_layer,
            "gcp_xref_field": self.gcp_xref_combo.currentField(),
            "gcp_yref_field": self.gcp_yref_combo.currentField(),
            "gcp_xinit_field": self.gcp_xinit_combo.currentField(),
            "gcp_yinit_field": self.gcp_yinit_combo.currentField(),
            "extent_layer": extent_layer,
            "polygons_layer": poly_layer,
            "id_field": id_field,
            "buffer": self.buffer_spin.value(),
            "epsg": self.crs_widget.crs().postgisSrid(),
            "runs": self.runs_spin.value(),
            "resol_x": self.resol_x_spin.value(),
            "resol_y": self.resol_y_spin.value(),
            "digit_error": self.digit_error_spin.value(),
            "douglas_peucker": self.douglas_peucker_check.isChecked(),
            "tolerance": self.tolerance_spin.value(),
            "output_folder": output_folder,
            "export_gcp_bias": self.export_gcp_bias_check.isChecked(),
            "export_sve": self.export_sve_check.isChecked(),
            "export_point_csv": self.export_point_csv_check.isChecked(),
            "export_poly_csv": self.export_poly_csv_check.isChecked(),
            "seed": seed if seed > 0 else None,
        }

    def _on_run_clicked(self):
        try:
            params = self._collect_params()
        except ValueError as exc:
            QMessageBox.warning(self, "PickShift", str(exc))
            return

        if not self.crs_widget.crs().isValid():
            QMessageBox.warning(self, "PickShift", "Please select a valid output CRS.")
            return

        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self._task = PickShiftTask(params)
        self._task.progressChanged.connect(self._on_progress_changed)
        self._task.message_logged.connect(self._append_log)
        self._task.taskCompleted.connect(self._on_task_finished)
        self._task.taskTerminated.connect(self._on_task_finished)
        QgsApplication.taskManager().addTask(self._task)

    def _on_cancel_clicked(self):
        if self._task is not None:
            self._task.cancel()
            self.cancel_button.setEnabled(False)

    def _on_progress_changed(self, progress):
        self.progress_bar.setValue(int(progress))

    def _append_log(self, message):
        self.log_view.appendPlainText(message)

    def _on_task_finished(self):
        task = self._task
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if task is None:
            return

        success = task.status() == QgsTask.TaskStatus.Complete
        if success:
            self._append_log(
                "[DONE] " + " ".join(task.summary_lines)
                if task.summary_lines else "[DONE] Simulation finished."
            )
            if self.add_to_map_check.isChecked():
                poly_mc_path = task.output_paths.get("poly_mc")
                if poly_mc_path and os.path.exists(poly_mc_path):
                    layer = QgsVectorLayer(poly_mc_path, "poly_MC", "ogr")
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
            self.iface.messageBar().pushMessage(
                "PickShift", "Simulation finished successfully.", level=Qgis.MessageLevel.Success
            )
        elif task.status() == QgsTask.TaskStatus.Terminated and task.error_message is None:
            self._append_log("[CANCELLED] Simulation cancelled by user.")
            self.iface.messageBar().pushMessage(
                "PickShift", "Simulation cancelled.", level=Qgis.MessageLevel.Warning
            )
        else:
            self._append_log(f"[ERROR] {task.error_message}")
            self.iface.messageBar().pushMessage(
                "PickShift", f"Simulation failed: {task.error_message}", level=Qgis.MessageLevel.Critical
            )

        self._task = None
