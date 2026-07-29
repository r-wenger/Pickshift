"""
Core PickShift algorithm, ported to run purely on libraries bundled with QGIS:
qgis.core (PyQGIS), osgeo.gdal / osgeo.ogr / osgeo.osr (GDAL/OGR Python bindings
shipped with QGIS) and numpy. No geopandas, fiona, shapely, rasterio, rasterstats
or pandas are used.

The heavy computation runs inside a QgsTask so the QGIS UI stays responsive and
the run can be cancelled from the QGIS Task Manager.
"""

import math
import os
import csv
from collections import defaultdict

import numpy as np
from osgeo import gdal, ogr

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsTask,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType, pyqtSignal

gdal.UseExceptions()
ogr.UseExceptions()

MESSAGE_CATEGORY = "PickShift"


class PickShiftError(Exception):
    pass


class PickShiftTask(QgsTask):
    """Runs the PickShift Monte-Carlo uncertainty simulation in the background."""

    message_logged = pyqtSignal(str)

    def __init__(self, params):
        super().__init__("PickShift Monte-Carlo simulation")
        self.params = params
        self.error_message = None
        self.output_paths = {}
        self.summary_lines = []

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def _log(self, msg):
        QgsMessageLog.logMessage(msg, MESSAGE_CATEGORY)
        self.message_logged.emit(msg)

    # ------------------------------------------------------------------ #
    # QgsTask entry point
    # ------------------------------------------------------------------ #
    def run(self):
        try:
            self._run_impl()
        except PickShiftError as exc:
            self.error_message = str(exc)
            return False
        except Exception as exc:  # noqa: BLE001 - surfaced to the user in the dialog
            self.error_message = f"Unexpected error: {exc}"
            return False
        # `_run_impl` returns early (without raising) when the task is
        # cancelled mid-way; make sure that is reported as a non-success
        # so QgsTask ends up in the Terminated state rather than Complete.
        return not self.isCanceled()

    def finished(self, result):
        if result:
            self._log("[OK] PickShift simulation finished successfully.")
        else:
            self._log(f"[ERROR] PickShift simulation failed: {self.error_message}")

    # ------------------------------------------------------------------ #
    # Main algorithm
    # ------------------------------------------------------------------ #
    def _run_impl(self):
        p = self.params
        seed = p.get("seed")
        if seed is not None:
            np.random.seed(seed)

        output_folder = p["output_folder"]
        os.makedirs(output_folder, exist_ok=True)

        crs = QgsCoordinateReferenceSystem(f"EPSG:{p['epsg']}")
        if not crs.isValid():
            raise PickShiftError(f"EPSG:{p['epsg']} is not a valid CRS.")
        transform_context = QgsProject.instance().transformContext()

        # -------------------------------------------------------------- #
        # 1. Read GCP data and compute planimetric biases
        # -------------------------------------------------------------- #
        self._log("[INFO] Reading GCP data and computing planimetric biases...")
        self.setProgress(2)
        xref, yref, xinit, yinit = self._read_gcp()
        bias_xy = np.sqrt((xref - xinit) ** 2 + (yref - yinit) ** 2)
        bias_x = np.abs(xref - xinit)
        bias_y = np.abs(yref - yinit)
        self._log(
            "[INFO] XY: mean bias = %.4f, std = %.4f | X: mean = %.4f, std = %.4f | "
            "Y: mean = %.4f, std = %.4f"
            % (
                np.mean(bias_xy), np.std(bias_xy, ddof=1),
                np.mean(bias_x), np.std(bias_x, ddof=1),
                np.mean(bias_y), np.std(bias_y, ddof=1),
            )
        )
        if self.isCanceled():
            return

        # -------------------------------------------------------------- #
        # 2. Write GCP bias points to GPKG
        # -------------------------------------------------------------- #
        gcp_bias_path = os.path.join(output_folder, "GCP_bias.gpkg")
        self._write_point_layer(
            gcp_bias_path,
            crs,
            fields=[("fid_gcp", QMetaType.Type.Int),
                    ("BiasXY", QMetaType.Type.Double),
                    ("BiasX", QMetaType.Type.Double),
                    ("BiasY", QMetaType.Type.Double)],
            rows=[
                (i, float(xinit[i]), float(yinit[i]),
                 [i, float(bias_xy[i]), float(bias_x[i]), float(bias_y[i])])
                for i in range(len(xinit))
            ],
        )
        self.output_paths["gcp_bias"] = gcp_bias_path
        self.setProgress(8)

        # -------------------------------------------------------------- #
        # 3. Interpolate biases (IDW) over the extent -> SVE rasters
        # -------------------------------------------------------------- #
        self._log("[INFO] Interpolating biases over the extent (IDW) to produce SVE rasters...")
        extent_layer = p["extent_layer"]
        ext = extent_layer.extent()
        xmin, ymin, xmax, ymax = ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum()
        width = max(1, int(round(abs((xmax - xmin) / p["resol_x"]))))
        height = max(1, int(round(abs((ymax - ymin) / p["resol_y"]))))

        output_srs = f"EPSG:{p['epsg']}"
        sve_xy_path = self._idw_grid(gcp_bias_path, output_folder, "BiasXY", "SVE_XY.tif",
                                      xmin, ymin, xmax, ymax, width, height, output_srs)
        sve_x_path = self._idw_grid(gcp_bias_path, output_folder, "BiasX", "SVE_X.tif",
                                     xmin, ymin, xmax, ymax, width, height, output_srs)
        sve_y_path = self._idw_grid(gcp_bias_path, output_folder, "BiasY", "SVE_Y.tif",
                                     xmin, ymin, xmax, ymax, width, height, output_srs)
        self._log("[OUTPUT] SVE_XY.tif, SVE_X.tif and SVE_Y.tif created.")
        self.setProgress(20)
        if self.isCanceled():
            return

        # -------------------------------------------------------------- #
        # 4. Explode polygons into single parts + compute mean node spacing
        # -------------------------------------------------------------- #
        polygons_layer = p["polygons_layer"]
        id_field = p["id_field"]
        singleparts = self._explode_polygons(polygons_layer, id_field)
        if not singleparts:
            raise PickShiftError("The polygon layer produced no usable single-part polygons.")

        mean_pixel_error = self._raster_mean(sve_xy_path)
        avg_dist_per_poly = []
        for _pid, ring in singleparts:
            dists = [ring[i - 1].distance(ring[i]) for i in range(1, len(ring))]
            if dists:
                avg_dist_per_poly.append(np.mean(dists))
        final_average_distance = float(np.mean(avg_dist_per_poly)) if avg_dist_per_poly else 0.0
        ratio = (mean_pixel_error / final_average_distance) if final_average_distance else 0.0
        self._log(f"[INFO] Ratio between mean SVE and mean node spacing: {ratio:.4f}")

        if ratio >= 1 and not p["douglas_peucker"]:
            suggestion = int(final_average_distance * 2)
            raise PickShiftError(
                "Monte-Carlo simulations would very likely produce topological errors "
                f"(SVE/node-spacing ratio = {ratio:.2f} >= 1) and Douglas-Peucker "
                "simplification is disabled. Enable it with a tolerance of about "
                f"{suggestion} (or more) and run again."
            )
        self.setProgress(25)

        # -------------------------------------------------------------- #
        # 5. Extract polygon nodes, buffer them, write Buffer.gpkg
        # -------------------------------------------------------------- #
        self._log("[INFO] Extracting polygon nodes and building error buffers...")
        nodes = self._extract_nodes(singleparts)
        num_reps = len(nodes)
        if num_reps == 0:
            raise PickShiftError("No polygon nodes could be extracted.")

        buffer_dist = p["buffer"]
        buffer_geoms = [
            QgsGeometry.fromPointXY(QgsPointXY(n["x"], n["y"])).buffer(buffer_dist, 12)
            for n in nodes
        ]
        buffer_path = os.path.join(output_folder, "Buffer.gpkg")
        self._write_buffer_layer(buffer_path, crs, nodes, buffer_geoms)
        self.output_paths["buffer"] = buffer_path
        self.setProgress(30)
        if self.isCanceled():
            return

        # -------------------------------------------------------------- #
        # 6. Zonal statistics (mean/std) of SVE rasters within each buffer
        # -------------------------------------------------------------- #
        self._log("[INFO] Computing zonal statistics for each node buffer...")
        stats_x = self._zonal_stats(sve_x_path, buffer_geoms, progress_start=30, progress_end=45)
        if self.isCanceled():
            return
        stats_y = self._zonal_stats(sve_y_path, buffer_geoms, progress_start=45, progress_end=60)
        if self.isCanceled():
            return
        stats_xy = self._zonal_stats(sve_xy_path, buffer_geoms, progress_start=60, progress_end=75)
        if self.isCanceled():
            return

        ids = np.array([n["id"] for n in nodes])
        x0 = np.array([n["x"] for n in nodes])
        y0 = np.array([n["y"] for n in nodes])
        mean_x = np.nan_to_num(np.array([stats_x[i][0] for i in range(num_reps)]))
        std_x = np.nan_to_num(np.array([stats_x[i][1] for i in range(num_reps)]))
        mean_y = np.nan_to_num(np.array([stats_y[i][0] for i in range(num_reps)]))
        std_y = np.nan_to_num(np.array([stats_y[i][1] for i in range(num_reps)]))
        mean_xy = np.nan_to_num(np.array([stats_xy[i][0] for i in range(num_reps)]))
        std_xy = np.nan_to_num(np.array([stats_xy[i][1] for i in range(num_reps)]))

        # -------------------------------------------------------------- #
        # 7. Monte-Carlo translations
        # -------------------------------------------------------------- #
        self._log(f"[INFO] Running {p['runs']} Monte-Carlo translations...")
        id_to_indices = defaultdict(list)
        for idx, pid in enumerate(ids):
            id_to_indices[pid].append(idx)

        prob = [-1, 1]
        runs = p["runs"]
        erreur = p["digit_error"]
        point_rows = []  # (run, id, esv_x, esv_y, esv_xy, x, y)

        for i in range(runs):
            if self.isCanceled():
                return
            xgaus = np.random.normal(mean_x, std_x, num_reps)
            ygaus = np.random.normal(mean_y, std_y, num_reps)
            xygaus = np.random.normal(mean_xy, std_xy, num_reps)

            beta1_x = np.empty(num_reps)
            beta1_y = np.empty(num_reps)
            for _pid, idxs in id_to_indices.items():
                beta1_x[idxs] = np.random.choice(prob, size=1)[0]
                beta1_y[idxs] = np.random.choice(prob, size=len(idxs))

            beta2_x = np.random.choice(prob, size=num_reps)
            beta2_y = np.random.choice(prob, size=num_reps)

            x1 = x0 + xgaus * beta1_x + erreur * beta2_x
            y1 = y0 + ygaus * beta1_y + erreur * beta2_y

            for j in range(num_reps):
                point_rows.append((i, int(ids[j]), xgaus[j], ygaus[j], xygaus[j], x1[j], y1[j]))

            if runs > 0 and i % max(1, runs // 20) == 0:
                self.setProgress(75 + 15 * (i + 1) / runs)

        self.setProgress(90)
        if self.isCanceled():
            return

        # -------------------------------------------------------------- #
        # 8. Rebuild polygons per (run, id) and compute area uncertainty
        # -------------------------------------------------------------- #
        self._log("[INFO] Rebuilding simulated polygons and computing area uncertainty...")
        groups = defaultdict(list)
        for run, pid, _ex, _ey, _exy, x, y in point_rows:
            groups[(run, pid)].append((x, y))

        douglas_peucker = p["douglas_peucker"]
        tolerance = p["tolerance"]
        areas_by_id = defaultdict(list)
        poly_sim_rows = []  # (run, id, area)

        for (run, pid), pts in groups.items():
            ring = [QgsPointXY(x, y) for x, y in pts]
            if ring[0] != ring[-1]:
                ring.append(QgsPointXY(ring[0]))
            geom = QgsGeometry.fromPolygonXY([ring])
            if douglas_peucker:
                geom = geom.simplify(tolerance)
            area = geom.area()
            areas_by_id[pid].append(area)
            poly_sim_rows.append((run, pid, area))

        initial_area_by_id = defaultdict(float)
        for pid, ring in singleparts:
            initial_area_by_id[pid] += QgsGeometry.fromPolygonXY([ring]).area()

        stats_by_id = {}
        for pid, areas in areas_by_id.items():
            arr = np.array(areas)
            mean_area = float(np.mean(arr))
            min_area = float(np.min(arr))
            max_area = float(np.max(arr))
            std_area = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            q2 = float(np.percentile(arr, 2.5))
            q9 = float(np.percentile(arr, 97.5))
            total_uncertainty = (0.5 * (max_area - min_area) / mean_area * 100) if mean_area else 0.0
            uncertainty_95 = (0.5 * (q9 - q2) / mean_area * 100) if mean_area else 0.0
            stats_by_id[pid] = {
                "mean_area": mean_area,
                "min_area": min_area,
                "max_area": max_area,
                "std_area": std_area,
                "initial_area": initial_area_by_id.get(pid, 0.0),
                "total_uncertainty_pct": total_uncertainty,
                "uncertainty_95_pct": uncertainty_95,
            }

        self.setProgress(95)

        # -------------------------------------------------------------- #
        # 9. Write outputs
        # -------------------------------------------------------------- #
        self._log("[INFO] Writing outputs...")
        poly_mc_path = os.path.join(output_folder, "poly_MC.gpkg")
        self._write_result_polygons(poly_mc_path, polygons_layer, id_field, stats_by_id,
                                     transform_context, crs)
        self.output_paths["poly_mc"] = poly_mc_path
        self._log("[OUTPUT] poly_MC.gpkg created.")

        if p["export_point_csv"]:
            path = os.path.join(output_folder, "point_sim_MC.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["run", "id", "ESV_X", "ESV_Y", "ESV_XY", "X", "Y"])
                writer.writerows(point_rows)
            self.output_paths["point_csv"] = path
            self._log("[OUTPUT] point_sim_MC.csv created.")

        if p["export_poly_csv"]:
            path = os.path.join(output_folder, "poly_sim_MC.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["run", "id", "Area"])
                writer.writerows(poly_sim_rows)
            self.output_paths["poly_csv"] = path
            self._log("[OUTPUT] poly_sim_MC.csv created.")

        if not p["export_sve"]:
            for path in (sve_xy_path, sve_x_path, sve_y_path):
                if os.path.exists(path):
                    os.remove(path)
        else:
            self.output_paths["sve_xy"] = sve_xy_path
            self.output_paths["sve_x"] = sve_x_path
            self.output_paths["sve_y"] = sve_y_path

        if not p["export_gcp_bias"]:
            if os.path.exists(gcp_bias_path):
                os.remove(gcp_bias_path)
            self.output_paths.pop("gcp_bias", None)

        self.setProgress(100)
        self.summary_lines.append(f"{len(stats_by_id)} polygon(s) processed over {runs} run(s).")

    # ------------------------------------------------------------------ #
    # Step helpers
    # ------------------------------------------------------------------ #
    def _read_gcp(self):
        p = self.params
        layer = p["gcp_layer"]
        fields = layer.fields()
        idx = {
            "xref": fields.indexOf(p["gcp_xref_field"]),
            "yref": fields.indexOf(p["gcp_yref_field"]),
            "xinit": fields.indexOf(p["gcp_xinit_field"]),
            "yinit": fields.indexOf(p["gcp_yinit_field"]),
        }
        for key, i in idx.items():
            if i < 0:
                raise PickShiftError(f"GCP field for '{key}' not found on the selected layer.")

        xref, yref, xinit, yinit = [], [], [], []
        for feat in layer.getFeatures():
            xref.append(float(feat[idx["xref"]]))
            yref.append(float(feat[idx["yref"]]))
            xinit.append(float(feat[idx["xinit"]]))
            yinit.append(float(feat[idx["yinit"]]))
        if not xref:
            raise PickShiftError("The GCP layer contains no features.")
        return np.array(xref), np.array(yref), np.array(xinit), np.array(yinit)

    def _write_point_layer(self, path, crs, fields, rows):
        """rows: list of (id, x, y, attribute_list)"""
        layer = QgsVectorLayer(f"Point?crs={crs.authid()}", "layer", "memory")
        prov = layer.dataProvider()
        prov.addAttributes([QgsField(name, qtype) for name, qtype in fields])
        layer.updateFields()

        feats = []
        for _id, x, y, attrs in rows:
            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            feat.setAttributes(attrs)
            feats.append(feat)
        prov.addFeatures(feats)
        layer.updateExtents()

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, path, QgsProject.instance().transformContext(), options
        )
        err = result[0] if isinstance(result, tuple) else result
        if err != QgsVectorFileWriter.NoError:
            raise PickShiftError(f"Could not write {path} ({result}).")

    def _write_buffer_layer(self, path, crs, nodes, buffer_geoms):
        layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}", "Buffer", "memory")
        prov = layer.dataProvider()
        prov.addAttributes([QgsField("id", QMetaType.Type.Int)])
        layer.updateFields()

        feats = []
        for node, geom in zip(nodes, buffer_geoms):
            feat = QgsFeature(layer.fields())
            feat.setGeometry(geom)
            feat.setAttributes([int(node["id"])])
            feats.append(feat)
        prov.addFeatures(feats)
        layer.updateExtents()

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, path, QgsProject.instance().transformContext(), options
        )
        err = result[0] if isinstance(result, tuple) else result
        if err != QgsVectorFileWriter.NoError:
            raise PickShiftError(f"Could not write {path} ({result}).")

    def _idw_grid(self, gcp_bias_path, output_folder, zfield, out_name,
                  xmin, ymin, xmax, ymax, width, height, output_srs):
        out_path = os.path.join(output_folder, out_name)
        ds = gdal.Grid(
            out_path,
            gcp_bias_path,
            zfield=zfield,
            algorithm="invdist:power=2:smoothing=1.0",
            outputBounds=[xmin, ymax, xmax, ymin],
            width=width,
            height=height,
            outputSRS=output_srs,
        )
        if ds is None:
            raise PickShiftError(f"gdal.Grid failed to produce {out_name}.")
        ds = None
        return out_path

    def _raster_mean(self, path):
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(float)
        nodata = band.GetNoDataValue()
        ds = None
        if nodata is not None:
            arr = np.ma.masked_equal(arr, nodata)
        return float(np.mean(arr))

    def _explode_polygons(self, polygons_layer, id_field):
        """Returns a list of (id, exterior_ring[QgsPointXY,...]) for every single
        polygon part (multi-part features are split, matching geopandas .explode())."""
        idx = polygons_layer.fields().indexOf(id_field)
        if idx < 0:
            raise PickShiftError(f"Field '{id_field}' not found on the polygon layer.")

        singleparts = []
        for feat in polygons_layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            pid = int(feat[idx])
            if geom.isMultipart():
                for poly in geom.asMultiPolygon():
                    singleparts.append((pid, poly[0]))
            else:
                poly = geom.asPolygon()
                if poly:
                    singleparts.append((pid, poly[0]))
        return singleparts

    def _extract_nodes(self, singleparts):
        seen = set()
        nodes = []
        for pid, ring in singleparts:
            vertices = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
            for pt in vertices:
                key = (pid, round(pt.x(), 6), round(pt.y(), 6))
                if key in seen:
                    continue
                seen.add(key)
                nodes.append({"id": pid, "x": pt.x(), "y": pt.y()})
        return nodes

    def _zonal_stats(self, raster_path, geometries, progress_start=0, progress_end=0):
        """Pure GDAL/numpy zonal mean+std (population std, matching rasterstats),
        one polygon at a time using a small in-memory rasterized mask."""
        ds = gdal.Open(raster_path)
        gt = ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        xsize, ysize = ds.RasterXSize, ds.RasterYSize

        mem_vec_drv = ogr.GetDriverByName("Memory")
        mem_rast_drv = gdal.GetDriverByName("MEM")

        results = {}
        n = len(geometries)
        for i, geom in enumerate(geometries):
            if self.isCanceled():
                break
            rect = geom.boundingBox()
            px0, py0 = gdal.ApplyGeoTransform(inv_gt, rect.xMinimum(), rect.yMaximum())
            px1, py1 = gdal.ApplyGeoTransform(inv_gt, rect.xMaximum(), rect.yMinimum())
            col_min = max(0, int(math.floor(min(px0, px1))))
            row_min = max(0, int(math.floor(min(py0, py1))))
            col_max = min(xsize, int(math.ceil(max(px0, px1))) + 1)
            row_max = min(ysize, int(math.ceil(max(py0, py1))) + 1)
            win_w, win_h = col_max - col_min, row_max - row_min

            if win_w <= 0 or win_h <= 0:
                results[i] = (float("nan"), float("nan"))
                continue

            window_gt = (
                gt[0] + col_min * gt[1], gt[1], gt[2],
                gt[3] + row_min * gt[5], gt[4], gt[5],
            )

            # No CRS is set on the mask layer/raster on purpose: rasterization only
            # relies on the pixel<->world geotransform below, and all inputs are
            # already assumed to share the same CRS (no reprojection is performed
            # anywhere in this plugin).
            vec_ds = mem_vec_drv.CreateDataSource("mem")
            ogr_layer = vec_ds.CreateLayer("mask", geom_type=ogr.wkbPolygon)
            ogr_feat = ogr.Feature(ogr_layer.GetLayerDefn())
            ogr_feat.SetGeometry(ogr.CreateGeometryFromWkt(geom.asWkt()))
            ogr_layer.CreateFeature(ogr_feat)

            mask_ds = mem_rast_drv.Create("", win_w, win_h, 1, gdal.GDT_Byte)
            mask_ds.SetGeoTransform(window_gt)
            gdal.RasterizeLayer(mask_ds, [1], ogr_layer, burn_values=[1])
            mask_arr = mask_ds.GetRasterBand(1).ReadAsArray()

            data_arr = band.ReadAsArray(col_min, row_min, win_w, win_h).astype(float)
            valid = mask_arr == 1
            if nodata is not None:
                valid &= data_arr != nodata

            values = data_arr[valid]
            if values.size == 0:
                results[i] = (float("nan"), float("nan"))
            else:
                results[i] = (float(np.mean(values)), float(np.std(values)))

            mask_ds = None
            vec_ds = None

            if progress_end > progress_start and n:
                self.setProgress(progress_start + (progress_end - progress_start) * (i + 1) / n)

        ds = None
        return results

    def _write_result_polygons(self, path, polygons_layer, id_field, stats_by_id,
                                transform_context, crs):
        layer = QgsVectorLayer(f"MultiPolygon?crs={crs.authid()}", "poly_MC", "memory")
        prov = layer.dataProvider()

        extra_fields = [
            QgsField("mean_area", QMetaType.Type.Double),
            QgsField("min_area", QMetaType.Type.Double),
            QgsField("max_area", QMetaType.Type.Double),
            QgsField("std_area", QMetaType.Type.Double),
            QgsField("initial_area", QMetaType.Type.Double),
            QgsField("total_uncert_pct", QMetaType.Type.Double),
            QgsField("uncert_95_pct", QMetaType.Type.Double),
        ]
        prov.addAttributes(list(polygons_layer.fields()) + extra_fields)
        layer.updateFields()

        idx = polygons_layer.fields().indexOf(id_field)
        feats = []
        for src_feat in polygons_layer.getFeatures():
            pid = int(src_feat[idx])
            stats = stats_by_id.get(pid)
            if stats is None:
                continue
            feat = QgsFeature(layer.fields())
            geom = src_feat.geometry()
            if not geom.isMultipart():
                geom.convertToMultiType()
            feat.setGeometry(geom)
            feat.setAttributes(
                list(src_feat.attributes()) + [
                    stats["mean_area"], stats["min_area"], stats["max_area"],
                    stats["std_area"], stats["initial_area"],
                    stats["total_uncertainty_pct"], stats["uncertainty_95_pct"],
                ]
            )
            feats.append(feat)
        prov.addFeatures(feats)
        layer.updateExtents()

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(layer, path, transform_context, options)
        err = result[0] if isinstance(result, tuple) else result
        if err != QgsVectorFileWriter.NoError:
            raise PickShiftError(f"Could not write {path} ({result}).")
