# v7 — NaN→0 fix; 200m pop = sum(children) + hierarchy columns
# grid3_dual_res_visit_report.py
"""
Grid3 Dual-Resolution Visit Report (100m + 200m)

- 100m & 200m tabs in one run
- 200m "visited cells" are derived from 100m indices (row//2, col//2)
- 200m population is computed as the sum of its four 100m children (not from a block-reduced raster)
- Extra hierarchy columns:
  * 100m grid: parent_200m_cell_id, position_in_block
  * 200m grid: child_100m_cell_ids, child_100m_population_breakdown
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer

from .base_report import BaseReport


class Grid3DualResVisitReport(BaseReport):
    """Generates both 100m and 200m visit/population tabs in a single workbook."""

    # ---------------- UI ----------------
    @staticmethod
    def setup_parameters(parent_frame):
        ttk.Label(parent_frame, text="Grid3 TIF file:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        grid3_file_var = tk.StringVar()

        # Try to load last-used file
        config_file = os.path.join(os.path.expanduser("~"), ".grid3_dual_last_file.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    last = f.read().strip()
                    if os.path.exists(last):
                        grid3_file_var.set(last)
            except Exception:
                pass

        file_frame = ttk.Frame(parent_frame)
        file_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Entry(file_frame, textvariable=grid3_file_var, width=44).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(
            file_frame,
            text="Browse...",
            command=lambda: Grid3DualResVisitReport._browse_and_save_grid3_file_static(grid3_file_var, config_file),
        ).grid(row=0, column=1, padx=(6, 0))
        file_frame.columnconfigure(0, weight=1)

        # Density thresholds
        ttk.Label(parent_frame, text="Population density thresholds (per km²):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=4
        )
        density_frame = ttk.Frame(parent_frame)
        density_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(density_frame, text="Low–Medium:").pack(side=tk.LEFT)
        low_medium_var = tk.StringVar(value="10")
        ttk.Entry(density_frame, textvariable=low_medium_var, width=8).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(density_frame, text="Medium–High:").pack(side=tk.LEFT)
        medium_high_var = tk.StringVar(value="50")
        ttk.Entry(density_frame, textvariable=medium_high_var, width=8).pack(side=tk.LEFT, padx=(2, 0))

        parent_frame.grid3_file_var = grid3_file_var
        parent_frame.low_medium_var = low_medium_var
        parent_frame.medium_high_var = medium_high_var

    @staticmethod
    def _browse_and_save_grid3_file_static(var, config_file):
        filename = filedialog.askopenfilename(
            title="Select Grid3 TIF file",
            filetypes=[("TIF files", "*.tif"), ("All files", "*.*")],
        )
        if filename:
            var.set(filename)
            try:
                with open(config_file, "w") as f:
                    f.write(filename)
            except Exception:
                pass

    # ---------------- Main entry ----------------
    def generate(self):
        output_files = []

        # Params
        grid3_file = self.get_parameter_value("grid3_file", "")
        low_medium_threshold = float(self.get_parameter_value("low_medium", "10"))
        medium_high_threshold = float(self.get_parameter_value("medium_high", "50"))

        if not grid3_file or not os.path.exists(grid3_file):
            grid3_file = self._find_grid3_file()
        if not grid3_file:
            raise ValueError("No Grid3 TIF file specified or found.")

        self.log(f"[dual] Using Grid3 file: {os.path.basename(grid3_file)}")

        # Prepare visits once
        visits_df = self._prepare_visit_data()
        self.log(f"[dual] Visits loaded: {len(visits_df)}")

        # Load the 100m raster
        arr100, T100, raster_crs, nodata = self._load_grid3_raster(grid3_file)
        self.log(f"[dual] 100m raster shape: {arr100.shape}")

        # ---- 100m path ----
        mapped100 = self._map_visits_to_indices(visits_df, T100, raster_crs, grid_shape=arr100.shape, grid3_file_path=grid3_file)
        grid100, opp100 = self._build_tabs_100m(
            mapped100, arr100, T100, raster_crs, nodata,
            low_thr=low_medium_threshold, high_thr=medium_high_threshold
        )
        self.log(f"[dual] 100m: grid_rows={len(grid100)}, opp_rows={len(opp100)}")

        # ---- 200m path ----
        mapped200 = mapped100.copy()
        mapped200['row'] = (mapped200['row'] // 2).astype('int64')
        mapped200['col'] = (mapped200['col'] // 2).astype('int64')
        grid200, opp200 = self._build_tabs_200m_from_children(
            mapped200, arr100, T100, raster_crs, nodata,
            low_thr=low_medium_threshold, high_thr=medium_high_threshold
        )
        self.log(f"[dual] 200m: grid_rows={len(grid200)}, opp_rows={len(opp200)}")

        # ---- Opportunity_Compare ----
        compare = self._make_compare_tab(opp100, opp200)

        # Create output directory
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"grid3_dual_{today}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dynamic filename based on inputs
        grid3_filename = os.path.basename(grid3_file)
        grid3_prefix = grid3_filename[:3] if grid3_filename else 'grid'
        
        opp_count = visits_df['opportunity_id'].nunique() if 'opportunity_id' in visits_df.columns else 0
        visit_count = len(visits_df)
        
        excel_filename = f"grid3_dual_{grid3_prefix}_{opp_count}opps_{visit_count}visits.xlsx"
        excel_file = os.path.join(output_dir, excel_filename)

        # Write one workbook with five tabs
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            grid100.to_excel(writer, sheet_name="Grid_Cells_100m", index=False)
            opp100.to_excel(writer, sheet_name="Opportunity_Summary_100m", index=False)
            grid200.to_excel(writer, sheet_name="Grid_Cells_200m", index=False)
            opp200.to_excel(writer, sheet_name="Opportunity_Summary_200m", index=False)
            compare.to_excel(writer, sheet_name="Opportunity_Compare", index=False)

        self.log(f"[dual] Complete. Wrote: {os.path.basename(excel_file)}")
        output_files.append(excel_file)
        return output_files

    # ---------------- Helpers ----------------
    def _find_grid3_file(self):
        search_paths = [
            Path(self.output_dir).parent / "data" / "grid3",
            Path(self.output_dir).parent / "data" / "grid3" / "population",
            Path(self.output_dir) / ".." / "data" / "grid3",
        ]
        for p in search_paths:
            if p.exists():
                tifs = list(p.rglob("*.tif"))
                if tifs:
                    self.log(f"[dual] Auto-detected Grid3 file: {tifs[0]}")
                    return str(tifs[0])
        return None

    def _prepare_visit_data(self):
        data = self.df.copy()
        data.columns = data.columns.str.lower().str.strip()

        col_map = {
            "latitude": ["latitude", "lat", "y"],
            "longitude": ["longitude", "lon", "lng", "x"],
            "opportunity_id": ["opportunity_id"],
            "opportunity_name": ["opportunity_name", "opportunity"],
        }

        def pick(cands):
            for c in cands:
                if c in data.columns:
                    return c
            return None

        lat_col = pick(col_map["latitude"])
        lon_col = pick(col_map["longitude"])
        if lat_col is None or lon_col is None:
            raise ValueError("Visit data must include latitude/longitude.")

        rename = {lat_col: "latitude", lon_col: "longitude"}
        oid = pick(col_map["opportunity_id"]) or "opportunity_id"
        oname = pick(col_map["opportunity_name"]) or None
        if oid in data.columns:
            rename[oid] = "opportunity_id"
        if oname and (oname in data.columns):
            rename[oname] = "opportunity_name"
        data = data.rename(columns=rename)

        data = data.dropna(subset=["latitude", "longitude"]).copy()
        valid = (data["latitude"].between(-90, 90, inclusive="both")) & (data["longitude"].between(-180, 180, inclusive="both"))
        data = data[valid].reset_index(drop=True)

        if "opportunity_id" not in data.columns:
            data["opportunity_id"] = "ALL"
        if "opportunity_name" not in data.columns:
            data["opportunity_name"] = data["opportunity_id"]

        return data

    def _load_grid3_raster(self, grid3_file):
        with rasterio.open(grid3_file) as src:
            arr = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            self.log(f"[dual] Raster meta: {src.width}x{src.height} | CRS={crs} | nodata={nodata}")
        return arr, transform, crs, nodata

    def _map_visits_to_indices(self, visits_df, transform: Affine, raster_crs,
                               grid_shape=None, grid3_file_path: str = None):
        # WGS84 -> raster CRS if needed
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            xs = visits_df["longitude"].to_numpy(np.float64)
            ys = visits_df["latitude"].to_numpy(np.float64)
        else:
            tfm = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            xs, ys = tfm.transform(
                visits_df["longitude"].to_numpy(np.float64),
                visits_df["latitude"].to_numpy(np.float64),
            )

        invT = ~transform
        cols_f, rows_f = invT * (xs, ys)
        cols = np.floor(cols_f).astype(np.int64)
        rows = np.floor(rows_f).astype(np.int64)

        if grid_shape is not None:
            H, W = grid_shape
        else:
            with rasterio.open(grid3_file_path) as src:
                H, W = src.height, src.width

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

        mapped = visits_df.loc[in_bounds, ["opportunity_id", "opportunity_name"]].copy()
        mapped["row"] = rows[in_bounds]
        mapped["col"] = cols[in_bounds]

        dropped = (~in_bounds).sum()
        if dropped:
            self.log(f"[dual] Dropped {dropped} visits outside raster bounds (res-specific)")
        return mapped

    # ---------- Attribute builders ----------
    def _cells_to_attributes_100m(self, unique_cells_df, array, transform: Affine, raster_crs, nodata):
        rows = unique_cells_df["row"].to_numpy(np.int64)
        cols = unique_cells_df["col"].to_numpy(np.int64)

        pop = array[rows, cols].astype(float)
        if nodata is not None:
            pop = np.where(pop == nodata, np.nan, pop)
        pop = np.nan_to_num(pop, nan=0.0)

        # centers
        cx = transform * (cols + 0.5, rows + 0.5)
        cx_x = np.asarray(cx[0], dtype=np.float64)
        cx_y = np.asarray(cx[1], dtype=np.float64)
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            lons = cx_x
            lats = cx_y
        else:
            tfm = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
            lons, lats = tfm.transform(cx_x, cx_y)

        out = unique_cells_df.copy()
        out["population"] = pop
        out["center_longitude"] = lons
        out["center_latitude"] = lats
        out["cell_id"] = "cell_100m_" + out["row"].astype(str) + "_" + out["col"].astype(str)

        # Hierarchy helpers
        prow = (out["row"] // 2).astype(int)
        pcol = (out["col"] // 2).astype(int)
        out["parent_200m_cell_id"] = "cell_200m_" + prow.astype(str) + "_" + pcol.astype(str)
        dr = (out["row"] % 2).astype(int)
        dc = (out["col"] % 2).astype(int)
        pos = np.where((dr==0) & (dc==0), "top_left",
              np.where((dr==0) & (dc==1), "top_right",
              np.where((dr==1) & (dc==0), "bottom_left", "bottom_right")))
        out["position_in_block"] = pos

        return out

    def _cells_to_attributes_200m_from_children(self, unique_parents_df, array100, T100: Affine, raster_crs, nodata):
        """Compute 200m attributes by summing four 100m children derived from indices."""
        prow = unique_parents_df["row"].to_numpy(np.int64)
        pcol = unique_parents_df["col"].to_numpy(np.int64)

        r0 = (2 * prow).astype(np.int64)
        c0 = (2 * pcol).astype(np.int64)

        H, W = array100.shape
        # child indices with bounds checks
        r00, c00 = r0, c0
        r01, c01 = r0, c0 + 1
        r10, c10 = r0 + 1, c0
        r11, c11 = r0 + 1, c0 + 1

        def safe_pick(r, c):
            inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            rr = np.where(inb, r, 0)
            cc = np.where(inb, c, 0)
            vals = array100[rr, cc].astype(float)
            # Treat both explicit nodata and NaN as 0.0
            if nodata is not None:
                vals = np.where(vals == nodata, 0.0, vals)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = np.where(inb, vals, 0.0)
            return vals

        p00 = safe_pick(r00, c00)
        p01 = safe_pick(r01, c01)
        p10 = safe_pick(r10, c10)
        p11 = safe_pick(r11, c11)

        pop = p00 + p01 + p10 + p11

        # centers from T200 = T100 * scale(2,2)
        T200 = T100 * Affine.scale(2, 2)
        cx = T200 * (pcol + 0.5, prow + 0.5)
        cx_x = np.asarray(cx[0], dtype=np.float64)
        cx_y = np.asarray(cx[1], dtype=np.float64)
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            lons = cx_x
            lats = cx_y
        else:
            tfm = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
            lons, lats = tfm.transform(cx_x, cx_y)

        out = unique_parents_df.copy()
        out["population"] = pop
        out["center_longitude"] = lons
        out["center_latitude"] = lats
        out["cell_id"] = "cell_200m_" + out["row"].astype(str) + "_" + out["col"].astype(str)

        # child ids + breakdown strings
        id00 = "cell_100m_" + r00.astype(str) + "_" + c00.astype(str)
        id01 = "cell_100m_" + r01.astype(str) + "_" + c01.astype(str)
        id10 = "cell_100m_" + r10.astype(str) + "_" + c10.astype(str)
        id11 = "cell_100m_" + r11.astype(str) + "_" + c11.astype(str)
        out["child_100m_cell_ids"] = id00 + ";" + id01 + ";" + id10 + ";" + id11
        # order matches ids: [top_left, top_right, bottom_left, bottom_right]
        breakdown = ["[" + ",".join(map(lambda x: f"{x:.6f}", row)) + "]"
                     for row in np.vstack([p00, p01, p10, p11]).T]
        out["child_100m_population_breakdown"] = breakdown
        return out

    # ---------- Builders for each resolution ----------
    def _build_tabs_100m(self, mapped, array, transform, raster_crs, nodata, low_thr, high_thr):
        if mapped.empty:
            return self._empty_grid_tab(100), self._empty_opp_tab()

        group_cols = ["row", "col", "opportunity_id"]
        if "opportunity_name" in mapped.columns:
            group_cols.append("opportunity_name")
        visit_counts = mapped.groupby(group_cols, dropna=False).size().rename("total_visits").reset_index()

        cells = visit_counts[["row", "col"]].drop_duplicates().reset_index(drop=True)
        attrs = self._cells_to_attributes_100m(cells, array, transform, raster_crs, nodata)
        grid = visit_counts.merge(attrs, on=["row", "col"], how="left")

        grid["visits_per_capita"] = grid["total_visits"] / grid["population"].replace(0, np.nan)
        cell_km2 = (100 / 1000.0) ** 2
        grid["population_density_per_km2"] = grid["population"] / cell_km2
        grid["resolution_m"] = 100

        def cat(d):
            if d < low_thr:
                return "Low"
            elif d < high_thr:
                return "Medium"
            else:
                return "High"
        grid["density_category"] = grid["population_density_per_km2"].apply(cat)

        ordered_cols = [
            "cell_id",
            "population",
            "center_latitude",
            "center_longitude",
            "opportunity_id",
            "opportunity_name",
            "total_visits",
            "visits_per_capita",
            "population_density_per_km2",
            "density_category",
            "resolution_m",
            "parent_200m_cell_id",
            "position_in_block",
        ]
        if "opportunity_name" not in grid.columns:
            grid["opportunity_name"] = np.nan
        grid = grid[ordered_cols].sort_values(["opportunity_id", "cell_id"]).reset_index(drop=True)

        opp = self._create_opportunity_summary(grid)
        return grid, opp

    def _build_tabs_200m_from_children(self, mapped200, array100, T100, raster_crs, nodata, low_thr, high_thr):
        if mapped200.empty:
            return self._empty_grid_tab(200, include_children_cols=True), self._empty_opp_tab()

        group_cols = ["row", "col", "opportunity_id"]
        if "opportunity_name" in mapped200.columns:
            group_cols.append("opportunity_name")
        visit_counts = mapped200.groupby(group_cols, dropna=False).size().rename("total_visits").reset_index()

        parents = visit_counts[["row", "col"]].drop_duplicates().reset_index(drop=True)
        attrs = self._cells_to_attributes_200m_from_children(parents, array100, T100, raster_crs, nodata)
        grid = visit_counts.merge(attrs, on=["row", "col"], how="left")

        grid["visits_per_capita"] = grid["total_visits"] / grid["population"].replace(0, np.nan)
        cell_km2 = (200 / 1000.0) ** 2
        grid["population_density_per_km2"] = grid["population"] / cell_km2
        grid["resolution_m"] = 200

        def cat(d):
            if d < low_thr:
                return "Low"
            elif d < high_thr:
                return "Medium"
            else:
                return "High"
        grid["density_category"] = grid["population_density_per_km2"].apply(cat)

        ordered_cols = [
            "cell_id",
            "population",
            "center_latitude",
            "center_longitude",
            "opportunity_id",
            "opportunity_name",
            "total_visits",
            "visits_per_capita",
            "population_density_per_km2",
            "density_category",
            "resolution_m",
            "child_100m_cell_ids",
            "child_100m_population_breakdown",
        ]
        if "opportunity_name" not in grid.columns:
            grid["opportunity_name"] = np.nan
        grid = grid[ordered_cols].sort_values(["opportunity_id", "cell_id"]).reset_index(drop=True)

        opp = self._create_opportunity_summary(grid)
        return grid, opp

    # ---------- Common helpers ----------
    def _empty_grid_tab(self, res_m: int, include_children_cols: bool = False):
        base = {
            "cell_id": [],
            "population": [],
            "center_latitude": [],
            "center_longitude": [],
            "opportunity_id": [],
            "opportunity_name": [],
            "total_visits": [],
            "visits_per_capita": [],
            "population_density_per_km2": [],
            "density_category": [],
            "resolution_m": [],
        }
        if res_m == 100:
            base["parent_200m_cell_id"] = []
            base["position_in_block"] = []
        if include_children_cols or res_m == 200:
            base["child_100m_cell_ids"] = []
            base["child_100m_population_breakdown"] = []
        return pd.DataFrame(base)

    def _empty_opp_tab(self):
        return pd.DataFrame(
            {
                "opportunity_id": [],
                "opportunity_name": [],
                "total_visits": [],
                "total_grids": [],
                "total_population": [],
                "visits_per_population": [],
                "low_density_visits": [],
                "low_density_grids": [],
                "low_density_pop": [],
                "low_density_visits_per_pop": [],
                "medium_density_visits": [],
                "medium_density_grids": [],
                "medium_density_pop": [],
                "medium_density_visits_per_pop": [],
                "high_density_visits": [],
                "high_density_grids": [],
                "high_density_pop": [],
                "high_density_visits_per_pop": [],
            }
        )

    def _create_opportunity_summary(self, grid_cells_df: pd.DataFrame) -> pd.DataFrame:
        if grid_cells_df.empty:
            return self._empty_opp_tab()

        def split_metrics(df):
            v = df["total_visits"].sum()
            g = len(df)
            p = df["population"].sum()
            r = (v / p) if p > 0 else 0.0
            return v, g, p, r

        rows = []
        for oid, grp in grid_cells_df.groupby("opportunity_id", dropna=False):
            oname = grp["opportunity_name"].dropna().iloc[0] if "opportunity_name" in grp.columns else oid

            total_v, total_g, total_p, total_r = split_metrics(grp)
            low_grp = grp[grp["density_category"] == "Low"]
            med_grp = grp[grp["density_category"] == "Medium"]
            high_grp = grp[grp["density_category"] == "High"]

            low_v, low_g, low_p, low_r = split_metrics(low_grp)
            med_v, med_g, med_p, med_r = split_metrics(med_grp)
            high_v, high_g, high_p, high_r = split_metrics(high_grp)

            rows.append(
                {
                    "opportunity_id": oid,
                    "opportunity_name": oname,
                    "total_visits": total_v,
                    "total_grids": total_g,
                    "total_population": total_p,
                    "visits_per_population": total_r,
                    "low_density_visits": low_v,
                    "low_density_grids": low_g,
                    "low_density_pop": low_p,
                    "low_density_visits_per_pop": low_r,
                    "medium_density_visits": med_v,
                    "medium_density_grids": med_g,
                    "medium_density_pop": med_p,
                    "medium_density_visits_per_pop": med_r,
                    "high_density_visits": high_v,
                    "high_density_grids": high_g,
                    "high_density_pop": high_p,
                    "high_density_visits_per_pop": high_r,
                }
            )
        return pd.DataFrame(rows).sort_values(["opportunity_id"]).reset_index(drop=True)

    def _make_compare_tab(self, opp100: pd.DataFrame, opp200: pd.DataFrame) -> pd.DataFrame:
        a = opp100[[
            "opportunity_id", "opportunity_name", "total_visits", "visits_per_population", "total_grids", "total_population"
        ]].copy()
        a = a.rename(columns={
            "total_visits": "total_visits_100m",
            "visits_per_population": "visits_per_population_100m",
            "total_grids": "total_grids_100m",
            "total_population": "total_population_100m",
        })

        b = opp200[[
            "opportunity_id", "opportunity_name", "total_visits", "visits_per_population", "total_grids", "total_population"
        ]].copy()
        b = b.rename(columns={
            "total_visits": "total_visits_200m",
            "visits_per_population": "visits_per_population_200m",
            "total_grids": "total_grids_200m",
            "total_population": "total_population_200m",
        })

        m = pd.merge(a, b, on=["opportunity_id"], how="outer", suffixes=("_100m", "_200m"))
        m["opportunity_name"] = m["opportunity_name_100m"].fillna(m.get("opportunity_name_200m"))
        m = m.drop(columns=[c for c in m.columns if c.startswith("opportunity_name_")])

        cols = [
            "opportunity_id", "opportunity_name",
            "total_visits_100m", "visits_per_population_100m",
            "total_visits_200m", "visits_per_population_200m",
            "total_grids_100m", "total_population_100m",
            "total_grids_200m", "total_population_200m",
        ]
        for c in cols:
            if c not in m.columns:
                m[c] = np.nan

        return m[cols].sort_values(["opportunity_id"]).reset_index(drop=True)
