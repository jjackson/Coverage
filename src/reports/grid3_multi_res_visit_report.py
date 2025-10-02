# v1.3 – Grid3 multi-resolution (100m, 200m, 300m) – index-based child sums + no density calcs
# grid3_multi_res_visit_report.py

"""
Grid3 Multi-Resolution Visit Report (100m + 200m + 300m)

Index-based approach (no polygonization, no spatial join):
- Map visits once to 100m (row, col).
- Build 100m grid from the raster array.
- 200m parents = (row//2,col//2), population = sum of 2×2 children (NaN/NoData -> 0).
- 300m parents = (row//3,col//3), population = sum of 3×3 children (NaN/NoData -> 0).
- Adds hierarchy columns:
  * 100m: parent_200m_cell_id, position_in_block
  * 200m/300m: child_100m_cell_ids, child_100m_population_breakdown
- Writes 7 tabs including Opportunity_Compare (100/200/300 side-by-side).
- Removed all density threshold inputs and calculations.
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

__VERSION__ = 'Grid3 Multi v1.3 – index-based (child sums), NaNs→0, no density calcs'


class Grid3MultiResVisitReport(BaseReport):
    # ---------------- UI ----------------
    @staticmethod
    def setup_parameters(parent_frame):
        ttk.Label(parent_frame, text="Grid3 TIF file:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        grid3_file_var = tk.StringVar()

        # Try to load last-used file
        config_file = os.path.join(os.path.expanduser("~"), ".grid3_multi_last_file.txt")
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
            command=lambda: Grid3MultiResVisitReport._browse_and_save_grid3_file_static(grid3_file_var, config_file),
        ).grid(row=0, column=1, padx=(6, 0))
        file_frame.columnconfigure(0, weight=1)

        parent_frame.grid3_file_var = grid3_file_var

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

        # ==== DEBUG BANNER ======================================================
        self.log('=================================================================')
        self.log(f'[{__VERSION__}] Starting multi-resolution run (100m/200m/300m, index-based)')
        self.log('Expect: visits match across resolutions; 200m pop >= 100m pop; NaNs treated as 0')
        self.log('=================================================================')
        # =======================================================================

        # Params
        grid3_file = self.get_parameter_value("grid3_file", "")

        if not grid3_file or not os.path.exists(grid3_file):
            grid3_file = self._find_grid3_file()
        if not grid3_file:
            raise ValueError("No Grid3 TIF file specified or found.")

        self.log(f"[multi] Using Grid3 file: {os.path.basename(grid3_file)}")

        # Output dir
        today = datetime.now().strftime("%Y_%m_%d")
        output_dir = os.path.join(self.output_dir, f"grid3_multi_{today}")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare visits once
        visits_df = self._prepare_visit_data()
        self.log(f"[multi] Visits loaded: {len(visits_df)}")

        # Load the 100m raster
        arr100, T100, raster_crs, nodata = self._load_grid3_raster(grid3_file)
        self.log(f"[multi] 100m raster shape: {arr100.shape}")

        # ---- 100m path ----
        mapped100 = self._map_visits_to_indices(visits_df, T100, raster_crs, grid_shape=arr100.shape, grid3_file_path=grid3_file)
        grid100, opp100 = self._build_tabs_100m(mapped100, arr100, T100, raster_crs, nodata)
        self.log(f"[multi] 100m: grid_rows={len(grid100)}, opp_rows={len(opp100)}")
        try:
            self.log(f"[multi] 100m totals – visits={int(opp100['total_visits'].sum())}, pop={opp100['total_population'].sum():.2f}")
        except Exception:
            pass

        # ---- 200m path (from 100m children) ----
        mapped200 = mapped100.copy()
        mapped200["row"] = (mapped200["row"] // 2).astype("int64")
        mapped200["col"] = (mapped200["col"] // 2).astype("int64")
        grid200, opp200 = self._build_tabs_parent_from_children(
            mapped200, arr100, T100, raster_crs, nodata, scale=2, res_m=200
        )
        self.log(f"[multi] 200m: grid_rows={len(grid200)}, opp_rows={len(opp200)}")
        try:
            self.log(f"[multi] 200m totals – visits={int(opp200['total_visits'].sum())}, pop={opp200['total_population'].sum():.2f}")
        except Exception:
            pass

        # ---- 300m path (from 100m children) ----
        mapped300 = mapped100.copy()
        mapped300["row"] = (mapped300["row"] // 3).astype("int64")
        mapped300["col"] = (mapped300["col"] // 3).astype("int64")
        grid300, opp300 = self._build_tabs_parent_from_children(
            mapped300, arr100, T100, raster_crs, nodata, scale=3, res_m=300
        )
        self.log(f"[multi] 300m: grid_rows={len(grid300)}, opp_rows={len(opp300)}")
        try:
            self.log(f"[multi] 300m totals – visits={int(opp300['total_visits'].sum())}, pop={opp300['total_population'].sum():.2f}")
        except Exception:
            pass

        # ---- Opportunity_Compare (100m, 200m, 300m) ----
        compare = self._make_compare_tab_three(opp100, opp200, opp300)

        # Write one workbook with seven tabs
        prefix = Path(grid3_file).stem[:3].lower()
        n_opps = visits_df["opportunity_id"].nunique()
        excel_file = os.path.join(output_dir, f"grid3_multi_{prefix}_{n_opps}opps.xlsx")

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            grid100.to_excel(writer, sheet_name="Grid_Cells_100m", index=False)
            opp100.to_excel(writer, sheet_name="Opportunity_Summary_100m", index=False)
            grid200.to_excel(writer, sheet_name="Grid_Cells_200m", index=False)
            opp200.to_excel(writer, sheet_name="Opportunity_Summary_200m", index=False)
            grid300.to_excel(writer, sheet_name="Grid_Cells_300m", index=False)
            opp300.to_excel(writer, sheet_name="Opportunity_Summary_300m", index=False)
            compare.to_excel(writer, sheet_name="Opportunity_Compare", index=False)

        # Final sanity
        self._log_sanity(opp100, opp200, opp300)

        self.log(f"[multi] Complete. Wrote: {os.path.basename(excel_file)}")
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
                    self.log(f"[multi] Auto-detected Grid3 file: {tifs[0]}")
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
            self.log(f"[multi] Raster meta: {src.width}x{src.height} | CRS={crs} | nodata={nodata}")
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
            self.log(f"[multi] Dropped {dropped} visits outside raster bounds (res-specific)")
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

    def _cells_to_attributes_parent_from_children(self, unique_parents_df, array100, T100: Affine, raster_crs, nodata, scale: int):
        """
        Compute parent (200m or 300m) attributes by summing NxN (scale x scale) 100m children.
        - Replace explicit nodata and NaN with 0.0 in children before summing.
        - Produce child ID lists and population breakdowns in row-major order.
        """
        prow = unique_parents_df["row"].to_numpy(np.int64)
        pcol = unique_parents_df["col"].to_numpy(np.int64)

        r0 = (scale * prow).astype(np.int64)
        c0 = (scale * pcol).astype(np.int64)

        H, W = array100.shape

        def safe_pick(r, c):
            inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            rr = np.where(inb, r, 0)
            cc = np.where(inb, c, 0)
            vals = array100[rr, cc].astype(float)
            if nodata is not None:
                vals = np.where(vals == nodata, 0.0, vals)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = np.where(inb, vals, 0.0)
            return vals

        # Collect children in row-major order
        child_vals = []
        child_ids = []
        for dr in range(scale):
            for dc in range(scale):
                r = r0 + dr
                c = c0 + dc
                vals = safe_pick(r, c)
                child_vals.append(vals)
                cid = "cell_100m_" + r.astype(str) + "_" + c.astype(str)
                child_ids.append(cid)

        # Sum across children
        pop = np.zeros_like(child_vals[0], dtype=float)
        for v in child_vals:
            pop = pop + v

        # centers from Tparent = T100 * scale(scale,scale)
        Tparent = T100 * Affine.scale(scale, scale)
        cx = Tparent * (pcol + 0.5, prow + 0.5)
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
        out["cell_id"] = f"cell_{scale*100}m_" + out["row"].astype(str) + "_" + out["col"].astype(str)

        # Build semicolon-separated ids and breakdown strings
        # child_vals is a list of arrays; stack as columns then serialize row-wise
        vals_matrix = np.vstack(child_vals).T  # shape: (nrows, scale*scale)
        ids_matrix = np.vstack([np.array(ids, dtype=object) for ids in child_ids]).T  # same shape

        out["child_100m_cell_ids"] = [";".join(row.astype(str)) for row in ids_matrix]
        out["child_100m_population_breakdown"] = [
            "[" + ",".join(f"{x:.6f}" for x in row) + "]" for row in vals_matrix
        ]

        return out

    # ---------- Builders for each resolution ----------
    def _build_tabs_100m(self, mapped, array, transform, raster_crs, nodata):
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
        grid["resolution_m"] = 100

        ordered_cols = [
            "cell_id",
            "population",
            "center_latitude",
            "center_longitude",
            "opportunity_id",
            "opportunity_name",
            "total_visits",
            "visits_per_capita",
            "resolution_m",
            "parent_200m_cell_id",
            "position_in_block",
        ]
        if "opportunity_name" not in grid.columns:
            grid["opportunity_name"] = np.nan
        grid = grid[ordered_cols].sort_values(["opportunity_id", "cell_id"]).reset_index(drop=True)

        opp = self._create_opportunity_summary(grid)
        return grid, opp

    def _build_tabs_parent_from_children(self, mapped_parent, array100, T100, raster_crs, nodata,
                                         scale, res_m):
        if mapped_parent.empty:
            return self._empty_grid_tab(res_m, include_children_cols=True), self._empty_opp_tab()

        group_cols = ["row", "col", "opportunity_id"]
        if "opportunity_name" in mapped_parent.columns:
            group_cols.append("opportunity_name")
        visit_counts = mapped_parent.groupby(group_cols, dropna=False).size().rename("total_visits").reset_index()

        parents = visit_counts[["row", "col"]].drop_duplicates().reset_index(drop=True)
        attrs = self._cells_to_attributes_parent_from_children(parents, array100, T100, raster_crs, nodata, scale=scale)
        grid = visit_counts.merge(attrs, on=["row", "col"], how="left")

        grid["visits_per_capita"] = grid["total_visits"] / grid["population"].replace(0, np.nan)
        grid["resolution_m"] = res_m

        ordered_cols = [
            "cell_id",
            "population",
            "center_latitude",
            "center_longitude",
            "opportunity_id",
            "opportunity_name",
            "total_visits",
            "visits_per_capita",
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
            "resolution_m": [],
        }
        if res_m == 100:
            base["parent_200m_cell_id"] = []
            base["position_in_block"] = []
        if include_children_cols or res_m in (200, 300):
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
            }
        )

    def _create_opportunity_summary(self, grid_cells_df: pd.DataFrame) -> pd.DataFrame:
        if grid_cells_df.empty:
            return self._empty_opp_tab()

        rows = []
        for oid, grp in grid_cells_df.groupby("opportunity_id", dropna=False):
            oname = grp["opportunity_name"].dropna().iloc[0] if "opportunity_name" in grp.columns else oid

            total_visits = grp["total_visits"].sum()
            total_grids = len(grp)
            total_population = grp["population"].sum()
            visits_per_population = (total_visits / total_population) if total_population > 0 else 0.0

            rows.append(
                {
                    "opportunity_id": oid,
                    "opportunity_name": oname,
                    "total_visits": total_visits,
                    "total_grids": total_grids,
                    "total_population": total_population,
                    "visits_per_population": visits_per_population,
                }
            )
        return pd.DataFrame(rows).sort_values(["opportunity_id"]).reset_index(drop=True)

    def _make_compare_tab_three(self, opp100: pd.DataFrame, opp200: pd.DataFrame, opp300: pd.DataFrame) -> pd.DataFrame:
        a = opp100[[
            "opportunity_id", "opportunity_name", "total_visits", "visits_per_population", "total_grids", "total_population"
        ]].copy().rename(columns={
            "total_visits": "total_visits_100m",
            "visits_per_population": "visits_per_population_100m",
            "total_grids": "total_grids_100m",
            "total_population": "total_population_100m",
        })

        b = opp200[[
            "opportunity_id", "opportunity_name", "total_visits", "visits_per_population", "total_grids", "total_population"
        ]].copy().rename(columns={
            "total_visits": "total_visits_200m",
            "visits_per_population": "visits_per_population_200m",
            "total_grids": "total_grids_200m",
            "total_population": "total_population_200m",
        })

        c = opp300[[
            "opportunity_id", "opportunity_name", "total_visits", "visits_per_population", "total_grids", "total_population"
        ]].copy().rename(columns={
            "total_visits": "total_visits_300m",
            "visits_per_population": "visits_per_population_300m",
            "total_grids": "total_grids_300m",
            "total_population": "total_population_300m",
        })

        m = pd.merge(a, b, on=["opportunity_id"], how="outer", suffixes=("_100m", "_200m"))
        m = pd.merge(m, c, on=["opportunity_id"], how="outer")
        # Resolve name
        m["opportunity_name"] = m["opportunity_name_100m"].fillna(m.get("opportunity_name_200m")).fillna(m.get("opportunity_name"))
        m = m.drop(columns=[c for c in m.columns if c.startswith("opportunity_name_")])

        cols = [
            "opportunity_id", "opportunity_name",
            "total_visits_100m", "visits_per_population_100m",
            "total_visits_200m", "visits_per_population_200m",
            "total_visits_300m", "visits_per_population_300m",
            "total_grids_100m", "total_population_100m",
            "total_grids_200m", "total_population_200m",
            "total_grids_300m", "total_population_300m",
        ]
        for col in cols:
            if col not in m.columns:
                m[col] = np.nan
        return m[cols].sort_values(["opportunity_id"]).reset_index(drop=True)

    # ---------------- Sanity logger ----------------
    def _log_sanity(self, opp100, opp200, opp300):
        try:
            import numpy as _np
            def s(df):
                if df is None or len(df) == 0: 
                    return (0.0, 0.0, 0.0, 0)
                return (
                    float(_np.nan_to_num(df["total_population"]).sum() if "total_population" in df else 0.0),
                    float(_np.nan_to_num(df["total_visits"]).sum() if "total_visits" in df else 0.0),
                    float(len(df) if df is not None else 0),
                    int(df["opportunity_id"].nunique()) if "opportunity_id" in df else 0,
                )
            p100, v100, g100, u100 = s(opp100)
            p200, v200, g200, u200 = s(opp200)
            p300, v300, g300, u300 = s(opp300)
            self.log(f"[multi][sanity] totals – pop: 100m={p100:.2f}, 200m={p200:.2f}, 300m={p300:.2f}")
            self.log(f"[multi][sanity] totals – visits: 100m={int(v100)}, 200m={int(v200)}, 300m={int(v300)}")
            if abs(v100 - v200) > 1e-9 or abs(v100 - v300) > 1e-9:
                self.log("[multi][warn] Visit totals differ across resolutions (expected to match).")
            if p200 + 1e-6 < p100:
                self.log("[multi][warn] Total population at 200m < 100m – investigate nodata/NaN handling or mapping.")
            if p300 + 1e-6 < p200:
                self.log("[multi][info] Total population at 300m < 200m – can happen depending on which parents were touched.")
        except Exception as e:
            self.log(f"[multi][sanity][error] {e}")