
# grid3_100m_visit_report.py
"""
Grid3 100m Visit Report

A minimal, robust 100 m analysis:
- Maps visit points directly to raster pixel indices using the inverse affine transform
- Pulls population per visited cell straight from the raster array (no polygonization)
- Produces exactly two Excel tabs: Grid_Cells and Opportunity_Summary
- No buffering, no time series, no maps, no GIS exports

Assumptions:
- Visit coordinates are in WGS84 (lat/lon).
- The Grid3 population raster is at (or close to) 100 m resolution and aligned.
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


class Grid3100mVisitReport(BaseReport):
    """Report that analyzes visits against a 100 m Grid3 population raster (sparse, index-driven)."""

    # ---------------- UI ----------------
    @staticmethod
    def setup_parameters(parent_frame):
        """Minimal UI: Grid3 file picker and density thresholds."""
        ttk.Label(parent_frame, text="Grid3 TIF file:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        grid3_file_var = tk.StringVar()

        # Try to load last-used file
        config_file = os.path.join(os.path.expanduser("~"), ".grid3_100m_last_file.txt")
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
            command=lambda: Grid3100mVisitReport._browse_and_save_grid3_file_static(grid3_file_var, config_file),
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

        # Store
        parent_frame.grid3_file_var = grid3_file_var
        parent_frame.low_medium_var = low_medium_var
        parent_frame.medium_high_var = medium_high_var

    @staticmethod
    def _browse_and_save_grid3_file_static(var, config_file):
        filename = filedialog.askopenfilename(
            title="Select Grid3 TIF file", filetypes=[("TIF files", "*.tif"), ("All files", "*.*")]
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
        """Run the 100 m analysis and write a two-tab Excel file."""
        output_files = []

        # Params
        grid3_file = self.get_parameter_value("grid3_file", "")
        low_medium_threshold = float(self.get_parameter_value("low_medium", "10"))
        medium_high_threshold = float(self.get_parameter_value("medium_high", "50"))

        # Auto-detect if empty
        if not grid3_file or not os.path.exists(grid3_file):
            grid3_file = self._find_grid3_file()

        if not grid3_file:
            raise ValueError("No Grid3 TIF file specified or found.")

        self.log(f"[100m] Using Grid3 file: {os.path.basename(grid3_file)}")

        # Output dir
        today = datetime.now().strftime("%Y_%m_%d")
        output_dir = os.path.join(self.output_dir, f"grid3_100m_{today}")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare visits
        visits_df = self._prepare_visit_data()
        self.log(f"[100m] Visits loaded: {len(visits_df)}")

        # Load raster (as the canonical 100 m grid)
        array, transform, raster_crs, nodata = self._load_grid3_100m_raster(grid3_file)

        # Map visits -> (row, col) indices in raster
        mapped = self._map_visits_to_indices(visits_df, transform, raster_crs, grid3_file)

        if mapped.empty:
            self.log("[100m] No visits fell within raster bounds; writing empty workbook.")
            empty_grid, empty_opp = self._make_empty_outputs()
            excel_file = self._write_two_tab_excel(empty_grid, empty_opp, output_dir)
            output_files.append(excel_file)
            return output_files

        # Aggregate visits sparsely: (row, col, opportunity_id [, name]) -> total_visits
        group_cols = ["row", "col", "opportunity_id"]
        if "opportunity_name" in mapped.columns:
            group_cols.append("opportunity_name")
        visit_counts = mapped.groupby(group_cols, dropna=False).size().rename("total_visits").reset_index()
        self.log(f"[100m] Unique (cell,opp) combos: {len(visit_counts)}")

        # Attach population + center coords to each visited (row,col)
        unique_cells = visit_counts[["row", "col"]].drop_duplicates().reset_index(drop=True)
        cell_attrs = self._cells_to_attributes(unique_cells, array, transform, raster_crs, nodata)

        # Join attrs back to per-opportunity counts
        grid_cells = visit_counts.merge(cell_attrs, on=["row", "col"], how="left")

        # Derived metrics
        grid_cells["visits_per_capita"] = grid_cells["total_visits"] / grid_cells["population"].replace(0, np.nan)
        # For 100 m, each cell is 0.01 km²
        grid_cells["population_density_per_km2"] = grid_cells["population"] / (0.1 * 0.1)
        grid_cells["resolution_m"] = 100

        # Density categories
        def cat(d):
            if d < low_medium_threshold:
                return "Low"
            elif d < medium_high_threshold:
                return "Medium"
            else:
                return "High"
        grid_cells["density_category"] = grid_cells["population_density_per_km2"].apply(cat)

        # Reorder/rename for final tab
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
        ]
        # Ensure missing optional column exists
        if "opportunity_name" not in grid_cells.columns:
            grid_cells["opportunity_name"] = np.nan
        grid_cells = grid_cells[ordered_cols].sort_values(["opportunity_id", "cell_id"]).reset_index(drop=True)

        # Build opportunity summary (by 100 m only)
        opp_summary = self._create_opportunity_summary(grid_cells)

        # Write the two-tab Excel
        excel_file = self._write_two_tab_excel(grid_cells, opp_summary, output_dir)
        output_files.append(excel_file)

        self.log(f"[100m] Complete. Wrote: {os.path.basename(excel_file)}")
        return output_files

    # ---------------- Helpers ----------------
    def _find_grid3_file(self):
        """Try a few common paths to locate a Grid3 tif."""
        search_paths = [
            Path(self.output_dir).parent / "data" / "grid3",
            Path(self.output_dir).parent / "data" / "grid3" / "population",
            Path(self.output_dir) / ".." / "data" / "grid3",
        ]
        for p in search_paths:
            if p.exists():
                tifs = list(p.rglob("*.tif"))
                if tifs:
                    self.log(f"[100m] Auto-detected Grid3 file: {tifs[0]}")
                    return str(tifs[0])
        return None

    def _prepare_visit_data(self):
        """Normalize visit columns and keep only valid WGS84 coordinates."""
        data = self.df.copy()
        data.columns = data.columns.str.lower().str.strip()

        # Map common variants
        col_map = {
            "latitude": ["latitude", "lat", "y"],
            "longitude": ["longitude", "lon", "lng", "x"],
            "opportunity_id": ["opportunity_id"],
            "opportunity_name": ["opportunity_name", "opportunity"],
        }

        def pick(colnames):
            for c in colnames:
                if c in data.columns:
                    return c
            return None

        lat_col = pick(col_map["latitude"])
        lon_col = pick(col_map["longitude"])
        if lat_col is None or lon_col is None:
            raise ValueError("Visit data must include latitude/longitude.")

        # Standardize names
        rename = {lat_col: "latitude", lon_col: "longitude"}
        opp_id_col = pick(col_map["opportunity_id"])
        if opp_id_col:
            rename[opp_id_col] = "opportunity_id"
        opp_name_col = pick(col_map["opportunity_name"])
        if opp_name_col:
            rename[opp_name_col] = "opportunity_name"

        data = data.rename(columns=rename)

        # Clean and bound-check
        data = data.dropna(subset=["latitude", "longitude"]).copy()
        mask_valid = (
            (data["latitude"].between(-90, 90, inclusive="both"))
            & (data["longitude"].between(-180, 180, inclusive="both"))
        )
        data = data[mask_valid].reset_index(drop=True)

        # Default opportunity if missing
        if "opportunity_id" not in data.columns:
            data["opportunity_id"] = "ALL"
        if "opportunity_name" not in data.columns:
            data["opportunity_name"] = data["opportunity_id"]

        return data

    def _load_grid3_100m_raster(self, grid3_file):
        """Load raster & metadata. We treat it as the canonical 100 m grid."""
        with rasterio.open(grid3_file) as src:
            arr = src.read(1)  # population counts/density assumed additive
            transform = src.transform  # Affine
            raster_crs = src.crs
            nodata = src.nodata
            self.log(f"[100m] Raster: {src.width}x{src.height} | CRS={raster_crs} | nodata={nodata}")
        return arr, transform, raster_crs, nodata

    def _map_visits_to_indices(self, visits_df, transform: Affine, raster_crs, grid3_file_path: str):
        """Convert (lon,lat) -> raster CRS, then -> (col,row) via inverse affine; keep in-bounds only."""
        # Transform WGS84 -> raster CRS if needed
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            xs = visits_df["longitude"].to_numpy(np.float64)
            ys = visits_df["latitude"].to_numpy(np.float64)
        else:
            tfm = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            xs, ys = tfm.transform(visits_df["longitude"].to_numpy(np.float64), visits_df["latitude"].to_numpy(np.float64))

        invT = ~transform
        cols, rows = invT * (xs, ys)
        cols_i = np.floor(cols).astype(np.int64)
        rows_i = np.floor(rows).astype(np.int64)

        # Bounds filter
        with rasterio.open(grid3_file_path) as src:
            H, W = src.height, src.width
        in_bounds = (rows_i >= 0) & (rows_i < H) & (cols_i >= 0) & (cols_i < W)

        mapped = visits_df.loc[in_bounds, ["opportunity_id", "opportunity_name"]].copy()
        mapped["row"] = rows_i[in_bounds]
        mapped["col"] = cols_i[in_bounds]

        dropped = (~in_bounds).sum()
        if dropped:
            self.log(f"[100m] Dropped {dropped} visits outside raster bounds")
        return mapped

    def _cells_to_attributes(self, unique_cells_df, array, transform: Affine, raster_crs, nodata):
        """For each (row,col), attach population and WGS84 center coordinates."""
        rows = unique_cells_df["row"].to_numpy(np.int64)
        cols = unique_cells_df["col"].to_numpy(np.int64)

        # Population (handle nodata)
        pop = array[rows, cols].astype(float)
        if nodata is not None:
            pop = np.where(pop == nodata, np.nan, pop)
        pop = np.nan_to_num(pop, nan=0.0)

        # Centers in raster CRS
        # Pixel center = (col + 0.5, row + 0.5)
        cx = transform * (cols + 0.5, rows + 0.5)
        cx_x = np.asarray(cx[0], dtype=np.float64)
        cx_y = np.asarray(cx[1], dtype=np.float64)

        # Raster CRS -> WGS84 if needed
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
        return out

    def _create_opportunity_summary(self, grid_cells_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize by opportunity with density splits."""
        if grid_cells_df.empty:
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

        # Helper to compute split metrics
        def split_metrics(df):
            v = df["total_visits"].sum()
            g = len(df)
            p = df["population"].sum()
            r = (v / p) if p > 0 else 0.0
            return v, g, p, r

        rows = []
        for opp_id, grp in grid_cells_df.groupby("opportunity_id", dropna=False):
            opp_name = grp["opportunity_name"].dropna().iloc[0] if "opportunity_name" in grp.columns else opp_id

            total_v, total_g, total_p, total_r = split_metrics(grp)

            low_grp = grp[grp["density_category"] == "Low"]
            med_grp = grp[grp["density_category"] == "Medium"]
            high_grp = grp[grp["density_category"] == "High"]

            low_v, low_g, low_p, low_r = split_metrics(low_grp)
            med_v, med_g, med_p, med_r = split_metrics(med_grp)
            high_v, high_g, high_p, high_r = split_metrics(high_grp)

            rows.append(
                {
                    "opportunity_id": opp_id,
                    "opportunity_name": opp_name,
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

    def _write_two_tab_excel(self, grid_cells_df: pd.DataFrame, opp_summary_df: pd.DataFrame, output_dir: str) -> str:
        """Write Grid_Cells and Opportunity_Summary to a single workbook."""
        excel_file = os.path.join(output_dir, "grid3_100m_visit_report.xlsx")
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Grid_Cells tab
            cols = [
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
            ]
            # Ensure empty but well-formed
            if grid_cells_df.empty:
                grid_cells_df = pd.DataFrame({c: [] for c in cols})
            grid_cells_df.to_excel(writer, sheet_name="Grid_Cells", index=False)

            # Opportunity_Summary tab
            if opp_summary_df.empty:
                opp_summary_df = pd.DataFrame(
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
            opp_summary_df.to_excel(writer, sheet_name="Opportunity_Summary", index=False)

        return excel_file

    def _make_empty_outputs(self):
        empty_grid = pd.DataFrame(
            {
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
        )
        empty_opp = pd.DataFrame(
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
        return empty_grid, empty_opp
