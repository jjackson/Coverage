# grid3_ward_analysis.py - PART 1 OF 3  

"""
Grid3 Ward Analysis Report - Complete Working Version with Building Integration

Performs comprehensive Grid3 analysis for all cells within wards that contain visits.
Uses index-based aggregation approach to ensure consistency across resolutions.
Includes building footprint data integration and analysis.

Key improvements:
- Index-based aggregation: 200m/300m cells are derived from 100m base grid
- Consistent cell counts and population totals across resolutions
- Each 100m cell assigned to exactly one ward (centroid-based)
- Building data loading and mapping to grid cells
- Population vs building discrepancy analysis
- Generates enriched visit file with grid cell assignments
- Formatted numbers with commas and one decimal place
- Excel output for analysis files (CSV for enriched visits)
- Interactive ward maps with visit markers and building footprints

Input: 
- Visit data with lat/lon coordinates
- Ward boundary files from Stage 1 output
- Grid3 raster file (100m resolution)
- Building CSV files (optional)

Output:
- Grid cell analysis for all cells in affected wards
- Ward-level summaries with accurate totals
- Building counts and metrics per cell/ward/cluster
- Population vs building comparison analysis
- Enriched visit file with grid/ward assignments (CSV)
- Multi-resolution comparison tables (Excel)
- Interactive HTML maps per ward with buildings
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Point
from .Grid3WardMapper import Grid3WardMapper
from .Grid3BuildingLoader import load_buildings_for_analysis
import warnings

from .base_report import BaseReport

__VERSION__ = 'Grid3 Ward Analysis v3.2 - With Building Integration'


class Grid3WardAnalysis(BaseReport):
    # ---------------- UI ----------------
    @staticmethod
    def setup_parameters(parent_frame):
        """Setup UI parameters for Grid3 ward analysis"""
        
        # Stage 1 output folder
        ttk.Label(parent_frame, text="Stage 1 Output Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        stage1_folder_var = tk.StringVar()
        
        # Try to find recent ward extractor output
        config_file = os.path.join(os.path.expanduser("~"), ".grid3_ward_config.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    last_folder = f.read().strip()
                    if os.path.exists(last_folder):
                        stage1_folder_var.set(last_folder)
            except Exception:
                pass

        folder_frame = ttk.Frame(parent_frame)
        folder_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Entry(folder_frame, textvariable=stage1_folder_var, width=44).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(
            folder_frame,
            text="Browse...",
            command=lambda: Grid3WardAnalysis._browse_and_save_folder(stage1_folder_var, config_file),
        ).grid(row=0, column=1, padx=(6, 0))
        folder_frame.columnconfigure(0, weight=1)

        # Grid3 TIF file
        ttk.Label(parent_frame, text="Grid3 TIF file:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        grid3_file_var = tk.StringVar()

        # Try to load last-used Grid3 file
        grid3_config = os.path.join(os.path.expanduser("~"), ".grid3_ward_tif_config.txt")
        if os.path.exists(grid3_config):
            try:
                with open(grid3_config, "r") as f:
                    last_tif = f.read().strip()
                    if os.path.exists(last_tif):
                        grid3_file_var.set(last_tif)
            except Exception:
                pass

        tif_frame = ttk.Frame(parent_frame)
        tif_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Entry(tif_frame, textvariable=grid3_file_var, width=44).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(
            tif_frame,
            text="Browse...",
            command=lambda: Grid3WardAnalysis._browse_and_save_tif_file(grid3_file_var, grid3_config),
        ).grid(row=0, column=1, padx=(6, 0))
        tif_frame.columnconfigure(0, weight=1)

        # Processing options
        ttk.Label(parent_frame, text="Include partial intersections:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        partial_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, variable=partial_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(parent_frame, text="(Include cells that partially overlap ward boundaries)").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

        # Buildings directory (optional)
        ttk.Label(parent_frame, text="Buildings Directory (optional):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        buildings_dir_var = tk.StringVar()
        
        buildings_frame = ttk.Frame(parent_frame)
        buildings_frame.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Entry(buildings_frame, textvariable=buildings_dir_var, width=44).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(
            buildings_frame,
            text="Browse...",
            command=lambda: Grid3WardAnalysis._browse_buildings_folder(buildings_dir_var),
        ).grid(row=0, column=1, padx=(6, 0))
        buildings_frame.columnconfigure(0, weight=1)

        # Store variables for access in generate()
        parent_frame.stage1_folder_var = stage1_folder_var
        parent_frame.grid3_file_var = grid3_file_var
        parent_frame.partial_var = partial_var
        parent_frame.buildings_dir_var = buildings_dir_var

    @staticmethod
    def _browse_buildings_folder(var):
        """Browse for buildings directory"""
        directory = filedialog.askdirectory(
            title="Select Buildings Directory (optional)"
        )
        if directory:
            var.set(directory)

    @staticmethod
    def _browse_and_save_folder(var, config_file):
        """Browse for Stage 1 output folder"""
        directory = filedialog.askdirectory(
            title="Select Stage 1 Output Folder (containing ward boundary files)"
        )
        if directory:
            var.set(directory)
            try:
                with open(config_file, "w") as f:
                    f.write(directory)
            except Exception:
                pass

    @staticmethod
    def _browse_and_save_tif_file(var, config_file):
        """Browse for Grid3 TIF file"""
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

    # ---------------- Number Formatting ----------------
    def _format_number(self, value, decimals=1, is_percentage=False):
        """Format numbers with commas and specified decimal places"""
        if pd.isna(value) or value is None:
            return ""
        
        try:
            if is_percentage:
                formatted = f"{float(value):.{decimals}f}%"
            else:
                formatted = f"{float(value):,.{decimals}f}"
            return formatted
        except (ValueError, TypeError):
            return str(value)
    
    def _format_dataframe_numbers(self, df):
        """Apply number formatting to numeric columns in a dataframe"""
        df_formatted = df.copy()
        
        percentage_cols = [col for col in df.columns if 'pct' in col.lower() or 'coverage' in col.lower()]
        
        integer_cols = ['total_cells', 'visited_cells', 'unvisited_cells', 'total_visits', 
                       'visits_in_cell', 'unique_opportunities', 'total_wards', 'num_buildings',
                       'total_buildings', 'buildings_in_visited_cells', 'buildings_in_unvisited_cells']
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if col in percentage_cols:
                    df_formatted[col] = df[col].apply(lambda x: self._format_number(x, 1, True))
                elif col in integer_cols:
                    df_formatted[col] = df[col].apply(lambda x: self._format_number(x, 0))
                elif 'population' in col.lower() or 'visits_per' in col.lower() or 'building' in col.lower():
                    df_formatted[col] = df[col].apply(lambda x: self._format_number(x, 1))
                else:
                    df_formatted[col] = df[col].apply(lambda x: self._format_number(x, 1))
        
        return df_formatted

    # ---------------- Main entry ----------------
    def generate(self):
        """Main entry point for Grid3 ward analysis"""
        output_files = []

        self.log('=================================================================')
        self.log(f'[{__VERSION__}] Starting Grid3 ward analysis')
        self.log('Index-based multi-resolution approach - ensures consistency')
        self.log('Building integration enabled - population vs building analysis')
        self.log('Formatted output with Excel files and interactive maps')
        self.log('=================================================================')

        # Get parameters
        stage1_folder = self.get_parameter_value("stage1_folder", "")
        grid3_file = self.get_parameter_value("grid3_file", "")
        include_partial = self.get_parameter_value("partial", True)
        buildings_dir = self.get_parameter_value("buildings_dir", "")

        if not stage1_folder or not os.path.exists(stage1_folder):
            raise ValueError("Please specify a valid Stage 1 output folder")
        
        if not grid3_file or not os.path.exists(grid3_file):
            grid3_file = self._find_grid3_file()
            if not grid3_file:
                raise ValueError("Please specify a valid Grid3 TIF file")

        self.log(f"[ward_analysis] Stage 1 folder: {stage1_folder}")
        self.log(f"[ward_analysis] Grid3 file: {os.path.basename(grid3_file)}")
        self.log(f"[ward_analysis] Include partial intersections: {include_partial}")
        self.log(f"[ward_analysis] Buildings directory: {buildings_dir if buildings_dir else 'Not specified'}")

        # Create output directory
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load inputs
        visits_df = self._prepare_visit_data()
        ward_boundaries_gdf = self._load_stage1_ward_boundaries(stage1_folder)
        
        self.log(f"[ward_analysis] Loaded {len(visits_df):,} visits")
        self.log(f"[ward_analysis] Loaded {len(ward_boundaries_gdf):,} ward boundaries")

        # Load building data if available
        buildings_df = None
        if buildings_dir and os.path.exists(buildings_dir):
            buildings_df = load_buildings_for_analysis(buildings_dir, ward_boundaries_gdf, self.log)
            if buildings_df is not None:
                self.log(f"[ward_analysis] Loaded {len(buildings_df):,} buildings")
        else:
            self.log(f"[ward_analysis] No buildings directory specified - skipping building analysis")

        # Load Grid3 raster
        arr100, T100, raster_crs, nodata = self._load_grid3_raster(grid3_file)
        self.log(f"[ward_analysis] Grid3 raster: {arr100.shape}, CRS: {raster_crs}")

        # Generate data tag for this run
        data_tag = self._generate_data_tag(visits_df, ward_boundaries_gdf)
        self.log(f"[ward_analysis] Dataset tag: {data_tag}")

        # INDEX-BASED PROCESSING: Create base 100m cell inventory
        self.log("[ward_analysis] Creating base 100m cell inventory...")
        base_cells_100m = self._create_100m_cell_inventory(
            ward_boundaries_gdf, arr100, T100, raster_crs, nodata, include_partial
        )
        
        # Generate higher resolution cells by index aggregation
        all_cells = self._create_multi_resolution_cells(base_cells_100m, arr100, T100, raster_crs, nodata)
        
        # Map visits to all resolutions
        enriched_visits = self._map_visits_to_all_resolutions(visits_df, all_cells, T100, raster_crs)
        
        # Map buildings to all resolutions if available
        if buildings_df is not None:
            self._map_buildings_to_all_resolutions(buildings_df, all_cells, T100, raster_crs)
        
        # Generate all outputs
        output_files.extend(self._write_all_outputs(
            all_cells, enriched_visits, ward_boundaries_gdf, data_tag, output_dir, buildings_df
        ))

        self.log(f"[ward_analysis] Complete! Generated {len(output_files)} files")
        return output_files

# grid3_ward_analysis.py - PART 2 OF 3
# This continues from Part 1 - paste this immediately after Part 1

    # ---------------- Helper methods ----------------
    
    def _prepare_visit_data(self):
        """Prepare and validate visit data"""
        data = self.df.copy()
        data.columns = data.columns.str.lower().str.strip()

        # Find coordinate columns
        col_map = {
            "latitude": ["latitude", "lat", "y"],
            "longitude": ["longitude", "lon", "lng", "x"],
            "opportunity_id": ["opportunity_id"],
            "opportunity_name": ["opportunity_name", "opportunity"],
            "visit_id": ["visit_id", "visitid", "id"],
            "flw_id": ["flw_id", "user_id", "username", "enumerator_id", "worker_id"],
        }

        def pick_column(candidates):
            for c in candidates:
                if c in data.columns:
                    return c
            return None

        lat_col = pick_column(col_map["latitude"])
        lon_col = pick_column(col_map["longitude"])
        
        if lat_col is None or lon_col is None:
            raise ValueError("Visit data must include latitude/longitude columns")

        # Rename columns
        rename_map = {lat_col: "latitude", lon_col: "longitude"}
        
        for key, candidates in col_map.items():
            if key not in ["latitude", "longitude"]:
                col = pick_column(candidates)
                if col and col in data.columns:
                    rename_map[col] = key
                    
        data = data.rename(columns=rename_map)

        # Clean and validate coordinates
        data = data.dropna(subset=["latitude", "longitude"]).copy()
        valid_coords = (
            data["latitude"].between(-90, 90, inclusive="both") & 
            data["longitude"].between(-180, 180, inclusive="both")
        )
        data = data[valid_coords].reset_index(drop=True)

        # Add default columns if missing
        if "opportunity_id" not in data.columns:
            data["opportunity_id"] = "ALL"
        if "opportunity_name" not in data.columns:
            data["opportunity_name"] = data["opportunity_id"]
        if "visit_id" not in data.columns:
            data["visit_id"] = "visit_" + data.index.astype(str)
        if "flw_id" not in data.columns:
            coords_str = data["latitude"].astype(str) + "_" + data["longitude"].astype(str)
            unique_coords = coords_str.drop_duplicates()
            coord_to_flw = {coord: f"flw_{i}" for i, coord in enumerate(unique_coords)}
            data["flw_id"] = coords_str.map(coord_to_flw)

        return data

    def _load_stage1_ward_boundaries(self, stage1_folder):
        """Auto-detect and load ward boundary files from Stage 1 output"""
        folder_path = Path(stage1_folder)
        
        shapefiles = list(folder_path.glob("affected_wards_*.shp"))
        geojsons = list(folder_path.glob("affected_wards_*.geojson"))
        
        if shapefiles:
            ward_file = shapefiles[0]
            self.log(f"[ward_analysis] Loading ward boundaries from: {ward_file.name}")
        elif geojsons:
            ward_file = geojsons[0]
            self.log(f"[ward_analysis] Loading ward boundaries from: {ward_file.name}")
        else:
            raise ValueError(f"No ward boundary files found in {stage1_folder}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ward_gdf = gpd.read_file(ward_file)
        
        if ward_gdf.crs != "EPSG:4326":
            ward_gdf = ward_gdf.to_crs("EPSG:4326")
        
        return ward_gdf

    def _load_grid3_raster(self, grid3_file):
        """Load Grid3 raster data"""
        with rasterio.open(grid3_file) as src:
            arr = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        return arr, transform, crs, nodata

    def _find_grid3_file(self):
        """Auto-find Grid3 file in common locations"""
        search_paths = [
            Path(self.output_dir).parent / "data" / "grid3",
            Path(self.output_dir).parent / "data" / "grid3" / "population",
            Path(self.output_dir) / ".." / "data" / "grid3",
        ]
        for p in search_paths:
            if p.exists():
                tifs = list(p.rglob("*.tif"))
                if tifs:
                    self.log(f"[ward_analysis] Auto-detected Grid3 file: {tifs[0]}")
                    return str(tifs[0])
        return None

    def _generate_data_tag(self, visits_df, ward_gdf):
        """Generate unique tag for this dataset"""
        unique_flws = visits_df['flw_id'].nunique()
        num_visits = len(visits_df)
        num_opps = visits_df['opportunity_id'].nunique()
        num_wards = len(ward_gdf)
        
        tag = f"{unique_flws}flw_{num_visits}v_{num_opps}opp_{num_wards}w"
        return tag

    def _create_100m_cell_inventory(self, ward_gdf, arr100, T100, raster_crs, nodata, include_partial):
        """Create base 100m cell inventory with ward assignments"""
        self.log("[ward_analysis] Building base 100m cell inventory...")
        
        all_100m_cells = []
        
        for _, ward_row in ward_gdf.iterrows():
            ward_id = ward_row['ward_id']
            ward_name = ward_row.get('ward_name', 'Unknown')
            state_name = ward_row.get('state_name', 'Unknown')
            ward_geom = ward_row['geometry']
            
            ward_cells = self._find_cells_for_ward(
                ward_geom, T100, arr100.shape[0], arr100.shape[1], raster_crs, include_partial
            )
            
            for cell in ward_cells:
                cell.update({
                    'ward_id': ward_id,
                    'ward_name': ward_name,
                    'state_name': state_name
                })
            
            all_100m_cells.extend(ward_cells)
        
        if all_100m_cells:
            cells_df = pd.DataFrame(all_100m_cells)
            cells_df = cells_df.drop_duplicates(subset=['row', 'col'], keep='first')
            cells_df = self._add_100m_cell_attributes(cells_df, arr100, T100, raster_crs, nodata)
            self.log(f"[ward_analysis] Base 100m: {len(cells_df):,} unique cells")
        else:
            cells_df = pd.DataFrame()
        
        return cells_df

    def _create_multi_resolution_cells(self, base_cells_100m, arr100, T100, raster_crs, nodata):
        """Create 200m, 300m, 500m, 700m, 1000m cells by aggregating 100m base cells"""
        all_cells = {100: base_cells_100m}
        
        if len(base_cells_100m) == 0:
            all_cells[300] = pd.DataFrame()
            return all_cells
        
        self.log("[ward_analysis] Generating 300m cells from 100m base...")
        cells_300m = self._aggregate_cells_to_resolution(base_cells_100m, arr100, T100, raster_crs, nodata, scale=3)
        all_cells[300] = cells_300m
        self.log(f"[ward_analysis] Generated {len(cells_300m):,} unique 300m cells")

        return all_cells

    def _aggregate_cells_to_resolution(self, base_cells_100m, arr100, T100, raster_crs, nodata, scale):
        """Aggregate 100m cells to higher resolution"""
        if len(base_cells_100m) == 0:
            return pd.DataFrame()
        
        parent_coords = base_cells_100m[['row', 'col']].copy()
        parent_coords['row'] = parent_coords['row'] // scale
        parent_coords['col'] = parent_coords['col'] // scale
        
        ward_assignments = base_cells_100m.groupby([parent_coords['row'], parent_coords['col']]).agg({
            'ward_id': 'first',
            'ward_name': 'first', 
            'state_name': 'first'
        }).reset_index()
        
        parent_cells = pd.DataFrame({
            'row': ward_assignments['row'],
            'col': ward_assignments['col'],
            'ward_id': ward_assignments['ward_id'],
            'ward_name': ward_assignments['ward_name'],
            'state_name': ward_assignments['state_name']
        })
        
        parent_cells = self._add_parent_cell_attributes(parent_cells, arr100, T100, raster_crs, nodata, scale)
        
        return parent_cells

    def _find_cells_for_ward(self, ward_geom, grid_transform, grid_height, grid_width, raster_crs, include_partial):
        """Find all 100m grid cells that intersect with a ward"""
        
        if str(raster_crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            ward_gdf_temp = gpd.GeoDataFrame([1], geometry=[ward_geom], crs="EPSG:4326")
            ward_gdf_raster = ward_gdf_temp.to_crs(raster_crs)
            ward_geom_raster = ward_gdf_raster.geometry.iloc[0]
        else:
            ward_geom_raster = ward_geom
        
        minx, miny, maxx, maxy = ward_geom_raster.bounds
        inv_transform = ~grid_transform
        
        col_min, row_max = inv_transform * (minx, miny)
        col_max, row_min = inv_transform * (maxx, maxy)
        
        row_min = max(0, int(np.floor(row_min)))
        row_max = min(grid_height - 1, int(np.ceil(row_max)))
        col_min = max(0, int(np.floor(col_min)))
        col_max = min(grid_width - 1, int(np.ceil(col_max)))
        
        intersecting_cells = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                x_min = grid_transform.c + col * grid_transform.a
                x_max = x_min + grid_transform.a
                y_max = grid_transform.f + row * grid_transform.e
                y_min = y_max + grid_transform.e
                
                from shapely.geometry import box
                cell_geom = box(x_min, y_min, x_max, y_max)
                
                if include_partial:
                    intersects = ward_geom_raster.intersects(cell_geom)
                else:
                    intersects = ward_geom_raster.contains(cell_geom)
                
                if intersects:
                    intersecting_cells.append({'row': row, 'col': col})
        
        return intersecting_cells

    def _add_100m_cell_attributes(self, cells_df, arr100, T100, raster_crs, nodata):
        """Add population and coordinate attributes to 100m cells"""
        
        if len(cells_df) == 0:
            return cells_df
        
        rows = cells_df['row'].values
        cols = cells_df['col'].values
        pop = arr100[rows, cols].astype(float)
        
        if nodata is not None:
            pop = np.where(pop == nodata, np.nan, pop)
        pop = np.nan_to_num(pop, nan=0.0)
        
        cx = T100 * (cells_df['col'] + 0.5, cells_df['row'] + 0.5)
        cx_x = np.asarray(cx[0], dtype=np.float64)
        cx_y = np.asarray(cx[1], dtype=np.float64)
        
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            lons = cx_x
            lats = cx_y
        else:
            tfm = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
            lons, lats = tfm.transform(cx_x, cx_y)
        
        cells_df = cells_df.copy()
        cells_df['population'] = pop
        cells_df['center_longitude'] = lons
        cells_df['center_latitude'] = lats
        cells_df['cell_id'] = "cell_100m_" + cells_df['row'].astype(str) + "_" + cells_df['col'].astype(str)
        cells_df['resolution_m'] = 100
        
        return cells_df

    def _add_parent_cell_attributes(self, parent_cells_df, arr100, T100, raster_crs, nodata, scale):
        """Add attributes to parent cells by aggregating from 100m children"""
        
        if len(parent_cells_df) == 0:
            return parent_cells_df
        
        prow = parent_cells_df['row'].to_numpy(np.int64)
        pcol = parent_cells_df['col'].to_numpy(np.int64)
        
        r0 = (scale * prow).astype(np.int64)
        c0 = (scale * pcol).astype(np.int64)
        
        H, W = arr100.shape
        
        def safe_pick(r, c):
            inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            rr = np.where(inb, r, 0)
            cc = np.where(inb, c, 0)
            vals = arr100[rr, cc].astype(float)
            if nodata is not None:
                vals = np.where(vals == nodata, 0.0, vals)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = np.where(inb, vals, 0.0)
            return vals
        
        pop = np.zeros_like(r0, dtype=float)
        for dr in range(scale):
            for dc in range(scale):
                r = r0 + dr
                c = c0 + dc
                vals = safe_pick(r, c)
                pop = pop + vals
        
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
        
        parent_cells_df = parent_cells_df.copy()
        parent_cells_df['population'] = pop
        parent_cells_df['center_longitude'] = lons
        parent_cells_df['center_latitude'] = lats
        parent_cells_df['cell_id'] = f"cell_{scale*100}m_" + parent_cells_df['row'].astype(str) + "_" + parent_cells_df['col'].astype(str)
        parent_cells_df['resolution_m'] = scale * 100
        
        return parent_cells_df

    def _map_visits_to_all_resolutions(self, visits_df, all_cells, T100, raster_crs):
        """Map visits to all resolution cells using index-based approach"""
        self.log("[ward_analysis] Mapping visits to all resolutions...")
        
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            visit_xs = visits_df["longitude"].to_numpy(np.float64)
            visit_ys = visits_df["latitude"].to_numpy(np.float64)
        else:
            tfm = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            visit_xs, visit_ys = tfm.transform(
                visits_df["longitude"].to_numpy(np.float64),
                visits_df["latitude"].to_numpy(np.float64),
            )
        
        inv_transform = ~T100
        cols_f, rows_f = inv_transform * (visit_xs, visit_ys)
        visit_cols_100m = np.floor(cols_f).astype(np.int64)
        visit_rows_100m = np.floor(rows_f).astype(np.int64)
        
        enriched_visits = visits_df.copy()
        
        for resolution in [100, 300]:
            if resolution in all_cells and len(all_cells[resolution]) > 0:
                cells_df = all_cells[resolution]
                
                if resolution == 100:
                    visit_rows = visit_rows_100m
                    visit_cols = visit_cols_100m
                else:
                    scale = resolution // 100
                    visit_rows = visit_rows_100m // scale
                    visit_cols = visit_cols_100m // scale
                
                cell_lookup = {}
                for _, cell in cells_df.iterrows():
                    key = (cell['row'], cell['col'])
                    cell_lookup[key] = {
                        'cell_id': cell['cell_id'],
                        'ward_id': cell['ward_id'],
                        'ward_name': cell['ward_name'],
                        'state_name': cell['state_name']
                    }
                
                cell_ids = []
                ward_ids = []
                ward_names = []
                state_names = []
                
                for row, col in zip(visit_rows, visit_cols):
                    key = (row, col)
                    if key in cell_lookup:
                        cell_info = cell_lookup[key]
                        cell_ids.append(cell_info['cell_id'])
                        ward_ids.append(cell_info['ward_id'])
                        ward_names.append(cell_info['ward_name'])
                        state_names.append(cell_info['state_name'])
                    else:
                        cell_ids.append(None)
                        ward_ids.append(None)
                        ward_names.append(None)
                        state_names.append(None)
                
                enriched_visits[f'cell_{resolution}m_id'] = cell_ids
                if resolution == 100:
                    enriched_visits['ward_id'] = ward_ids
                    enriched_visits['ward_name'] = ward_names
                    enriched_visits['state_name'] = state_names
            else:
                enriched_visits[f'cell_{resolution}m_id'] = None
                if resolution == 100:
                    enriched_visits['ward_id'] = None
                    enriched_visits['ward_name'] = None
                    enriched_visits['state_name'] = None
        
        for resolution in [100, 300]:
            mapped = enriched_visits[f'cell_{resolution}m_id'].notna().sum()
            self.log(f"[ward_analysis] {resolution}m: {mapped:,} visits mapped to cells")
        
        return enriched_visits

    def _map_buildings_to_all_resolutions(self, buildings_df, all_cells, T100, raster_crs):
        """Map buildings to all resolution cells using index-based approach"""
        self.log("[ward_analysis] Mapping buildings to all resolutions...")
        
        if str(raster_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
            bldg_xs = buildings_df["longitude"].to_numpy(np.float64)
            bldg_ys = buildings_df["latitude"].to_numpy(np.float64)
        else:
            tfm = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            bldg_xs, bldg_ys = tfm.transform(
                buildings_df["longitude"].to_numpy(np.float64),
                buildings_df["latitude"].to_numpy(np.float64),
            )
        
        inv_transform = ~T100
        cols_f, rows_f = inv_transform * (bldg_xs, bldg_ys)
        bldg_cols_100m = np.floor(cols_f).astype(np.int64)
        bldg_rows_100m = np.floor(rows_f).astype(np.int64)
        
        bldg_areas = buildings_df['area_in_meters'].to_numpy(np.float64)
        bldg_confidences = buildings_df['confidence'].to_numpy(np.float64)
        
        for resolution in [100, 300]:
            if resolution in all_cells and len(all_cells[resolution]) > 0:
                cells_df = all_cells[resolution]
                
                if resolution == 100:
                    bldg_rows = bldg_rows_100m
                    bldg_cols = bldg_cols_100m
                else:
                    scale = resolution // 100
                    bldg_rows = bldg_rows_100m // scale
                    bldg_cols = bldg_cols_100m // scale
                
                cells_df['num_buildings'] = 0
                cells_df['total_building_area_m2'] = 0.0
                cells_df['avg_building_confidence'] = 0.0

                cell_data = {}
                for i, (row, col) in enumerate(zip(bldg_rows, bldg_cols)):
                    key = (row, col)
                    if key not in cell_data:
                        cell_data[key] = {
                            'count': 0,
                            'total_area': 0.0,
                            'total_confidence': 0.0
                        }
                   
                    cell_data[key]['count'] += 1
                    cell_data[key]['total_area'] += bldg_areas[i]
                    cell_data[key]['total_confidence'] += bldg_confidences[i]
                 
                for idx, cell in cells_df.iterrows():
                    key = (cell['row'], cell['col'])
                    if key in cell_data:
                        data = cell_data[key]
                        cells_df.at[idx, 'num_buildings'] = data['count']
                        cells_df.at[idx, 'total_building_area_m2'] = data['total_area']
                        cells_df.at[idx, 'avg_building_confidence'] = (
                            data['total_confidence'] / data['count'] if data['count'] > 0 else 0.0
                        )
                
                all_cells[resolution] = cells_df
                
                mapped = (cells_df['num_buildings'] > 0).sum()
                total_buildings = cells_df['num_buildings'].sum()
                self.log(f"[ward_analysis] {resolution}m: {int(total_buildings):,} buildings mapped to {mapped:,} cells")

# grid3_ward_analysis.py - PART 3 OF 3 (FINAL)
# This continues from Part 2 - paste this immediately after Part 2

    def _add_visit_info_to_cells(self, cells_df, enriched_visits, resolution):
        """Add visit information to cell dataframe"""
        cells_with_visits = cells_df.copy()
        
        cells_with_visits['has_visits'] = False
        cells_with_visits['visits_in_cell'] = 0
        cells_with_visits['opportunity_ids'] = ''
        cells_with_visits['opportunity_names'] = ''
        cells_with_visits['unique_flws'] = 0
        
        cell_col = f'cell_{resolution}m_id'
        if cell_col in enriched_visits.columns:
            visits_with_cells = enriched_visits[enriched_visits[cell_col].notna()]
            
            if len(visits_with_cells) > 0:
                visit_agg = visits_with_cells.groupby(cell_col).agg({
                    'visit_id': 'count',
                    'opportunity_id': lambda x: ';'.join(sorted(x.unique().astype(str))),
                    'opportunity_name': lambda x: ';'.join(sorted(x.unique().astype(str))),
                    'flw_id': 'nunique'
                }).rename(columns={
                    'visit_id': 'visits_in_cell',
                    'opportunity_id': 'opportunity_ids',
                    'opportunity_name': 'opportunity_names',
                    'flw_id': 'unique_flws'
                })
                
                cells_with_visits = cells_with_visits.merge(
                    visit_agg, left_on='cell_id', right_index=True, how='left', suffixes=('', '_new')
                )
                
                visit_cols_to_update = ['visits_in_cell', 'opportunity_ids', 'opportunity_names', 'unique_flws']
                for col in visit_cols_to_update:
                    new_col = f'{col}_new'
                    if new_col in cells_with_visits.columns:
                        cells_with_visits[col] = cells_with_visits[new_col]
                        cells_with_visits = cells_with_visits.drop(columns=[new_col])
                
                cells_with_visits['visits_in_cell'] = cells_with_visits['visits_in_cell'].fillna(0).astype(int)
                cells_with_visits['opportunity_ids'] = cells_with_visits['opportunity_ids'].fillna('')
                cells_with_visits['unique_flws'] = cells_with_visits['unique_flws'].fillna(0).astype(int)
                cells_with_visits['has_visits'] = cells_with_visits['visits_in_cell'] > 0
        
        return cells_with_visits

    def _create_ward_summary_from_cells(self, cells_df, enriched_visits):
        """Create ward summary from cell-level data"""
        if len(cells_df) == 0:
            return pd.DataFrame()
        
        ward_summaries = []
        for ward_id in cells_df['ward_id'].unique():
            ward_cells = cells_df[cells_df['ward_id'] == ward_id]
            visited_cells = ward_cells[ward_cells['has_visits']]
            unvisited_cells = ward_cells[~ward_cells['has_visits']]
            
            all_opp_ids = []
            for opp_str in ward_cells['opportunity_ids']:
                if pd.notna(opp_str) and opp_str:
                    all_opp_ids.extend(opp_str.split(';'))
            unique_opportunities = len(set(all_opp_ids)) if all_opp_ids else 0
            
            all_opp_names = []
            for opp_str in ward_cells['opportunity_names']:
                if pd.notna(opp_str) and opp_str:
                    all_opp_names.extend(opp_str.split(';'))
            unique_opportunity_names = ';'.join(sorted(set(all_opp_names))) if all_opp_names else ''
            
            ward_visits = enriched_visits[enriched_visits['ward_id'] == ward_id]
            unique_flws_total = ward_visits['flw_id'].nunique() if len(ward_visits) > 0 else 0
            
            total_cells = len(ward_cells)
            total_cells_with_pop = len(ward_cells[ward_cells['population'] > 0])
            visited_cells_count = len(visited_cells)
            visit_cell_coverage_pct = (visited_cells_count / total_cells_with_pop) if total_cells_with_pop > 0 else 0
            
            total_pop = ward_cells['population'].sum()
            visited_pop = visited_cells['population'].sum() if len(visited_cells) > 0 else 0.0
            population_coverage_pct = (visited_pop / total_pop) if total_pop > 0 else 0
            
            total_visits = ward_cells['visits_in_cell'].sum()
            visits_per_1000_people = (total_visits / total_pop * 1000) if total_pop > 0 else 0
            
            expected_visits_per_pop = self.get_parameter_value('expected_visits_per_pop', 0.18)
            
            crude_coverage_total_pop = (total_visits / (total_pop * expected_visits_per_pop)) if total_pop > 0 else 0
            crude_coverage_visited_pop = (total_visits / (visited_pop * expected_visits_per_pop)) if visited_pop > 0 else 0
            
            total_buildings = int(ward_cells['num_buildings'].sum()) if 'num_buildings' in ward_cells.columns else 0
            buildings_in_visited_cells = int(visited_cells['num_buildings'].sum()) if len(visited_cells) > 0 and 'num_buildings' in visited_cells.columns else 0
            buildings_in_unvisited_cells = total_buildings - buildings_in_visited_cells
            total_building_area = ward_cells['total_building_area_m2'].sum() if 'total_building_area_m2' in ward_cells.columns else 0
            
            population_per_building = (total_pop / total_buildings) if total_buildings > 0 else 0
            buildings_per_1000_people = (total_buildings / total_pop * 1000) if total_pop > 0 else 0
            
            summary = {
                'ward_id': ward_id,
                'ward_name': ward_cells['ward_name'].iloc[0],
                'state_name': ward_cells['state_name'].iloc[0],
                'unique_opportunities': unique_opportunities,
                'opportunity_names': unique_opportunity_names,
                'unique_flws': unique_flws_total,
                'total_visits': total_visits,
                'total_cells': total_cells,
                'total_cells_with_pop': total_cells_with_pop,
                'visited_cells': visited_cells_count,
                'unvisited_cells': len(unvisited_cells),
                'total_population': total_pop,
                'visited_population': visited_pop,
                'unvisited_population': total_pop - visited_pop,
                'visits_per_1000_people': visits_per_1000_people,
                'crude_coverage_total_pop': crude_coverage_total_pop,
                'crude_coverage_visited_pop': crude_coverage_visited_pop,
                'visit_cell_coverage_pct': visit_cell_coverage_pct,
                'population_coverage_pct': population_coverage_pct,
                'total_buildings': total_buildings,
                'buildings_in_visited_cells': buildings_in_visited_cells,
                'buildings_in_unvisited_cells': buildings_in_unvisited_cells,
                'total_building_area_m2': total_building_area,
                'population_per_building': population_per_building,
                'buildings_per_1000_people': buildings_per_1000_people
            }
            
            ward_summaries.append(summary)
        
        return pd.DataFrame(ward_summaries)

    def _create_resolution_comparison(self, ward_summaries):
        """Create comparison table across resolutions"""
        comparison_rows = []
        
        for resolution in [100, 300]:
            if resolution in ward_summaries:
                ward_df = ward_summaries[resolution]
                
                if len(ward_df) > 0:
                    total_cells = ward_df['total_cells'].sum()
                    visited_cells = ward_df['visited_cells'].sum()
                    total_pop = ward_df['total_population'].sum()
                    visited_pop = ward_df['visited_population'].sum()
                    total_visits = ward_df['total_visits'].sum()
                    total_opportunities = ward_df['unique_opportunities'].sum()
                    total_flws = ward_df['unique_flws'].sum()
                    
                    total_buildings = ward_df['total_buildings'].sum() if 'total_buildings' in ward_df.columns else 0
                    buildings_visited = ward_df['buildings_in_visited_cells'].sum() if 'buildings_in_visited_cells' in ward_df.columns else 0
                    buildings_unvisited = ward_df['buildings_in_unvisited_cells'].sum() if 'buildings_in_unvisited_cells' in ward_df.columns else 0
                    
                    visit_coverage_pct = (visited_cells / total_cells * 100) if total_cells > 0 else 0
                    population_coverage_pct = (visited_pop / total_pop * 100) if total_pop > 0 else 0
                    visits_per_1000_people = (total_visits / total_pop * 1000) if total_pop > 0 else 0
                    
                    comparison_rows.append({
                        'resolution_m': resolution,
                        'total_wards': len(ward_df),
                        'total_cells': total_cells,
                        'visited_cells': visited_cells,
                        'unvisited_cells': total_cells - visited_cells,
                        'visit_coverage_pct': visit_coverage_pct,
                        'total_population': total_pop,
                        'visited_population': visited_pop,
                        'unvisited_population': total_pop - visited_pop,
                        'population_coverage_pct': population_coverage_pct,
                        'total_visits': total_visits,
                        'unique_opportunities': total_opportunities,
                        'unique_flws': total_flws,
                        'visits_per_1000_people': visits_per_1000_people,
                        'total_buildings': total_buildings,
                        'buildings_in_visited_cells': buildings_visited,
                        'buildings_in_unvisited_cells': buildings_unvisited
                    })
        
        return pd.DataFrame(comparison_rows)

    def _create_population_building_comparison(self, all_cells):
        """Create analysis comparing population and building distributions"""
        comparison_rows = []
        
        if 300 not in all_cells or len(all_cells[300]) == 0:
            return pd.DataFrame()
        
        cells_df = all_cells[300]
        
        if 'num_buildings' not in cells_df.columns:
            return pd.DataFrame()
        
        for ward_id in cells_df['ward_id'].unique():
            ward_cells = cells_df[cells_df['ward_id'] == ward_id]
            
            cells_pop_only = ward_cells[(ward_cells['population'] > 0) & (ward_cells['num_buildings'] == 0)]
            cells_buildings_only = ward_cells[(ward_cells['population'] == 0) & (ward_cells['num_buildings'] > 0)]
            cells_both = ward_cells[(ward_cells['population'] > 0) & (ward_cells['num_buildings'] > 0)]
            cells_neither = ward_cells[(ward_cells['population'] == 0) & (ward_cells['num_buildings'] == 0)]
            
            total_pop = ward_cells['population'].sum()
            total_buildings = ward_cells['num_buildings'].sum()
            
            pop_per_building = (total_pop / total_buildings) if total_buildings > 0 else 0
            buildings_per_1000_people = (total_buildings / total_pop * 1000) if total_pop > 0 else 0
            
            count_pop_only = len(cells_pop_only)
            count_buildings_only = len(cells_buildings_only)
            count_both = len(cells_both)
            count_neither = len(cells_neither)
            
            pop_in_pop_only = cells_pop_only['population'].sum()
            buildings_in_buildings_only = int(cells_buildings_only['num_buildings'].sum())
            pop_in_both = cells_both['population'].sum()
            buildings_in_both = int(cells_both['num_buildings'].sum())
            
            comparison_rows.append({
                'ward_id': ward_id,
                'ward_name': ward_cells['ward_name'].iloc[0],
                'state_name': ward_cells['state_name'].iloc[0],
                'total_population': total_pop,
                'total_buildings': int(total_buildings),
                'population_per_building': pop_per_building,
                'buildings_per_1000_people': buildings_per_1000_people,
                'cells_pop_only': count_pop_only,
                'cells_buildings_only': count_buildings_only,
                'cells_both': count_both,
                'cells_neither': count_neither,
                'pop_in_pop_only_cells': pop_in_pop_only,
                'buildings_in_buildings_only_cells': buildings_in_buildings_only,
                'pop_in_both_cells': pop_in_both,
                'buildings_in_both_cells': buildings_in_both,
                'pct_cells_with_mismatch': ((count_pop_only + count_buildings_only) / len(ward_cells) * 100) if len(ward_cells) > 0 else 0
            })
        
        return pd.DataFrame(comparison_rows)

    def _write_all_outputs(self, all_cells, enriched_visits, ward_boundaries_gdf, data_tag, output_dir, buildings_df):
        """Write all output files"""
        output_files = []
        
        # 1. Enriched visit file (CSV)
        visit_cols = ['visit_id', 'opportunity_id', 'flw_id', 'longitude', 'latitude',
                     'cell_100m_id', 'cell_300m_id',
                     'ward_id', 'ward_name', 'state_name']
        
        available_cols = [col for col in visit_cols if col in enriched_visits.columns]
        enriched_subset = enriched_visits[available_cols]
        
        visit_file = os.path.join(output_dir, f"enriched_visits_{data_tag}.csv")
        enriched_subset.to_csv(visit_file, index=False)
        output_files.append(visit_file)
        self.log(f"[ward_analysis] Wrote enriched visits: {os.path.basename(visit_file)}")
        
        # Create csvs subdirectory
        csvs_dir = os.path.join(output_dir, "csvs")
        os.makedirs(csvs_dir, exist_ok=True)
        
        # 2. Process each resolution with clustering
        from .Grid3CellClusterer import Grid3CellClusterer
        clusterer = Grid3CellClusterer(self.log)
        
        excel_file = os.path.join(output_dir, f"grid3_analysis_{data_tag}.xlsx")
        ward_summaries = {}
        sheets_written = 0
        all_data_for_excel = {}
        
        cells_300m_clustered = None
        
        for resolution in [100, 300]:
            if resolution in all_cells and len(all_cells[resolution]) > 0:
                cells_df = all_cells[resolution]
                
                cells_with_visits = self._add_visit_info_to_cells(cells_df, enriched_visits, resolution)
                
                cells_with_clusters, cluster_summaries = clusterer.add_clustering_to_cells(
                    cells_with_visits, resolution
                )
                
                if resolution == 300:
                    cells_300m_clustered = cells_with_clusters
                
                cell_csv_file = os.path.join(csvs_dir, f"grid_cells_{resolution}m_{data_tag}.csv")
                cells_with_clusters.to_csv(cell_csv_file, index=False)
                output_files.append(cell_csv_file)
                self.log(f"[ward_analysis] Wrote {resolution}m cells CSV: {os.path.basename(cell_csv_file)}")
                
                cluster_csv_file = os.path.join(csvs_dir, f"cluster_summary_{resolution}m_{data_tag}.csv")
                clusterer.create_cluster_summary_file(cluster_summaries, cluster_csv_file)
                output_files.append(cluster_csv_file)
                self.log(f"[ward_analysis] Wrote {resolution}m cluster summary CSV: {os.path.basename(cluster_csv_file)}")

                hamlet_analysis = clusterer.create_hamlet_analysis(cluster_summaries, cells_with_clusters)
                hamlet_csv_file = os.path.join(csvs_dir, f"hamlet_analysis_{resolution}m_{data_tag}.csv")
                clusterer.create_hamlet_analysis_file(hamlet_analysis, hamlet_csv_file)
                output_files.append(hamlet_csv_file)
                
                ward_summary = self._create_ward_summary_from_cells(cells_with_clusters, enriched_visits)
                ward_summaries[resolution] = ward_summary
                
                ward_csv_file = os.path.join(csvs_dir, f"ward_summary_{resolution}m_{data_tag}.csv")
                ward_summary.to_csv(ward_csv_file, index=False)
                output_files.append(ward_csv_file)
                self.log(f"[ward_analysis] Wrote {resolution}m wards CSV: {os.path.basename(ward_csv_file)}")
                
                all_data_for_excel[f"cells_{resolution}"] = self._format_dataframe_numbers(cells_with_clusters)
                all_data_for_excel[f"wards_{resolution}"] = self._format_dataframe_numbers(ward_summary)
                all_data_for_excel[f"clusters_{resolution}"] = self._format_dataframe_numbers(cluster_summaries)
                
            else:
                self.log(f"[ward_analysis] No cells found for {resolution}m resolution")
        
        # 4. Write resolution comparison CSV
        if ward_summaries:
            comparison_df = self._create_resolution_comparison(ward_summaries)
            comparison_csv_file = os.path.join(csvs_dir, f"resolution_comparison_{data_tag}.csv")
            comparison_df.to_csv(comparison_csv_file, index=False)
            output_files.append(comparison_csv_file)
            self.log(f"[ward_analysis] Wrote resolution comparison CSV: {os.path.basename(comparison_csv_file)}")
            
            all_data_for_excel["comparison"] = self._format_dataframe_numbers(comparison_df)
            
            # 4b. Write population vs building comparison if building data exists
            pop_bldg_comparison = self._create_population_building_comparison(all_cells)
            if len(pop_bldg_comparison) > 0:
                pop_bldg_csv_file = os.path.join(csvs_dir, f"population_building_comparison_{data_tag}.csv")
                pop_bldg_comparison.to_csv(pop_bldg_csv_file, index=False)
                output_files.append(pop_bldg_csv_file)
                self.log(f"[ward_analysis] Wrote population vs building comparison CSV: {os.path.basename(pop_bldg_csv_file)}")
                
                all_data_for_excel["pop_building_comparison"] = self._format_dataframe_numbers(pop_bldg_comparison)
        
        # 5. Create Excel workbook with formatted data
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for resolution in [100, 300]:
                    cells_key = f"cells_{resolution}"
                    wards_key = f"wards_{resolution}"
                    clusters_key = f"clusters_{resolution}"
                    
                    if cells_key in all_data_for_excel:
                        sheet_name = f"Cells_{resolution}m"
                        all_data_for_excel[cells_key].to_excel(writer, sheet_name=sheet_name, index=False)
                        sheets_written += 1
                        self.log(f"[ward_analysis] Added {resolution}m cells to Excel: {len(all_data_for_excel[cells_key]):,} rows")
                        
                        ward_sheet_name = f"Wards_{resolution}m"
                        all_data_for_excel[wards_key].to_excel(writer, sheet_name=ward_sheet_name, index=False)
                        sheets_written += 1
                        self.log(f"[ward_analysis] Added {resolution}m ward summary to Excel: {len(all_data_for_excel[wards_key]):,} rows")
                        
                        if clusters_key in all_data_for_excel and len(all_data_for_excel[clusters_key]) > 0:
                            cluster_sheet_name = f"Clusters_{resolution}m"
                            all_data_for_excel[clusters_key].to_excel(writer, sheet_name=cluster_sheet_name, index=False)
                            sheets_written += 1
                            self.log(f"[ward_analysis] Added {resolution}m cluster summary to Excel: {len(all_data_for_excel[clusters_key]):,} rows")
                
                if "comparison" in all_data_for_excel:
                    all_data_for_excel["comparison"].to_excel(writer, sheet_name="Resolution_Comparison", index=False)
                    sheets_written += 1
                    self.log(f"[ward_analysis] Added resolution comparison to Excel")
                
                if "pop_building_comparison" in all_data_for_excel:
                    all_data_for_excel["pop_building_comparison"].to_excel(writer, sheet_name="Pop_Building_Comparison", index=False)
                    sheets_written += 1
                    self.log(f"[ward_analysis] Added population vs building comparison to Excel")
                
                if sheets_written == 0:
                    summary_data = {
                        'Status': ['No cells found'],
                        'Visits_Loaded': [len(enriched_visits)],
                        'Wards_Loaded': [len(ward_boundaries_gdf)],
                        'Note': ['No grid cells intersected with ward boundaries']
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    self.log(f"[ward_analysis] No analysis data found - created summary sheet")
            
            output_files.append(excel_file)
            self.log(f"[ward_analysis] Wrote Excel file: {os.path.basename(excel_file)}")
            
        except Exception as e:
            self.log(f"[ward_analysis] Error creating Excel file: {str(e)}")
            self.log(f"[ward_analysis] Excel creation failed - but CSV files are available in csvs/ folder")
        
        # 6. Generate ward maps using clustered 300m cell data
        if cells_300m_clustered is not None and len(cells_300m_clustered) > 0:
            try:
                if buildings_df is not None:
                    self.log(f"[ward_analysis*] buildings_df has {len(buildings_df):,} rows and columns: {list(buildings_df.columns)}")
                else:
                    self.log(f"[ward_analysis] buildings_df is None!")
 
                self.log(f"[ward_analysis] Generating interactive ward maps with cluster and building information...")
                
                map_files = Grid3WardMapper.generate_ward_maps(
                    cells_300m_clustered,
                    ward_boundaries_gdf, 
                    enriched_visits,
                    output_dir,
                    buildings_df,
                    log_func=self.log
                )
                output_files.extend(map_files)
                self.log(f"[ward_analysis] Created {len(map_files)} ward maps")
            except Exception as e:
                self.log(f"[ward_analysis] Warning: Could not generate maps: {str(e)}")
        else:
            self.log(f"[ward_analysis] No 300m cell data available for mapping")

        return output_files

    @classmethod
    def create_for_automation(cls, df, output_dir, stage1_folder, grid3_file, include_partial=True, expected_visits_per_pop=0.18, buildings_dir=None):
        """Create instance for automation with direct parameters"""
        def log_func(message):
            print(message)
        
        instance = cls(df=df, output_dir=output_dir, log_callback=log_func, params_frame=None)
        instance._direct_params = {
            'stage1_folder': stage1_folder,
            'grid3_file': grid3_file,
            'partial': include_partial,
            'expected_visits_per_pop': expected_visits_per_pop,
            'buildings_dir': buildings_dir
        }
        return instance

    def get_parameter_value(self, param_name, default_value=""):
        """Get parameter value, checking direct params first, then falling back to GUI system"""
        if hasattr(self, '_direct_params') and param_name in self._direct_params:
            return self._direct_params[param_name]
        return super().get_parameter_value(param_name, default_value)