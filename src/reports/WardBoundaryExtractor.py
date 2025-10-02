# ward_boundary_extractor.py

"""
Ward Boundary Extractor Report

Extracts Nigerian ward boundaries that contain visit locations from Geonode shapefiles.
Discovers shapefiles automatically from state directories and performs spatial joins
to identify affected wards.

Input: Visit data with lat/lon coordinates
Output: 
- Shapefile of affected ward boundaries
- Excel summary of wards and visit counts
- Extraction log
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings

from .base_report import BaseReport

__VERSION__ = 'Ward Boundary Extractor v1.0'


class WardBoundaryExtractor(BaseReport):
    # ---------------- UI ----------------
    @staticmethod
    def setup_parameters(parent_frame):
        """Setup UI parameters for ward boundary extraction"""
        
        # Shapefile directory
        ttk.Label(parent_frame, text="Shapefile Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        shapefile_dir_var = tk.StringVar()
        
        # Try to load last-used directory
        config_file = os.path.join(os.path.expanduser("~"), ".ward_extractor_config.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    last_dir = f.read().strip()
                    if os.path.exists(last_dir):
                        shapefile_dir_var.set(last_dir)
            except Exception:
                pass

        dir_frame = ttk.Frame(parent_frame)
        dir_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Entry(dir_frame, textvariable=shapefile_dir_var, width=44).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(
            dir_frame,
            text="Browse...",
            command=lambda: WardBoundaryExtractor._browse_and_save_directory(shapefile_dir_var, config_file),
        ).grid(row=0, column=1, padx=(6, 0))
        dir_frame.columnconfigure(0, weight=1)

        # Buffer distance (optional)
        ttk.Label(parent_frame, text="Buffer Distance (meters):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        buffer_var = tk.StringVar(value="0")
        ttk.Entry(parent_frame, textvariable=buffer_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(parent_frame, text="(0 = no buffer, >0 = expand search area)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # Minimum visits threshold
        ttk.Label(parent_frame, text="Minimum visits per ward:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        min_visits_var = tk.StringVar(value="100")
        ttk.Entry(parent_frame, textvariable=min_visits_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(parent_frame, text="(Only include wards with at least this many visits)").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

        # Output format options
        ttk.Label(parent_frame, text="Output Format:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        format_var = tk.StringVar(value="Shapefile + GeoJSON")
        format_combo = ttk.Combobox(parent_frame, textvariable=format_var, 
                                   values=["Shapefile only", "GeoJSON only", "Shapefile + GeoJSON"],
                                   state="readonly", width=20)
        format_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Store variables for access in generate()
        parent_frame.shapefile_dir_var = shapefile_dir_var
        parent_frame.buffer_var = buffer_var
        parent_frame.min_visits_var = min_visits_var
        parent_frame.format_var = format_var

    @staticmethod
    def _browse_and_save_directory(var, config_file):
        """Browse for directory and save selection"""
        directory = filedialog.askdirectory(
            title="Select Shapefile Base Directory (containing state folders)"
        )
        if directory:
            var.set(directory)
            try:
                with open(config_file, "w") as f:
                    f.write(directory)
            except Exception:
                pass

    # ---------------- Main entry ----------------
    def generate(self):
        """Main entry point for ward boundary extraction"""
        output_files = []

        self.log('=================================================================')
        self.log(f'[{__VERSION__}] Starting ward boundary extraction')
        self.log('=================================================================')

        # Get parameters
        shapefile_dir = self.get_parameter_value("shapefile_dir", "")
        buffer_distance = float(self.get_parameter_value("buffer", "0"))
        min_visits = int(self.get_parameter_value("min_visits", "100"))
        output_format = self.get_parameter_value("format", "Shapefile + GeoJSON")

        if not shapefile_dir or not os.path.exists(shapefile_dir):
            raise ValueError("Please specify a valid shapefile directory")

        self.log(f"[extractor] Shapefile directory: {shapefile_dir}")
        self.log(f"[extractor] Buffer distance: {buffer_distance}m")
        self.log(f"[extractor] Minimum visits per ward: {min_visits}")

        # Create output directory
        today = datetime.now().strftime("%Y_%m_%d")
        output_dir = os.path.join(self.output_dir, f"ward_extractor_{today}")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare visit data
        visits_df = self._prepare_visit_data()
        self.log(f"[extractor] Loaded {len(visits_df)} valid visits")

        # Discover and load ward shapefiles
        ward_shapefiles = self._discover_ward_shapefiles(shapefile_dir)
        if not ward_shapefiles:
            raise ValueError(f"No ward shapefiles found in {shapefile_dir}")
        
        self.log(f"[extractor] Found {len(ward_shapefiles)} state ward files")
        
        # Load and merge all ward boundaries
        all_wards_gdf = self._load_and_merge_wards(ward_shapefiles)
        self.log(f"[extractor] Loaded {len(all_wards_gdf)} ward polygons")

        # Dissolve multi-part wards to ensure one record per ward_id
        dissolved_wards_gdf = self._dissolve_multipart_wards(all_wards_gdf)
        self.log(f"[extractor] Dissolved to {len(dissolved_wards_gdf)} unique wards")

        # Convert visits to GeoDataFrame
        visits_gdf = self._visits_to_geodataframe(visits_df, buffer_distance)
        
        # Perform spatial join to find affected wards and apply minimum visits filter
        affected_wards_gdf, visit_ward_summary = self._find_affected_wards(visits_gdf, dissolved_wards_gdf, min_visits)
        
        self.log(f"[extractor] Found {len(affected_wards_gdf)} affected wards (after min visits filter)")
        self.log(f"[extractor] {len(visit_ward_summary)} visits matched to qualifying wards")

        # Generate unique tag for this dataset
        data_tag = self._generate_data_tag(visits_df, affected_wards_gdf)
        self.log(f"[extractor] Dataset tag: {data_tag}")

        # Generate outputs with unique tag
        output_files.extend(self._write_outputs(
            affected_wards_gdf, visit_ward_summary, visits_df, output_dir, output_format, data_tag, min_visits
        ))

        self.log(f"[extractor] Complete! Generated {len(output_files)} files")
        return output_files

    # ---------------- Helper methods ----------------
    
    def _prepare_visit_data(self):
        """Prepare and validate visit data"""
        data = self.df.copy()
        data.columns = data.columns.str.lower().str.strip()

        # Find coordinate columns (same logic as Grid3 report)
        col_map = {
            "latitude": ["latitude", "lat", "y"],
            "longitude": ["longitude", "lon", "lng", "x"],
            "opportunity_id": ["opportunity_id"],
            "opportunity_name": ["opportunity_name", "opportunity"],
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
        
        oid_col = pick_column(col_map["opportunity_id"])
        oname_col = pick_column(col_map["opportunity_name"])
        
        if oid_col and oid_col in data.columns:
            rename_map[oid_col] = "opportunity_id"
        if oname_col and oname_col in data.columns:
            rename_map[oname_col] = "opportunity_name"
            
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

        return data

    def _discover_ward_shapefiles(self, base_dir):
        """Discover ward shapefiles following the Geonode pattern"""
        base_path = Path(base_dir)
        
        # Look for pattern: {state}/boundary_ward_default/boundary_ward_default.shp
        pattern = "**/boundary_ward_default/boundary_ward_default.shp"
        shapefiles = list(base_path.glob(pattern))
        
        #self.log(f"[extractor] Discovered shapefiles:")
        #for shp in shapefiles:
            #state_name = shp.parent.parent.name
            #self.log(f"[extractor]   - {state_name}: {shp}")
            
        return shapefiles

    def _load_and_merge_wards(self, shapefile_paths):
        """Load and merge all ward shapefiles into a single GeoDataFrame"""
        ward_gdfs = []
        
        for shp_path in shapefile_paths:
            try:
                state_name = shp_path.parent.parent.name
                #self.log(f"[extractor] Loading {state_name} wards...")
                
                # Suppress warnings about mixed geometry types
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gdf = gpd.read_file(shp_path)
                
                # Add state information
                gdf['state_name'] = state_name
                gdf['source_file'] = str(shp_path)
                
                # Ensure we have some kind of ward identifier
                if 'ward_name' not in gdf.columns:
                    # Look for common ward name columns
                    ward_name_candidates = ['WARD_NAME', 'Ward_Name', 'wardname', 'NAME', 'name']
                    for candidate in ward_name_candidates:
                        if candidate in gdf.columns:
                            gdf['ward_name'] = gdf[candidate]
                            break
                    else:
                        # Create a default ward identifier
                        gdf['ward_name'] = f"{state_name}_ward_" + gdf.index.astype(str)
                
                # Ensure CRS is WGS84 for consistent spatial operations
                if gdf.crs is None:
                    self.log(f"[extractor] Warning: {state_name} has no CRS, assuming WGS84")
                    gdf = gdf.set_crs("EPSG:4326")
                elif gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                
                ward_gdfs.append(gdf)
                #self.log(f"[extractor] Loaded {len(gdf)} wards from {state_name}")
                
            except Exception as e:
                self.log(f"[extractor] Error loading {shp_path}: {e}")
                continue
        
        if not ward_gdfs:
            raise ValueError("No ward shapefiles could be loaded")
        
        # Merge all wards
        all_wards = gpd.GeoDataFrame(pd.concat(ward_gdfs, ignore_index=True))
        
        # Create unique ward IDs
        all_wards['ward_id'] = all_wards['state_name'] + "_" + all_wards['ward_name'].astype(str)
        
        return all_wards

    def _dissolve_multipart_wards(self, wards_gdf):
        """Dissolve ward polygons by ward_id to handle multi-part wards"""
        self.log("[extractor] Dissolving multi-part wards...")
        
        # Group by ward_id and aggregate other attributes
        dissolved = wards_gdf.dissolve(by='ward_id', aggfunc={
            'ward_name': 'first',  # Take first occurrence
            'state_name': 'first',
            'source_file': lambda x: ';'.join(x.unique())  # Combine source files
        }).reset_index()
        
        # Ensure ward_id is preserved as a column (dissolve makes it an index)
        if 'ward_id' not in dissolved.columns:
            dissolved['ward_id'] = dissolved.index
        
        return dissolved

    def _visits_to_geodataframe(self, visits_df, buffer_distance):
        """Convert visits DataFrame to GeoDataFrame with optional buffering"""
        # Create Point geometries
        geometry = [Point(xy) for xy in zip(visits_df['longitude'], visits_df['latitude'])]
        
        visits_gdf = gpd.GeoDataFrame(visits_df, geometry=geometry, crs="EPSG:4326")
        
        # Apply buffer if specified
        if buffer_distance > 0:
            self.log(f"[extractor] Applying {buffer_distance}m buffer around visit points")
            # Convert to UTM for accurate buffering, then back to WGS84
            # For Nigeria, use UTM Zone 32N (EPSG:32632) as a reasonable approximation
            utm_crs = "EPSG:32632"
            visits_utm = visits_gdf.to_crs(utm_crs)
            visits_utm['geometry'] = visits_utm.geometry.buffer(buffer_distance)
            visits_gdf = visits_utm.to_crs("EPSG:4326")
        
        return visits_gdf

    def _find_affected_wards(self, visits_gdf, wards_gdf, min_visits):
        """Find wards that intersect with visit locations and apply minimum visits filter"""
        #self.log("[extractor] Performing spatial join...")
        
        # Spatial join to find which wards contain visits
        visit_ward_join = gpd.sjoin(visits_gdf, wards_gdf, how='left', predicate='intersects')
        
        # Create visit summary by ward
        visit_summary = visit_ward_join.groupby('ward_id').agg({
            'latitude': 'count',  # Total visits (using any non-null column)
            'opportunity_id': 'nunique',  # Unique opportunities
        }).rename(columns={
            'latitude': 'total_visits',
            'opportunity_id': 'unique_opportunities'
        })
        
        # Add opportunity details
        visit_summary['opportunity_list'] = visit_ward_join.groupby('ward_id')['opportunity_id'].apply(
            lambda x: '; '.join(sorted(x.unique().astype(str)))
        )
        
        # Apply minimum visits filter
        total_wards_before_filter = len(visit_summary)
        visit_summary = visit_summary[visit_summary['total_visits'] >= min_visits]
        wards_after_filter = len(visit_summary)
        
        self.log(f"[extractor] Filtered from {total_wards_before_filter} to {wards_after_filter} wards (min {min_visits} visits)")
        
        # Get qualifying ward IDs
        qualifying_ward_ids = visit_summary.index.tolist()
        
        # Filter wards to only qualifying ones
        affected_wards = wards_gdf[wards_gdf['ward_id'].isin(qualifying_ward_ids)].copy()
        
        return affected_wards, visit_summary

    def _generate_data_tag(self, visits_df, affected_wards_gdf):
        """Generate a unique tag based on dataset characteristics"""
        # Count unique FLWs (look for common FLW ID columns)
        flw_candidates = ['flw_id', 'user_id', 'username', 'enumerator_id', 'worker_id']
        unique_flws = 0
        for col in flw_candidates:
            if col in visits_df.columns:
                unique_flws = visits_df[col].nunique()
                break
        if unique_flws == 0:
            # Fallback: count unique lat/lon combinations as proxy for unique workers
            unique_flws = visits_df[['latitude', 'longitude']].drop_duplicates().shape[0]
        
        # Get other counts
        num_visits = len(visits_df)
        num_opps = visits_df['opportunity_id'].nunique()
        num_wards = len(affected_wards_gdf)
        
        # Create compact tag
        tag = f"{unique_flws}flw_{num_visits}v_{num_opps}opp_{num_wards}w"
        return tag

    def _write_outputs(self, affected_wards_gdf, visit_summary, visits_df, output_dir, output_format, data_tag, min_visits):
        """Write output files"""
        output_files = []
        
        # Merge ward data with visit summary
        output_wards = affected_wards_gdf.merge(visit_summary, left_on='ward_id', right_index=True, how='left')
        
        # Write spatial files
        base_filename = f"affected_wards_{data_tag}"
        
        if "Shapefile" in output_format:
            shp_path = os.path.join(output_dir, f"{base_filename}.shp")
            output_wards.to_file(shp_path)
            output_files.append(shp_path)
            self.log(f"[extractor] Wrote shapefile: {os.path.basename(shp_path)}")
        
        if "GeoJSON" in output_format:
            geojson_path = os.path.join(output_dir, f"{base_filename}.geojson")
            output_wards.to_file(geojson_path, driver='GeoJSON')
            output_files.append(geojson_path)
            self.log(f"[extractor] Wrote GeoJSON: {os.path.basename(geojson_path)}")
        
        # Write CSV summary instead of Excel to avoid data compatibility issues
        csv_path = os.path.join(output_dir, f"ward_summary_{data_tag}.csv")
        
        # Ward summary - only essential columns
        essential_columns = ['ward_id', 'ward_name', 'state_name', 'total_visits', 'unique_opportunities', 'opportunity_list']
        ward_summary_df = output_wards[essential_columns].copy()
        ward_summary_df = self._clean_dataframe_for_excel(ward_summary_df)
        ward_summary_df.to_csv(csv_path, index=False)
        
        # Write metadata as simple text file
        metadata_path = os.path.join(output_dir, f"extraction_metadata_{data_tag}.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Ward Boundary Extraction Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Visits Processed: {len(visits_df):,}\n")
            f.write(f"Visits Matched to Wards: {visit_summary['total_visits'].sum():,}\n")
            f.write(f"Affected Wards Found: {len(affected_wards_gdf)}\n")
            f.write(f"Unique Opportunities: {visits_df['opportunity_id'].nunique()}\n")
            f.write(f"States Covered: {affected_wards_gdf['state_name'].nunique()}\n")
            f.write(f"Minimum Visits Filter: {min_visits}\n")
        
        output_files.extend([csv_path, metadata_path])
        self.log(f"[extractor] Wrote CSV summary: {os.path.basename(csv_path)}")
        self.log(f"[extractor] Wrote metadata: {os.path.basename(metadata_path)}")
        
        return output_files

    def _clean_dataframe_for_excel(self, df):
        """Clean DataFrame for Excel compatibility"""
        cleaned_df = df.copy()
        
        # Convert object columns to string and handle problematic values
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                # Convert to string and replace problematic values
                cleaned_df[col] = cleaned_df[col].astype(str)
                cleaned_df[col] = cleaned_df[col].replace(['None', 'nan', 'NaT'], '')
            
            # Handle datetime columns
            elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].astype(str)
            
            # Handle numeric columns with potential issues
            elif cleaned_df[col].dtype in ['float64', 'int64']:
                # Replace inf/-inf with NaN, then fill NaN with 0
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                if cleaned_df[col].dtype == 'float64':
                    cleaned_df[col] = cleaned_df[col].fillna(0.0)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
        
        return cleaned_df
    @classmethod
    def create_for_automation(cls, df, output_dir, shapefile_dir, buffer_distance=0, min_visits=100, output_format="Shapefile + GeoJSON"):
        """Create instance for automation with direct parameters"""
        def log_func(message):
            print(message)
        
        instance = cls(df=df, output_dir=output_dir, log_callback=log_func, params_frame=None)
        instance._direct_params = {
            'shapefile_dir': shapefile_dir,
            'buffer': buffer_distance,
            'min_visits': min_visits,
            'format': output_format
        }
        return instance

    def get_parameter_value(self, param_name, default_value=""):
        """Get parameter value, checking direct params first, then falling back to GUI system"""
        if hasattr(self, '_direct_params') and param_name in self._direct_params:
            return self._direct_params[param_name]
        return super().get_parameter_value(param_name, default_value)