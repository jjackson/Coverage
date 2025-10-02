# opportunity_boundary_generator.py

"""
Opportunity Boundary Generator

Creates synthetic "proxy ward" boundaries around visit locations grouped by opportunity_id.
Uses convex hull with buffer to define service areas for each opportunity.
Produces output compatible with Grid3WardAnalysis pipeline.

Input: Visit data with lat/lon coordinates and opportunity_id
Output: 
- Shapefile of proxy ward boundaries (one per opportunity)
- Excel summary of proxy wards and visit counts
- Generation log

Key differences from WardBoundaryExtractor:
- Creates boundaries FROM visits rather than extracting existing boundaries
- Groups by opportunity_id instead of spatial intersection
- Uses convex hull + buffer instead of existing shapefiles
"""

import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union

from .base_report import BaseReport

__VERSION__ = 'Opportunity Boundary Generator v1.0'


class OpportunityBoundaryGenerator(BaseReport):
    # ---------------- UI ----------------
    @staticmethod
    def setup_parameters(parent_frame):
        """Setup UI parameters for opportunity boundary generation"""
        
        # Buffer distance
        ttk.Label(parent_frame, text="Convex Hull Buffer (meters):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        buffer_var = tk.StringVar(value="50")
        ttk.Entry(parent_frame, textvariable=buffer_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(parent_frame, text="(Buffer distance around convex hull)").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        # Maximum radius
        ttk.Label(parent_frame, text="Maximum Radius (km):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        max_radius_var = tk.StringVar(value="75")
        ttk.Entry(parent_frame, textvariable=max_radius_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(parent_frame, text="(Exclude visits beyond this distance from centroid)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # Minimum visits threshold
        ttk.Label(parent_frame, text="Minimum visits per opportunity:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        min_visits_var = tk.StringVar(value="500")
        ttk.Entry(parent_frame, textvariable=min_visits_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(parent_frame, text="(Only include opportunities with at least this many visits)").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

        # Output format options
        ttk.Label(parent_frame, text="Output Format:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        format_var = tk.StringVar(value="Shapefile + GeoJSON")
        format_combo = ttk.Combobox(parent_frame, textvariable=format_var, 
                                   values=["Shapefile only", "GeoJSON only", "Shapefile + GeoJSON"],
                                   state="readonly", width=20)
        format_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Store variables for access in generate()
        parent_frame.buffer_var = buffer_var
        parent_frame.max_radius_var = max_radius_var
        parent_frame.min_visits_var = min_visits_var
        parent_frame.format_var = format_var

    # ---------------- Main entry ----------------
    def generate(self):
        """Main entry point for opportunity boundary generation"""
        output_files = []

        self.log('=================================================================')
        self.log(f'[{__VERSION__}] Starting opportunity boundary generation')
        self.log('=================================================================')

        # Get parameters
        buffer_distance = float(self.get_parameter_value("buffer", "50"))
        max_radius_km = float(self.get_parameter_value("max_radius", "75"))
        min_visits = int(self.get_parameter_value("min_visits", "500"))
        output_format = self.get_parameter_value("format", "Shapefile + GeoJSON")

        self.log(f"[proxy_wards] Convex hull buffer: {buffer_distance}m")
        self.log(f"[proxy_wards] Maximum radius from centroid: {max_radius_km}km")
        self.log(f"[proxy_wards] Minimum visits per opportunity: {min_visits}")

        # Create output directory
        today = datetime.now().strftime("%Y_%m_%d")
        output_dir = os.path.join(self.output_dir, f"proxy_wards_{today}")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare visit data
        visits_df = self._prepare_visit_data()
        self.log(f"[proxy_wards] Loaded {len(visits_df)} valid visits")

        # Generate proxy ward boundaries
        proxy_wards_gdf, visit_summary = self._generate_proxy_wards(visits_df, buffer_distance, max_radius_km, min_visits)
        
        self.log(f"[proxy_wards] Generated {len(proxy_wards_gdf)} proxy ward boundaries")
        self.log(f"[proxy_wards] {len(visit_summary)} visits in qualifying opportunities")

        # Generate unique tag for this dataset
        data_tag = self._generate_data_tag(visits_df, proxy_wards_gdf)
        self.log(f"[proxy_wards] Dataset tag: {data_tag}")

        # Generate outputs with unique tag
        output_files.extend(self._write_outputs(
            proxy_wards_gdf, visit_summary, visits_df, output_dir, output_format, data_tag, min_visits
        ))

        self.log(f"[proxy_wards] Complete! Generated {len(output_files)} files")
        return output_files

    # ---------------- Helper methods ----------------
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between points using haversine formula
        Returns distance in kilometers
        
        lat1, lon1: arrays of latitudes and longitudes for points
        lat2, lon2: scalars for reference point (centroid)
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return c * r
    
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

    def _generate_proxy_wards(self, visits_df, buffer_distance, max_radius_km, min_visits):
        """Generate proxy ward boundaries from visit locations"""
        
        # Group visits by opportunity
        opp_groups = visits_df.groupby('opportunity_id')
        
        proxy_wards = []
        qualifying_visits = []
        
        for opp_id, opp_visits in opp_groups:
            # Apply minimum visits filter
            if len(opp_visits) < min_visits:
                continue
            
            # Get opportunity name
            opp_name = opp_visits['opportunity_name'].iloc[0]
            
            # Calculate centroid
            centroid_lon = opp_visits['longitude'].median()
            centroid_lat = opp_visits['latitude'].median()
            
            # Filter visits by distance from centroid
            distances_km = self._haversine_distance(
                opp_visits['latitude'].values,
                opp_visits['longitude'].values,
                centroid_lat,
                centroid_lon
            )
            
            within_radius = distances_km <= max_radius_km
            filtered_visits = opp_visits[within_radius]
            
            excluded_count = len(opp_visits) - len(filtered_visits)
            if excluded_count > 0:
                self.log(f"[proxy_wards] Opportunity {opp_id}: Excluded {excluded_count} visits beyond {max_radius_km}km radius")
            
            # Need at least min_visits after filtering
            if len(filtered_visits) < min_visits:
                self.log(f"[proxy_wards] Opportunity {opp_id}: Only {len(filtered_visits)} visits within radius - skipping")
                continue
            
            # Create points from filtered visit coordinates
            points = [Point(xy) for xy in zip(filtered_visits['longitude'], filtered_visits['latitude'])]
            
            # Create convex hull
            if len(points) < 3:
                # For 1-2 points, create a small buffer around the points
                multipoint = MultiPoint(points)
                hull = multipoint.buffer(buffer_distance / 111320)  # Rough conversion to degrees
            else:
                multipoint = MultiPoint(points)
                hull = multipoint.convex_hull
                
                # Apply buffer to the hull
                if buffer_distance > 0:
                    # Convert buffer from meters to degrees (rough approximation at equator)
                    # 1 degree ≈ 111,320 meters
                    buffer_degrees = buffer_distance / 111320
                    hull = hull.buffer(buffer_degrees)
            
            # Print bounding box diagnostics
            minx, miny, maxx, maxy = hull.bounds
            width_deg = maxx - minx
            height_deg = maxy - miny
            # Rough conversion: 1 degree ~ 111 km at equator
            width_km = width_deg * 111
            height_km = height_deg * 111
            area_km2 = width_km * height_km
            
            self.log(f"[proxy_wards] Opportunity {opp_id}: {len(filtered_visits)} visits (after filtering)")
            self.log(f"[proxy_wards]   Bounding box: {width_km:.1f} km x {height_km:.1f} km (~{area_km2:.0f} sq km)")
            self.log(f"[proxy_wards]   Bounds: lon [{minx:.4f}, {maxx:.4f}], lat [{miny:.4f}, {maxy:.4f}]")
            
            # Create proxy ward entry
            proxy_ward = {
                'ward_id': f'ward_proxy_{opp_id}',
                'ward_name': f'ward_proxy_{opp_id}',
                'state_name': f'state_proxy_{opp_id}',
                'opportunity_id': opp_id,
                'opportunity_name': opp_name,
                'total_visits': len(opp_visits),
                'unique_opportunities': 1,
                'geometry': hull
            }
            
            proxy_wards.append(proxy_ward)
            qualifying_visits.extend(opp_visits.to_dict('records'))
        
        # Create GeoDataFrame
        if proxy_wards:
            proxy_wards_gdf = gpd.GeoDataFrame(proxy_wards, crs="EPSG:4326")
            qualifying_visits_df = pd.DataFrame(qualifying_visits)
        else:
            proxy_wards_gdf = gpd.GeoDataFrame()
            qualifying_visits_df = pd.DataFrame()
        
        return proxy_wards_gdf, qualifying_visits_df

    def _generate_data_tag(self, visits_df, proxy_wards_gdf):
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
        num_wards = len(proxy_wards_gdf)
        
        # Create compact tag
        tag = f"{unique_flws}flw_{num_visits}v_{num_opps}opp_{num_wards}w"
        return tag

    def _write_outputs(self, proxy_wards_gdf, visit_summary, visits_df, output_dir, output_format, data_tag, min_visits):
        """Write output files"""
        output_files = []
        
        # Write spatial files (using same naming as WardBoundaryExtractor for compatibility)
        base_filename = f"affected_wards_{data_tag}"
        
        if "Shapefile" in output_format:
            shp_path = os.path.join(output_dir, f"{base_filename}.shp")
            proxy_wards_gdf.to_file(shp_path)
            output_files.append(shp_path)
            self.log(f"[proxy_wards] Wrote shapefile: {os.path.basename(shp_path)}")
        
        if "GeoJSON" in output_format:
            geojson_path = os.path.join(output_dir, f"{base_filename}.geojson")
            proxy_wards_gdf.to_file(geojson_path, driver='GeoJSON')
            output_files.append(geojson_path)
            self.log(f"[proxy_wards] Wrote GeoJSON: {os.path.basename(geojson_path)}")
        
        # Write CSV summary
        csv_path = os.path.join(output_dir, f"ward_summary_{data_tag}.csv")
        
        # Ward summary - only essential columns
        summary_columns = ['ward_id', 'ward_name', 'state_name', 'opportunity_id', 
                          'opportunity_name', 'total_visits', 'unique_opportunities']
        ward_summary_df = proxy_wards_gdf[summary_columns].copy()
        ward_summary_df.to_csv(csv_path, index=False)
        
        # Write metadata
        metadata_path = os.path.join(output_dir, f"extraction_metadata_{data_tag}.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Proxy Ward Boundary Generation Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Visits Processed: {len(visits_df):,}\n")
            f.write(f"Visits in Qualifying Opportunities: {len(visit_summary):,}\n")
            f.write(f"Proxy Wards Generated: {len(proxy_wards_gdf)}\n")
            f.write(f"Unique Opportunities: {visits_df['opportunity_id'].nunique()}\n")
            f.write(f"Minimum Visits Filter: {min_visits}\n")
        
        output_files.extend([csv_path, metadata_path])
        self.log(f"[proxy_wards] Wrote CSV summary: {os.path.basename(csv_path)}")
        self.log(f"[proxy_wards] Wrote metadata: {os.path.basename(metadata_path)}")
        
        return output_files

    @classmethod
    def create_for_automation(cls, df, output_dir, buffer_distance=50, max_radius_km=75, min_visits=500, output_format="Shapefile + GeoJSON"):
        """Create instance for automation with direct parameters"""
        def log_func(message):
            print(message)
        
        instance = cls(df=df, output_dir=output_dir, log_callback=log_func, params_frame=None)
        instance._direct_params = {
            'buffer': buffer_distance,
            'max_radius': max_radius_km,
            'min_visits': min_visits,
            'format': output_format
        }
        return instance

    def get_parameter_value(self, param_name, default_value=""):
        """Get parameter value, checking direct params first, then falling back to GUI system"""
        if hasattr(self, '_direct_params') and param_name in self._direct_params:
            return self._direct_params[param_name]
        return super().get_parameter_value(param_name, default_value)
