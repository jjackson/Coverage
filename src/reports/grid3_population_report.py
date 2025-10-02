"""
Grid3 Population Analysis Report

Analyzes visit data against Grid3 population grids at multiple resolutions.
Generates comprehensive population density analysis and maps.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path
import json
from datetime import datetime
from .base_report import BaseReport


class Grid3PopulationReport(BaseReport):
    """Report that analyzes visit data against Grid3 population data"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for Grid3 population analysis"""
        
        # Grid3 data file path
        ttk.Label(parent_frame, text="Grid3 TIF file:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        grid3_file_var = tk.StringVar()
        
        # Try to load the last used Grid3 file from a config file
        config_file = os.path.join(os.path.expanduser("~"), ".grid3_last_file.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    last_file = f.read().strip()
                    if os.path.exists(last_file):
                        grid3_file_var.set(last_file)
            except:
                pass  # If there's any error, just use empty default
        
        file_frame = ttk.Frame(parent_frame)
        file_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        ttk.Entry(file_frame, textvariable=grid3_file_var, width=40).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Browse...", 
                  command=lambda: Grid3PopulationReport._browse_and_save_grid3_file_static(grid3_file_var, config_file)).grid(row=0, column=1, padx=(5, 0))
        
        file_frame.columnconfigure(0, weight=1)
        
        # Analysis resolutions
        ttk.Label(parent_frame, text="Analysis resolutions:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        resolution_frame = ttk.Frame(parent_frame)
        resolution_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        res_100m_var = tk.BooleanVar(value=True)
        res_200m_var = tk.BooleanVar(value=False)
        res_500m_var = tk.BooleanVar(value=False)
        res_1km_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(resolution_frame, text="100m", variable=res_100m_var).pack(side=tk.LEFT)
        ttk.Checkbutton(resolution_frame, text="200m", variable=res_200m_var).pack(side=tk.LEFT, padx=(10,0))
        ttk.Checkbutton(resolution_frame, text="500m", variable=res_500m_var).pack(side=tk.LEFT, padx=(10,0))
        ttk.Checkbutton(resolution_frame, text="1km", variable=res_1km_var).pack(side=tk.LEFT, padx=(10,0))
        
        # Buffer distance around visit points
        ttk.Label(parent_frame, text="Visit buffer distance (meters):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        buffer_distance_var = tk.StringVar(value="0")
        ttk.Entry(parent_frame, textvariable=buffer_distance_var, width=15).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Population density categories
        ttk.Label(parent_frame, text="Population density thresholds:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        density_frame = ttk.Frame(parent_frame)
        density_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(density_frame, text="Low-Medium:").pack(side=tk.LEFT)
        low_medium_var = tk.StringVar(value="10")
        ttk.Entry(density_frame, textvariable=low_medium_var, width=8).pack(side=tk.LEFT, padx=(2,10))
        
        ttk.Label(density_frame, text="Medium-High:").pack(side=tk.LEFT)
        medium_high_var = tk.StringVar(value="50")
        ttk.Entry(density_frame, textvariable=medium_high_var, width=8).pack(side=tk.LEFT, padx=(2,0))
        
        # Time aggregation
        ttk.Label(parent_frame, text="Time aggregation:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        time_agg_var = tk.StringVar(value="Monthly")
        time_agg_combo = ttk.Combobox(parent_frame, textvariable=time_agg_var, 
                                     values=["All time", "Monthly", "Quarterly"], 
                                     state="readonly", width=15)
        time_agg_combo.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Output options
        ttk.Label(parent_frame, text="Generate interactive map:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        generate_map_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, variable=generate_map_var, 
                       text="Create HTML map visualization").grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(parent_frame, text="Export GIS formats:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        export_gis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent_frame, variable=export_gis_var, 
                       text="Export GeoJSON and Shapefile").grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Store variables in parent frame
        parent_frame.grid3_file_var = grid3_file_var
        parent_frame.res_100m_var = res_100m_var
        parent_frame.res_200m_var = res_200m_var
        parent_frame.res_500m_var = res_500m_var
        parent_frame.res_1km_var = res_1km_var
        parent_frame.buffer_distance_var = buffer_distance_var
        parent_frame.low_medium_var = low_medium_var
        parent_frame.medium_high_var = medium_high_var
        parent_frame.time_agg_var = time_agg_var
        parent_frame.generate_map_var = generate_map_var
        parent_frame.export_gis_var = export_gis_var
    
    @staticmethod
    def _browse_and_save_grid3_file_static(grid3_file_var, config_file):
        """Browse for Grid3 file and save the selection"""
        filename = filedialog.askopenfilename(
            title="Select Grid3 TIF file",
            filetypes=[("TIF files", "*.tif"), ("All files", "*.*")]
        )
        if filename:
            grid3_file_var.set(filename)
            # Save the selected file to config for next time
            try:
                with open(config_file, 'w') as f:
                    f.write(filename)
            except:
                pass  # If we can't save, that's okay
    
    def generate(self):
        """Generate Grid3 population analysis reports"""
        output_files = []
        
        # Get parameters
        grid3_file = self.get_parameter_value('grid3_file', '')
        buffer_distance = float(self.get_parameter_value('buffer_distance', '50'))
        low_medium_threshold = float(self.get_parameter_value('low_medium', '10'))
        medium_high_threshold = float(self.get_parameter_value('medium_high', '50'))
        time_aggregation = self.get_parameter_value('time_agg', 'Monthly')
        generate_map = self.get_parameter_value('generate_map', True)
        export_gis = self.get_parameter_value('export_gis', False)
        
        # Get requested resolutions
        resolutions = []
        if self.get_parameter_value('res_100m', True):
            resolutions.append(100)
        if self.get_parameter_value('res_200m', False):
            resolutions.append(200)
        if self.get_parameter_value('res_500m', False):
            resolutions.append(500)
        if self.get_parameter_value('res_1km', False):
            resolutions.append(1000)
        
        self.log(f"Starting Grid3 population analysis")
        self.log(f"Resolutions: {resolutions}m")
        self.log(f"Buffer distance: {buffer_distance}m")
        self.log(f"Population density thresholds: {low_medium_threshold}, {medium_high_threshold}")
        
        # Auto-detect Grid3 file if not specified
        if not grid3_file or not os.path.exists(grid3_file):
            grid3_file = self._find_grid3_file()
        
        if not grid3_file:
            raise ValueError("No Grid3 TIF file specified or found. Please browse to select your Grid3 population file.")
        
        self.log(f"Using Grid3 file: {os.path.basename(grid3_file)}")
        
        # Create output directory
        today = datetime.now().strftime('%Y_%m_%d')
        output_dir = os.path.join(self.output_dir, f"grid3_analysis_{today}")
        os.makedirs(output_dir, exist_ok=True)
        self.log(f"Created output directory: {os.path.basename(output_dir)}")
        
        # Validate and prepare visit data
        visits_data = self._prepare_visit_data()
        self.log(f"Prepared {len(visits_data)} visit records for analysis")
        
        # Load and validate Grid3 data
        population_grids = self._load_grid3_data(grid3_file, resolutions)
        
        # Perform spatial analysis for each resolution
        all_results = []
        
        for resolution in resolutions:
            self.log(f"Analyzing visits against {resolution}m population grid...")
            
            # Get population grid for this resolution
            pop_grid = population_grids[resolution]
            
            # Spatial join visits to grid cells
            grid_results = self._analyze_visits_by_grid(visits_data, pop_grid, buffer_distance, resolution)
            
            # Add density categories
            grid_results = self._add_density_categories(grid_results, low_medium_threshold, medium_high_threshold)
            
            # Add to results
            grid_results['resolution_m'] = resolution
            all_results.append(grid_results)
            
            self.log(f"  Found visits in {len(grid_results)} grid cells at {resolution}m resolution")
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        
        # Generate time series analysis if requested
        time_series_data = None
        if (False):  # TIME_SERIES_ANALYSIS
            if time_aggregation != "All time":
                time_series_data = self._generate_time_series_analysis(visits_data, population_grids, time_aggregation, buffer_distance)
        
        # Create Excel output
        excel_file = self._create_excel_output(combined_results, time_series_data, visits_data, output_dir, resolutions)
        output_files.append(excel_file)
        
        # Generate interactive map if requested
        if (False):  # INTERACTIVE_MAP
            if generate_map and not combined_results.empty:
                map_file = self._create_interactive_map(combined_results, visits_data, output_dir)
                if map_file:
                    output_files.append(map_file)
        
        # Export GIS formats if requested
        if (False):  # GIS_EXPORT
            if export_gis and not combined_results.empty:
                gis_files = self._export_gis_formats(combined_results, output_dir)
                output_files.extend(gis_files)
        
        self.log(f"Grid3 population analysis complete! Generated {len(output_files)} files")
        
        # Log information about both tabs
        if not combined_results.empty:
            total_rows = len(combined_results)
            self.log(f"  Tab 1 (Grid_Cells): {total_rows} rows (one per grid cell per opportunity)")
            
            # Check opportunity summary info
            if 'opportunity_id' in combined_results.columns:
                unique_opps = combined_results['opportunity_id'].nunique()
                self.log(f"  Tab 2 (Opportunity_Summary): {unique_opps} opportunities from 100m data")
            else:
                self.log(f"  Tab 2 (Opportunity_Summary): No opportunity data found")
        
        # Log which features are disabled
        self.log(f"  NOTE: Time series analysis is DISABLED (set if (TRUE) to enable)")
        self.log(f"  NOTE: Interactive maps are DISABLED (set if (TRUE) to enable)")
        self.log(f"  NOTE: GIS export is DISABLED (set if (TRUE) to enable)")
        self.log(f"  NOTE: Multiple Excel sheets are DISABLED (set if (TRUE) to enable)")
        self.log(f"  NOTE: Complex aggregations are DISABLED (set if (TRUE) to enable)")
        
        return output_files
    
    def _find_grid3_file(self):
        """Auto-detect Grid3 TIF files in the project data directory"""
        
        # Look for Grid3 files in common locations
        search_paths = [
            Path(self.output_dir).parent / "data" / "grid3",
            Path(self.output_dir).parent / "data" / "grid3" / "kenya",
            Path(self.output_dir).parent / "data" / "grid3" / "population",
            Path(self.output_dir) / ".." / "data" / "grid3"
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                # Look for TIF files
                tif_files = list(search_path.rglob("*.tif"))
                if tif_files:
                    # Prefer files with 'population' or 'pop' in the name
                    pop_files = [f for f in tif_files if 'pop' in f.name.lower()]
                    if pop_files:
                        self.log(f"Auto-detected Grid3 file: {pop_files[0]}")
                        return str(pop_files[0])
                    else:
                        # Return the first TIF file found
                        self.log(f"Auto-detected Grid3 file: {tif_files[0]}")
                        return str(tif_files[0])
        
        return None
    
    def _prepare_visit_data(self):
        """Prepare and validate visit data for Grid3 analysis"""
        
        # Use the same column detection logic as the clustering report
        data = self.df.copy()
        data.columns = data.columns.str.lower().str.strip()
        
        # Check for required columns
        required_cols = ['latitude', 'longitude']
        col_variations = {
            'latitude': ['latitude', 'lat', 'y'],
            'longitude': ['longitude', 'lon', 'lng', 'x'],
            'opportunity_id': ['opportunity_id'],
            'flw_id': ['flw_id', 'flwid', 'flw', 'agent_id'],
            'date': ['date', 'visit_date', 'created_at', 'timestamp']
        }
        
        final_cols = {}
        for req_col in required_cols:
            for variation in col_variations[req_col]:
                if variation in data.columns:
                    final_cols[req_col] = variation
                    break
        
        # Optional columns
        for opt_col in ['opportunity_id', 'flw_id', 'date']:
            for variation in col_variations[opt_col]:
                if variation in data.columns:
                    final_cols[opt_col] = variation
                    break
        
        if len(final_cols) < 2:  # Need at least lat/lon
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns (latitude, longitude). Available columns: {available_cols}")
        
        # Rename columns to standard names
        data = data.rename(columns={v: k for k, v in final_cols.items()})
        
        # Remove rows with missing coordinates
        before_clean = len(data)
        data = data.dropna(subset=['latitude', 'longitude'])
        after_clean = len(data)
        if before_clean != after_clean:
            self.log(f"Removed {before_clean - after_clean} records with missing coordinates")
        
        # Validate coordinate ranges
        invalid_coords = (
            (data['latitude'] < -90) | (data['latitude'] > 90) |
            (data['longitude'] < -180) | (data['longitude'] > 180)
        )
        if invalid_coords.any():
            invalid_count = invalid_coords.sum()
            self.log(f"Warning: Removed {invalid_count} records with invalid coordinates")
            data = data[~invalid_coords]
        
        # Add visit ID if not present
        if 'visit_id' not in data.columns:
            data['visit_id'] = [f"visit_{i+1}" for i in range(len(data))]
        
        # Convert date if present
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'])
            except:
                self.log("Warning: Could not parse date column")
        
        return data.reset_index(drop=True)
    
    def _load_grid3_data(self, grid3_file, resolutions):
        """Load Grid3 population data and create grids for different resolutions"""
        
        self.log(f"Loading Grid3 population data from: {os.path.basename(grid3_file)}")
        
        population_grids = {}
        
        try:
            with rasterio.open(grid3_file) as src:
                self.log(f"  Grid dimensions: {src.width} x {src.height} pixels")
                self.log(f"  Coordinate system: {src.crs}")
                self.log(f"  Bounds: {src.bounds}")
                
                # Calculate pixel resolution
                bounds = src.bounds
                pixel_width_m = (bounds.right - bounds.left) / src.width * 111320
                pixel_height_m = (bounds.top - bounds.bottom) / src.height * 111320
                self.log(f"  Estimated pixel size: {pixel_width_m:.0f}m x {pixel_height_m:.0f}m")
                
                # Check if this looks like 100m resolution data
                base_resolution = 100
                if pixel_width_m > 150:
                    # Might be coarser resolution data
                    base_resolution = int(round(pixel_width_m, -1))  # Round to nearest 10
                    self.log(f"  Detected base resolution: ~{base_resolution}m")
                
                # Load data for each requested resolution
                for resolution in resolutions:
                    self.log(f"  Creating {resolution}m resolution grid...")
                    
                    if resolution == base_resolution:
                        # Use original resolution
                        population_grids[resolution] = self._raster_to_grid_dataframe(src, resolution)
                    else:
                        # Resample to target resolution
                        population_grids[resolution] = self._resample_grid_data(src, resolution, base_resolution)
                    
                    grid_count = len(population_grids[resolution])
                    total_pop = population_grids[resolution]['population'].sum()
                    self.log(f"    Created {grid_count} grid cells, total population: {total_pop:,.0f}")
        
        except Exception as e:
            raise ValueError(f"Error loading Grid3 file: {str(e)}")
        
        return population_grids

    def _resample_grid_data(self, src, target_resolution, base_resolution):
        """Resample grid data using padded block reduction - preserves all original data"""
        
        self.log(f"    Block-reducing from {base_resolution}m to {target_resolution}m...")
        
        # Calculate reduction factor (should be integer for clean block reduction)
        factor = int(target_resolution / base_resolution)
        
        if factor < 2:
            self.log(f"    Cannot reduce by factor < 2, using original data")
            return self._raster_to_grid_dataframe(src, target_resolution)
        
        try:
            # Read the original population data
            population_data = src.read(1)
            original_transform = src.transform
            
            self.log(f"    Original data shape: {population_data.shape}")
            self.log(f"    Block reduction factor: {factor}x{factor}")
            
            # Calculate padding needed to make dimensions divisible by factor
            height, width = population_data.shape
            pad_height = factor - (height % factor) if height % factor != 0 else 0
            pad_width = factor - (width % factor) if width % factor != 0 else 0
            
            # Pad with zeros instead of trimming
            if pad_height > 0 or pad_width > 0:
                padded_data = np.pad(population_data, ((0, pad_height), (0, pad_width)), 
                                   mode='constant', constant_values=0)
                self.log(f"    Padded from {population_data.shape} to {padded_data.shape} (added {pad_height}x{pad_width} zero cells)")
            else:
                padded_data = population_data
                self.log(f"    No padding needed - dimensions already divisible by {factor}")
            
            # Reshape into blocks and sum
            new_height, new_width = padded_data.shape
            blocked = padded_data.reshape(
                new_height // factor, factor,
                new_width // factor, factor
            )
            
            # Sum over the block dimensions (axes 1 and 3)
            reduced_data = blocked.sum(axis=(1, 3))
            
            self.log(f"    Reduced to {reduced_data.shape} ({target_resolution}m cells)")
            self.log(f"    Population totals - Original: {population_data.sum():,.0f}, Padded: {padded_data.sum():,.0f}, Reduced: {reduced_data.sum():,.0f}")
            
            # Create new transform for the reduced resolution
            # Scale by the reduction factor to make pixels larger
            from affine import Affine
            new_transform = original_transform * Affine.scale(factor, factor)
            
            # Convert reduced array to grid cells
            mask = reduced_data > 0
            
            if not np.any(mask):
                self.log(f"    No populated cells after reduction")
                return gpd.GeoDataFrame({
                    'cell_id': [], 'population': [], 'center_latitude': [], 'center_longitude': [], 'geometry': []
                }, crs='EPSG:4326')
            
            self.log(f"    Converting {np.count_nonzero(mask)} reduced cells to geometries...")
            
            # Use rasterio.features.shapes to convert to polygons
            shapes_iter = shapes(reduced_data, mask=mask, transform=new_transform)
            
            grid_cells = []
            for i, (geom, pop_value) in enumerate(shapes_iter):
                if pop_value > 0:
                    # Convert GeoJSON dict to Shapely geometry
                    from shapely.geometry import shape
                    if isinstance(geom, dict):
                        shapely_geom = shape(geom)
                    else:
                        shapely_geom = geom
                    
                    grid_cells.append({
                        'cell_id': f"cell_{target_resolution}m_{i}",
                        'population': float(pop_value),
                        'geometry': shapely_geom
                    })
            
            if len(grid_cells) == 0:
                self.log(f"    No valid grid cells created")
                return gpd.GeoDataFrame({
                    'cell_id': [], 'population': [], 'center_latitude': [], 'center_longitude': [], 'geometry': []
                }, crs='EPSG:4326')
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(grid_cells, crs=src.crs)
            
            # Convert to WGS84 if needed
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Add centroid coordinates
            centroids = gdf.geometry.centroid
            gdf['center_latitude'] = centroids.y
            gdf['center_longitude'] = centroids.x
            
            self.log(f"    Successfully created {len(gdf)} grid cells at {target_resolution}m resolution")
            self.log(f"    Population range: {gdf['population'].min():.1f} to {gdf['population'].max():.1f}")
            
            return gdf
            
        except Exception as e:
            self.log(f"    Error in padded block reduction: {str(e)}")
            self.log(f"    Falling back to empty result")
            return gpd.GeoDataFrame({
                'cell_id': [], 'population': [], 'center_latitude': [], 'center_longitude': [], 'geometry': []
            }, crs='EPSG:4326')
    
    def _raster_to_grid_dataframe(self, src, resolution):
        """Convert raster data to GeoDataFrame of grid cells - Fixed geometry version"""
        
        self.log(f"    Reading population data from raster...")
        
        try:
            # Read the population data
            population_data = src.read(1)
            transform = src.transform
            
            self.log(f"    Data shape: {population_data.shape}")
            self.log(f"    Data range: {population_data.min()} to {population_data.max()}")
            self.log(f"    Non-zero pixels: {np.count_nonzero(population_data)}")
            
            # Create mask for pixels with population > 0
            mask = population_data > 0
            
            if not np.any(mask):
                self.log(f"    Warning: No pixels with population > 0 found")
                return gpd.GeoDataFrame({
                    'cell_id': [],
                    'population': [],
                    'center_latitude': [],
                    'center_longitude': [],
                    'geometry': []
                }, crs='EPSG:4326')
            
            self.log(f"    Converting {np.count_nonzero(mask)} populated pixels to grid cells...")
            
            try:
                # Get shapes (polygons) for each pixel with population
                self.log(f"    Generating shapes from raster...")
                shapes_iter = list(shapes(population_data, mask=mask, transform=transform))
                self.log(f"    Generated {len(shapes_iter)} shapes")
                
                if len(shapes_iter) == 0:
                    self.log(f"    No shapes generated from raster")
                    return gpd.GeoDataFrame({
                        'cell_id': [],
                        'population': [],
                        'center_latitude': [],
                        'center_longitude': [],
                        'geometry': []
                    }, crs='EPSG:4326')
                
                grid_cells = []
                self.log(f"    Processing shapes into grid cells...")
                
                # Import shapely for geometry conversion
                from shapely.geometry import shape
                
                for i, (geom, pop_value) in enumerate(shapes_iter):
                    if pop_value > 0:  # Double-check population value
                        # Convert GeoJSON-like dict to Shapely geometry
                        if isinstance(geom, dict):
                            shapely_geom = shape(geom)
                        else:
                            shapely_geom = geom
                        
                        grid_cells.append({
                            'cell_id': f"cell_{resolution}m_{i}",
                            'population': float(pop_value),
                            'geometry': shapely_geom
                        })
                        
                        # Log progress for large datasets
                        if i > 0 and i % 100000 == 0:
                            self.log(f"      Processed {i} shapes...")
                        elif i < 3:  # Log first few for debugging
                            self.log(f"      Sample cell {i}: pop={pop_value}, geom_type={type(shapely_geom).__name__}")
                
                self.log(f"    Created {len(grid_cells)} grid cells with population > 0")
                
                if len(grid_cells) == 0:
                    self.log(f"    No grid cells with population > 0")
                    return gpd.GeoDataFrame({
                        'cell_id': [],
                        'population': [],
                        'center_latitude': [],
                        'center_longitude': [],
                        'geometry': []
                    }, crs='EPSG:4326')
                
                # Create GeoDataFrame
                self.log(f"    Creating GeoDataFrame...")
                gdf = gpd.GeoDataFrame(grid_cells, crs=src.crs)
                self.log(f"    GeoDataFrame created with CRS: {gdf.crs}")
                
                # Ensure we're in WGS84 for consistency
                if gdf.crs != 'EPSG:4326':
                    self.log(f"    Converting from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs('EPSG:4326')
                    self.log(f"    CRS conversion complete")
                
                # Add centroid coordinates for easier analysis
                self.log(f"    Calculating centroids...")
                centroids = gdf.geometry.centroid
                gdf['center_latitude'] = centroids.y
                gdf['center_longitude'] = centroids.x
                
                # Log final statistics
                self.log(f"    Final GeoDataFrame statistics:")
                self.log(f"      - Total cells: {len(gdf)}")
                self.log(f"      - Population range: {gdf['population'].min():.2f} to {gdf['population'].max():.2f}")
                self.log(f"      - Total population: {gdf['population'].sum():,.0f}")
                self.log(f"      - Latitude range: {gdf['center_latitude'].min():.6f} to {gdf['center_latitude'].max():.6f}")
                self.log(f"      - Longitude range: {gdf['center_longitude'].min():.6f} to {gdf['center_longitude'].max():.6f}")
                
                self.log(f"    Successfully created GeoDataFrame with {len(gdf)} grid cells")
                return gdf
                
            except Exception as e:
                self.log(f"    Error in shape processing: {str(e)}")
                self.log(f"    Error type: {type(e).__name__}")
                import traceback
                self.log(f"    Traceback: {traceback.format_exc()}")
                # Return empty GeoDataFrame on error
                return gpd.GeoDataFrame({
                    'cell_id': [],
                    'population': [],
                    'center_latitude': [],
                    'center_longitude': [],
                    'geometry': []
                }, crs='EPSG:4326')
                
        except Exception as e:
            self.log(f"    Error reading raster data: {str(e)}")
            self.log(f"    Error type: {type(e).__name__}")
            import traceback
            self.log(f"    Traceback: {traceback.format_exc()}")
            # Return empty GeoDataFrame on error
            return gpd.GeoDataFrame({
                'cell_id': [],
                'population': [],
                'center_latitude': [],
                'center_longitude': [],
                'geometry': []
            }, crs='EPSG:4326')


    def _analyze_visits_by_grid(self, visits_data, pop_grid, buffer_distance, resolution):
        """Simplified analysis - just count visits per grid cell per opportunity"""
        
        if len(pop_grid) == 0:
            self.log(f"  Warning: No population grid cells found for {resolution}m resolution")
            return pd.DataFrame()
        
        self.log(f"  Creating visit points from {len(visits_data)} visits...")
        
        # Create GeoDataFrame of visit points
        geometry = [Point(xy) for xy in zip(visits_data['longitude'], visits_data['latitude'])]
        visits_gdf = gpd.GeoDataFrame(visits_data, geometry=geometry, crs='EPSG:4326')
        
        # Apply buffer if specified
        if buffer_distance > 0:
            self.log(f"  Applying {buffer_distance}m buffer to visit points...")
            visits_projected = visits_gdf.to_crs('EPSG:3857')
            visits_projected.geometry = visits_projected.geometry.buffer(buffer_distance)
            visits_gdf = visits_projected.to_crs('EPSG:4326')
        
        self.log(f"  Performing spatial join...")
        
        # Spatial join
        joined = gpd.sjoin(pop_grid, visits_gdf, how='inner', predicate='intersects')
        
        if len(joined) == 0:
            self.log(f"  No intersections found")
            return pd.DataFrame()
        
        self.log(f"  Found {len(joined)} visit-grid intersections")
        
        # Simple aggregation: count visits by grid cell and opportunity
        result = joined.groupby([
            'cell_id', 
            'population', 
            'center_latitude', 
            'center_longitude',
            'opportunity_id',
            'opportunity_name'
        ]).size().reset_index(name='total_visits')
        
        # Add derived metrics
        result['visits_per_capita'] = result['total_visits'] / result['population']
        result['population_density_per_km2'] = result['population'] / ((resolution/1000) ** 2)
        
        self.log(f"  Results: {len(result)} rows (grid cell + opportunity combinations)")
        self.log(f"  Total visits: {result['total_visits'].sum()}")
        
        return result

    def _add_density_categories(self, grid_results, low_medium_threshold, medium_high_threshold):
        """Add population density categories to results"""
        
        if 'population_density_per_km2' not in grid_results.columns:
            return grid_results
        
        def categorize_density(density):
            if density < low_medium_threshold:
                return 'Low'
            elif density < medium_high_threshold:
                return 'Medium'
            else:
                return 'High'
        
        grid_results['density_category'] = grid_results['population_density_per_km2'].apply(categorize_density)
        
        return grid_results
    
    def _generate_time_series_analysis(self, visits_data, population_grids, time_aggregation, buffer_distance):
        """Generate time-based analysis of visits by population density"""
        
        if 'date' not in visits_data.columns:
            return None
        
        # For time series, use the finest resolution grid
        finest_resolution = min(population_grids.keys())
        pop_grid = population_grids[finest_resolution]
        
        # Analyze visits by grid (reuse existing method)
        grid_results = self._analyze_visits_by_grid(visits_data, pop_grid, buffer_distance, finest_resolution)
        
        if grid_results.empty:
            return None
        
        # Add time period column to visits
        visits_with_time = visits_data.copy()
        if time_aggregation == "Monthly":
            visits_with_time['time_period'] = visits_with_time['date'].dt.to_period('M')
        elif time_aggregation == "Quarterly":
            visits_with_time['time_period'] = visits_with_time['date'].dt.to_period('Q')
        
        # Create time series aggregation
        # This is a simplified version - could be expanded
        time_series = visits_with_time.groupby('time_period').agg({
            'visit_id': 'count',
            'opportunity_id': 'nunique' if 'opportunity_id' in visits_with_time.columns else lambda x: 0,
            'flw_id': 'nunique' if 'flw_id' in visits_with_time.columns else lambda x: 0
        }).reset_index()
        
        time_series = time_series.rename(columns={'visit_id': 'total_visits'})
        
        return time_series
    
    def _create_excel_output(self, combined_results, time_series_data, visits_data, output_dir, resolutions):
        """Create Excel output with two tabs: grid cells and opportunity summary"""
        
        # Create dynamic filename based on inputs
        grid3_filename = os.path.basename(self.get_parameter_value('grid3_file', ''))
        grid3_prefix = grid3_filename[:3] if grid3_filename else 'grid'
        
        opp_count = visits_data['opportunity_id'].nunique() if 'opportunity_id' in visits_data.columns else 0
        visit_count = len(visits_data)
        
        excel_filename = f"grid3_{grid3_prefix}_{opp_count}opps_{visit_count}visits.xlsx"
        excel_file = os.path.join(output_dir, excel_filename)
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # TWO TABS: Grid cells and opportunity summary
            if combined_results.empty:
                # Create empty DataFrames for both tabs
                empty_grid_df = pd.DataFrame({
                    'cell_id': [], 'opportunity_id': [], 'opportunity_name': [], 'resolution_m': [],
                    'center_latitude': [], 'center_longitude': [], 'population': [],
                    'population_density_per_km2': [], 'total_visits': [], 'visits_per_capita': [],
                    'density_category': []
                })
                empty_opp_df = pd.DataFrame({
                    'opportunity_id': [], 'opportunity_name': [], 'total_visits': [], 
                    'total_grids': [], 'total_population': [], 'visits_per_population': []
                })
                
                empty_grid_df.to_excel(writer, sheet_name='Grid_Cells', index=False)
                empty_opp_df.to_excel(writer, sheet_name='Opportunity_Summary', index=False)
            else:
                # Tab 1: Grid cell data (one row per grid cell per opportunity)
                grid_results = combined_results.copy()
                grid_results.to_excel(writer, sheet_name='Grid_Cells', index=False)
                
                # Tab 2: Opportunity summary (100m data only)
                opp_summary = self._create_opportunity_summary(combined_results)
                opp_summary.to_excel(writer, sheet_name='Opportunity_Summary', index=False)
            
            # COMPLEX MULTI-SHEET OUTPUT (disabled)
            if (False):  # MULTIPLE_SHEETS
                # Summary sheet
                summary_data = self._create_summary_data(combined_results, resolutions)
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual resolution sheets
                for resolution in resolutions:
                    resolution_data = combined_results[combined_results['resolution_m'] == resolution]
                    if not resolution_data.empty:
                        sheet_name = f'Grid_{resolution}m'
                        resolution_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Time series sheet
                if time_series_data is not None:
                    time_series_data.to_excel(writer, sheet_name='Time_Series', index=False)
                
                # Visit details sheet
                visits_data.to_excel(writer, sheet_name='Visit_Details', index=False)
                
                # Methodology sheet
                methodology = self._create_methodology_sheet(resolutions)
                methodology.to_excel(writer, sheet_name='Methodology', index=False)
        
        self.log(f"Created Excel analysis with two tabs: {os.path.basename(excel_file)}")
        return excel_file
    
    def _create_summary_data(self, combined_results, resolutions):
        """Create summary statistics across all resolutions"""
        
        summary_data = []
        
        for resolution in resolutions:
            resolution_data = combined_results[combined_results['resolution_m'] == resolution]
            
            if resolution_data.empty:
                continue
            
            summary_data.append({
                'resolution_m': resolution,
                'grid_cells_with_visits': len(resolution_data),
                'total_visits': resolution_data['total_visits'].sum(),
                'total_population': resolution_data['population'].sum(),
                'unique_opportunities': resolution_data.get('opportunity_id_nunique', pd.Series([0])).sum(),
                'unique_flws': resolution_data.get('flw_id_nunique', pd.Series([0])).sum(),
                'avg_visits_per_cell': resolution_data['total_visits'].mean(),
                'avg_population_per_cell': resolution_data['population'].mean(),
                'avg_visits_per_capita': resolution_data['visits_per_capita'].mean(),
                'high_density_cells': len(resolution_data[resolution_data.get('density_category', '') == 'High']),
                'medium_density_cells': len(resolution_data[resolution_data.get('density_category', '') == 'Medium']),
                'low_density_cells': len(resolution_data[resolution_data.get('density_category', '') == 'Low'])
            })

    def _create_opportunity_summary(self, combined_results):
        """Create opportunity-level summary with separate rows for each opportunity and resolution"""
        
        if combined_results.empty:
            return pd.DataFrame({
                'opportunity_id': [], 'opportunity_name': [], 'resolution_m': [], 'total_visits': [], 
                'total_grids': [], 'total_population': [], 'visits_per_population': [],
                'low_density_visits': [], 'low_density_grids': [], 'low_density_pop': [], 'low_density_visits_per_pop': [],
                'medium_density_visits': [], 'medium_density_grids': [], 'medium_density_pop': [], 'medium_density_visits_per_pop': [],
                'high_density_visits': [], 'high_density_grids': [], 'high_density_pop': [], 'high_density_visits_per_pop': []
            })
        
        # Check if we have opportunity information
        if 'opportunity_id' not in combined_results.columns:
            self.log(f"  No opportunity_id found in data")
            return pd.DataFrame({
                'total_visits': [combined_results['total_visits'].sum()],
                'total_grids': [len(combined_results)],
                'total_population': [combined_results['population'].sum()]
            })
        
        # Group by opportunity and resolution, then calculate summary metrics
        opp_summary = []
        
        for opp_id in combined_results['opportunity_id'].unique():
            opp_data = combined_results[combined_results['opportunity_id'] == opp_id]
            opp_name = opp_data['opportunity_name'].iloc[0] if 'opportunity_name' in opp_data.columns else f"Opportunity {opp_id}"
            
            # Process each resolution separately
            for resolution in opp_data['resolution_m'].unique():
                resolution_data = opp_data[opp_data['resolution_m'] == resolution]
                
                # Overall metrics for this opportunity at this resolution
                total_visits = resolution_data['total_visits'].sum()
                total_grids = len(resolution_data)
                total_population = resolution_data['population'].sum()
                visits_per_population = total_visits / total_population if total_population > 0 else 0
                
                # Density breakdown metrics
                low_density = resolution_data[resolution_data['density_category'] == 'Low']
                medium_density = resolution_data[resolution_data['density_category'] == 'Medium']
                high_density = resolution_data[resolution_data['density_category'] == 'High']
                
                # Low density metrics
                low_visits = low_density['total_visits'].sum()
                low_grids = len(low_density)
                low_pop = low_density['population'].sum()
                low_visits_per_pop = low_visits / low_pop if low_pop > 0 else 0
                
                # Medium density metrics
                med_visits = medium_density['total_visits'].sum()
                med_grids = len(medium_density)
                med_pop = medium_density['population'].sum()
                med_visits_per_pop = med_visits / med_pop if med_pop > 0 else 0
                
                # High density metrics
                high_visits = high_density['total_visits'].sum()
                high_grids = len(high_density)
                high_pop = high_density['population'].sum()
                high_visits_per_pop = high_visits / high_pop if high_pop > 0 else 0
                
                opp_summary.append({
                    'opportunity_id': opp_id,
                    'opportunity_name': opp_name,
                    'resolution_m': resolution,
                    'total_visits': total_visits,
                    'total_grids': total_grids,
                    'total_population': total_population,
                    'visits_per_population': visits_per_population,
                    'low_density_visits': low_visits,
                    'low_density_grids': low_grids,
                    'low_density_pop': low_pop,
                    'low_density_visits_per_pop': low_visits_per_pop,
                    'medium_density_visits': med_visits,
                    'medium_density_grids': med_grids,
                    'medium_density_pop': med_pop,
                    'medium_density_visits_per_pop': med_visits_per_pop,
                    'high_density_visits': high_visits,
                    'high_density_grids': high_grids,
                    'high_density_pop': high_pop,
                    'high_density_visits_per_pop': high_visits_per_pop
                })
        
        self.log(f"  Created opportunity summary for {len(opp_summary)} opportunity-resolution combinations")
        return pd.DataFrame(opp_summary)

