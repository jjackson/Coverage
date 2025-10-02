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
        buffer_distance_var = tk.StringVar(value="5")
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
            'opp_id': ['opp_id', 'oppid', 'opportunity_id', 'campaign_id'],
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
        for opt_col in ['opp_id', 'flw_id', 'date']:
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
    
    def _resample_grid_data(self, src, target_resolution, base_resolution):
        """Resample grid data to different resolution"""
        
        # Calculate resampling factor
        factor = target_resolution / base_resolution
        
        if factor < 1:
            # Upsampling (making pixels smaller) - not typically needed
            resampling_method = Resampling.bilinear
        else:
            # Downsampling (making pixels larger) - aggregate population
            resampling_method = Resampling.sum
        
        # Calculate new dimensions
        new_width = int(src.width / factor)
        new_height = int(src.height / factor)
        
        # Create new transform
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Resample the data
        resampled_data = np.empty((new_height, new_width), dtype=src.dtypes[0])
        
        reproject(
            source=src.read(1),
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=new_transform,
            dst_crs=src.crs,
            resampling=resampling_method
        )
        
        # Convert to grid cells
        mask = resampled_data > 0
        shapes_iter = shapes(resampled_data, mask=mask, transform=new_transform)
        
        grid_cells = []
        for i, (geom, pop_value) in enumerate(shapes_iter):
            if pop_value > 0:
                grid_cells.append({
                    'cell_id': f"cell_{target_resolution}m_{i}",
                    'population': float(pop_value),
                    'geometry': geom
                })
        
        if grid_cells:
            gdf = gpd.GeoDataFrame(grid_cells, crs=src.crs)
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Add centroids
            centroids = gdf.geometry.centroid
            gdf['center_latitude'] = centroids.y
            gdf['center_longitude'] = centroids.x
            
            return gdf
        else:
            # Return empty GeoDataFrame with proper structure
            from shapely.geometry import Point
            empty_geom = [Point(0, 0)]  # Dummy geometry
            empty_df = gpd.GeoDataFrame({'cell_id': [''], 'population': [0], 'center_latitude': [0], 'center_longitude': [0]}, 
                                      geometry=empty_geom, crs='EPSG:4326')
            return empty_df.iloc[0:0]  # Return empty but properly structured GDF

    def _analyze_visits_by_grid(self, visits_data, pop_grid, buffer_distance, resolution):
        """Analyze visits against population grid cells - Fixed date handling"""
        
        if len(pop_grid) == 0:
            self.log(f"  Warning: No population grid cells found for {resolution}m resolution")
            return pd.DataFrame()
        
        self.log(f"  Creating visit points from {len(visits_data)} visits...")
        
        # Create GeoDataFrame of visit points
        geometry = [Point(xy) for xy in zip(visits_data['longitude'], visits_data['latitude'])]
        visits_gdf = gpd.GeoDataFrame(visits_data, geometry=geometry, crs='EPSG:4326')
        
        # Apply buffer to visit points if specified
        if buffer_distance > 0:
            self.log(f"  Applying {buffer_distance}m buffer to visit points...")
            # Convert to projected CRS for accurate buffer
            visits_projected = visits_gdf.to_crs('EPSG:3857')  # Web Mercator
            visits_buffered = visits_projected.copy()
            visits_buffered.geometry = visits_projected.geometry.buffer(buffer_distance)
            visits_buffered = visits_buffered.to_crs('EPSG:4326')  # Back to WGS84
            
            # Use buffered geometry for spatial join
            join_data = visits_buffered
        else:
            join_data = visits_gdf
        
        self.log(f"  Performing spatial join between {len(pop_grid)} grid cells and {len(join_data)} visit points...")
        
        # Spatial join visits to grid cells
        joined = gpd.sjoin(pop_grid, join_data, how='inner', predicate='intersects')
        
        if len(joined) == 0:
            self.log(f"  Warning: No spatial intersection found between visits and {resolution}m grid")
            return pd.DataFrame()
        
        self.log(f"  Found {len(joined)} visit-grid intersections")
        
        # Check what columns we actually have for aggregation
        available_columns = list(joined.columns)
        self.log(f"  Available columns for aggregation: {available_columns[:10]}...")  # Show first 10
        
        # Build aggregation functions based on available columns
        agg_functions = {}
        
        # Count visits - use visit_id if available, otherwise use index
        if 'visit_id' in joined.columns:
            agg_functions['visit_id'] = 'count'
            visit_count_col = 'visit_id'
        else:
            # Create a temporary column for counting
            joined['temp_visit_count'] = 1
            agg_functions['temp_visit_count'] = 'sum'
            visit_count_col = 'temp_visit_count'
        
        # COMPLEX AGGREGATIONS (disabled)
        if (False):  # COMPLEX_AGGREGATIONS
            # Add optional aggregations if columns exist
            if 'opp_id' in joined.columns:
                agg_functions['opp_id'] = 'nunique'
            if 'flw_id' in joined.columns:
                agg_functions['flw_id'] = 'nunique'
                
            # Only add date aggregation if date column exists AND contains valid datetime data
            if 'date' in joined.columns:
                try:
                    # Test if date column has datetime-like values
                    if pd.api.types.is_datetime64_any_dtype(joined['date']):
                        agg_functions['date'] = ['min', 'max']
                        self.log(f"  Including date aggregation (valid datetime column)")
                    else:
                        self.log(f"  Skipping date aggregation (not datetime type: {joined['date'].dtype})")
                except Exception as e:
                    self.log(f"  Skipping date aggregation due to error: {str(e)}")
        
        self.log(f"  Aggregating by grid cell using functions: {list(agg_functions.keys())}")
        
        # Group by grid cell and aggregate
        groupby_cols = ['cell_id', 'population', 'center_latitude', 'center_longitude']
        grid_stats = joined.groupby(groupby_cols).agg(agg_functions).reset_index()
        
        self.log(f"  Aggregation complete: {len(grid_stats)} grid cells with visits")
        
        # SIMPLIFIED COLUMN HANDLING (since we're only keeping essential columns)
        # Just ensure we have the basic columns we need
        if 'total_visits' not in grid_stats.columns:
            if visit_count_col in grid_stats.columns:
                grid_stats['total_visits'] = grid_stats[visit_count_col]
            else:
                # Fallback: count number of rows per group
                grid_stats['total_visits'] = 1
        
        # COMPLEX COLUMN FLATTENING (disabled)
        if (False):  # COMPLEX_COLUMN_FLATTENING
            # Flatten column names
            new_columns = ['cell_id', 'population', 'center_latitude', 'center_longitude']
            
            for col in grid_stats.columns[4:]:  # Skip the first 4 groupby columns
                if isinstance(col, tuple):
                    if col[1] in ['min', 'max']:
                        new_columns.append(f"{col[0]}_{col[1]}")
                    elif col[1] == 'count' or col[1] == 'sum':
                        new_columns.append('total_visits')
                    elif col[1] == 'nunique':
                        new_columns.append(f"{col[0]}_unique")
                    else:
                        new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    # Handle single-level columns
                    if col == visit_count_col:
                        new_columns.append('total_visits')
                    else:
                        new_columns.append(col)
            
            grid_stats.columns = new_columns
        
        # Calculate derived metrics
        grid_stats['visits_per_capita'] = grid_stats['total_visits'] / grid_stats['population']
        grid_stats['population_density_per_km2'] = grid_stats['population'] / ((resolution/1000) ** 2)
        
        self.log(f"  Analysis complete: {len(grid_stats)} grid cells with visits")
        self.log(f"  Total visits processed: {grid_stats['total_visits'].sum()}")
        self.log(f"  Visit range per cell: {grid_stats['total_visits'].min()} to {grid_stats['total_visits'].max()}")
        
        return grid_stats
    
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
            'opp_id': 'nunique' if 'opp_id' in visits_with_time.columns else lambda x: 0,
            'flw_id': 'nunique' if 'flw_id' in visits_with_time.columns else lambda x: 0
        }).reset_index()
        
        time_series = time_series.rename(columns={'visit_id': 'total_visits'})
        
        return time_series
    
    def _create_excel_output(self, combined_results, time_series_data, visits_data, output_dir, resolutions):
        """Create simplified Excel output with essential grid cell data"""
        
        excel_file = os.path.join(output_dir, "grid3_population_analysis.xlsx")
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # SIMPLIFIED OUTPUT: Just one sheet with essential columns
            if combined_results.empty:
                # Create empty DataFrame with proper structure
                empty_df = pd.DataFrame({
                    'cell_id': [],
                    'resolution_m': [],
                    'center_latitude': [],
                    'center_longitude': [],
                    'population': [],
                    'population_density_per_km2': [],
                    'total_visits': [],
                    'visits_per_capita': [],
                    'density_category': []
                })
                empty_df.to_excel(writer, sheet_name='Grid_Cells', index=False)
            else:
                # Select only essential columns for simplified output
                essential_columns = [
                    'cell_id', 'resolution_m', 'center_latitude', 'center_longitude',
                    'population', 'population_density_per_km2', 'total_visits', 
                    'visits_per_capita', 'density_category'
                ]
                
                # Filter to only include columns that exist
                available_columns = [col for col in essential_columns if col in combined_results.columns]
                simplified_results = combined_results[available_columns]
                
                simplified_results.to_excel(writer, sheet_name='Grid_Cells', index=False)
            
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
        
        self.log(f"Created simplified Excel analysis: {os.path.basename(excel_file)}")
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
                'unique_opportunities': resolution_data.get('opp_id_nunique', pd.Series([0])).sum(),
                'unique_flws': resolution_data.get('flw_id_nunique', pd.Series([0])).sum(),
                'avg_visits_per_cell': resolution_data['total_visits'].mean(),
                'avg_population_per_cell': resolution_data['population'].mean(),
                'avg_visits_per_capita': resolution_data['visits_per_capita'].mean(),
                'high_density_cells': len(resolution_data[resolution_data.get('density_category', '') == 'High']),
                'medium_density_cells': len(resolution_data[resolution_data.get('density_category', '') == 'Medium']),
                'low_density_cells': len(resolution_data[resolution_data.get('density_category', '') == 'Low'])
            })
