"""
Visit Clustering Report

Generates geographic clusters from visit data for all OPP IDs.
Creates interactive maps and exports cluster assignments.
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, pi
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import BallTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd
try:
    import alphashape
    ALPHASHAPE_AVAILABLE = True
except ImportError:
    ALPHASHAPE_AVAILABLE = False
from .base_report import BaseReport


class VisitClusteringReport(BaseReport):
    """Report that generates geographic clusters from visit data"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for visit clustering report"""
        
        # Number of clusters
        ttk.Label(parent_frame, text="Number of clusters:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        n_clusters_var = tk.StringVar(value="5")
        ttk.Entry(parent_frame, textvariable=n_clusters_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Min points per cluster (for validation)
        ttk.Label(parent_frame, text="Min points per cluster:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        min_points_var = tk.StringVar(value="3")
        ttk.Entry(parent_frame, textvariable=min_points_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Auto-determine clusters option
        ttk.Label(parent_frame, text="Auto-determine clusters:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        auto_clusters_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent_frame, variable=auto_clusters_var, 
                       text="Use silhouette analysis (ignores fixed number)").grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Algorithm choice
        ttk.Label(parent_frame, text="Clustering algorithm:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        algorithm_var = tk.StringVar(value="K-means")
        algorithm_frame = ttk.Frame(parent_frame)
        algorithm_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(algorithm_frame, text="K-means + outliers", variable=algorithm_var, value="K-means").pack(side=tk.LEFT)
        ttk.Radiobutton(algorithm_frame, text="Simple distance", variable=algorithm_var, value="Simple").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(algorithm_frame, text="Cluster growth", variable=algorithm_var, value="Growth").pack(side=tk.LEFT, padx=(10,0))
        
        # Buffer distance for cluster polygons
        ttk.Label(parent_frame, text="Cluster polygon buffer (meters):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        buffer_distance_var = tk.StringVar(value="25")
        ttk.Entry(parent_frame, textvariable=buffer_distance_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Outlier detection distance
        ttk.Label(parent_frame, text="Outlier detection distance (meters):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        outlier_distance_var = tk.StringVar(value="500")
        ttk.Entry(parent_frame, textvariable=outlier_distance_var, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Polygon type selection
        ttk.Label(parent_frame, text="Polygon type:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        polygon_type_var = tk.StringVar(value="Convex hull")
        polygon_frame = ttk.Frame(parent_frame)
        polygon_frame.grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(polygon_frame, text="Convex hull", variable=polygon_type_var, value="Convex hull").pack(side=tk.LEFT)
        if ALPHASHAPE_AVAILABLE:
            ttk.Radiobutton(polygon_frame, text="Alpha shape", variable=polygon_type_var, value="Alpha shape").pack(side=tk.LEFT, padx=(10,0))
        else:
            ttk.Label(polygon_frame, text="(Alpha shape: install alphashape package)", foreground="gray").pack(side=tk.LEFT, padx=(10,0))
        
        # Alpha parameter (only relevant for alpha shapes)
        ttk.Label(parent_frame, text="Alpha parameter (0=auto):").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        alpha_param_var = tk.StringVar(value="0")
        ttk.Entry(parent_frame, textvariable=alpha_param_var, width=10).grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Store variables
        parent_frame.n_clusters_var = n_clusters_var
        parent_frame.min_points_var = min_points_var
        parent_frame.auto_clusters_var = auto_clusters_var
        parent_frame.algorithm_var = algorithm_var
        parent_frame.buffer_distance_var = buffer_distance_var
        parent_frame.outlier_distance_var = outlier_distance_var
        parent_frame.polygon_type_var = polygon_type_var
        parent_frame.alpha_param_var = alpha_param_var
        
    def generate(self):
        """Generate visit clustering reports for all OPP IDs"""
        output_files = []
        
        # Get parameters
        n_clusters = int(self.get_parameter_value('n_clusters', '5'))
        min_points = int(self.get_parameter_value('min_points', '3'))
        auto_clusters = self.get_parameter_value('auto_clusters', True)
        algorithm = self.get_parameter_value('algorithm', 'K-means')
        buffer_distance = float(self.get_parameter_value('buffer_distance', '25'))
        outlier_distance = float(self.get_parameter_value('outlier_distance', '500'))
        polygon_type = self.get_parameter_value('polygon_type', 'Convex hull')
        alpha_param = float(self.get_parameter_value('alpha_param', '0'))
        
        self.log(f"Starting visit clustering analysis for all OPP IDs")
        self.log(f"Algorithm: {algorithm}")
        self.log(f"Polygon type: {polygon_type}")
        if polygon_type == "Alpha shape" and alpha_param > 0:
            self.log(f"Alpha parameter: {alpha_param}")
        self.log(f"Parameters: n_clusters={n_clusters}, min_points={min_points}, auto_clusters={auto_clusters}")
        self.log(f"Buffer: {buffer_distance}m, Outlier threshold: {outlier_distance}m")
        
        # Create output directory with today's date
        today = datetime.now().strftime('%Y_%m_%d')
        cluster_dir = os.path.join(self.output_dir, f"visit_clusters_{today}")
        os.makedirs(cluster_dir, exist_ok=True)
        self.log(f"Created output directory: {os.path.basename(cluster_dir)}")
        
        # Validate and prepare data
        visits_data = self._prepare_visit_data()
        
        if len(visits_data) == 0:
            raise ValueError("No valid visit data found")
        
        # Get unique OPP IDs
        opp_ids = visits_data['opp_id'].unique()
        self.log(f"Found {len(opp_ids)} unique OPP IDs: {list(opp_ids)}")
        
        # Collect cluster data for details file
        all_cluster_data = []
        
        # Process each OPP ID
        for current_opp_id in opp_ids:
            self.log(f"Processing OPP ID: {current_opp_id}")
            
            # Filter data for this OPP ID
            opp_data = visits_data[visits_data['opp_id'] == current_opp_id].copy()
            
            if len(opp_data) < min_points:
                self.log(f"  Skipping {current_opp_id}: only {len(opp_data)} visits (< {min_points} minimum)")
                continue
                
            self.log(f"  Found {len(opp_data)} visits for clustering")
            
            # Get opportunity name for file naming
            opp_name_prefix = self._get_opportunity_name_prefix(opp_data, current_opp_id)
            
            # Perform clustering based on selected algorithm
            if algorithm == "K-means":
                clustered_data = self._cluster_kmeans_with_outliers(opp_data, n_clusters, auto_clusters, outlier_distance)
            elif algorithm == "Simple":
                clustered_data = self._cluster_simple_distance(opp_data, outlier_distance, min_points)
            else:  # Growth
                clustered_data = self._cluster_growth(opp_data, outlier_distance, min_points)
            
            # Generate outputs for this OPP ID
            opp_files = self._generate_single_opp_outputs(clustered_data, current_opp_id, cluster_dir, auto_clusters, buffer_distance, opp_name_prefix, outlier_distance, algorithm, polygon_type, alpha_param)
            output_files.extend(opp_files)
            
            # Collect cluster data for details file
            cluster_data_for_details = self._extract_cluster_details(clustered_data, current_opp_id, opp_name_prefix, algorithm, buffer_distance, outlier_distance, polygon_type, alpha_param)
            all_cluster_data.extend(cluster_data_for_details)
        
        # Create cluster details file from collected data
        if all_cluster_data:
            cluster_details_file = self._save_cluster_details_file(all_cluster_data, cluster_dir, algorithm, buffer_distance, outlier_distance, polygon_type, alpha_param)
            if cluster_details_file:
                output_files.append(cluster_details_file)
        
        # Create summary across all OPP IDs
        summary_file = self._create_summary_report(visits_data, cluster_dir)
        if summary_file:
            output_files.append(summary_file)
        
        self.log(f"Visit clustering complete! Generated {len(output_files)} files in {os.path.basename(cluster_dir)}")
        
        return output_files
    
    def _prepare_visit_data(self):
        """Prepare and validate visit data for clustering (all OPP IDs)"""
        
        # Make a copy of the data
        data = self.df.copy()
        
        # Standardize column names (case-insensitive)
        data.columns = data.columns.str.lower().str.strip()
        
        # Check for required columns
        required_cols = ['opp_id', 'latitude', 'longitude']
        missing_cols = []
        
        # Try different possible column name variations
        col_variations = {
            'opp_id': ['opp_id', 'oppid', 'opportunity_id', 'campaign_id'],
            'latitude': ['latitude', 'lat', 'y'],
            'longitude': ['longitude', 'lon', 'lng', 'x']
        }
        
        final_cols = {}
        for req_col in required_cols:
            found = False
            for variation in col_variations[req_col]:
                if variation in data.columns:
                    final_cols[req_col] = variation
                    found = True
                    break
            if not found:
                missing_cols.append(req_col)
        
        if missing_cols:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {available_cols}")
        
        # Rename columns to standard names
        data = data.rename(columns={v: k for k, v in final_cols.items()})
        
        self.log(f"Processing {len(data)} total visit records")
        
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
            self.log(f"Warning: Found {invalid_count} records with invalid coordinates")
            data = data[~invalid_coords]
        
        # Add unique visit ID if not present
        if 'visit_id' not in data.columns:
            data['visit_id'] = [f"visit_{i+1}" for i in range(len(data))]
        
        return data.reset_index(drop=True)
    
    def _get_opportunity_name_prefix(self, opp_data, opp_id):
        """Extract the first word of opportunity name for file naming"""
        
        # Look for opportunity name columns
        name_columns = [col for col in opp_data.columns if 'opportunity' in col.lower() and 'name' in col.lower()]
        if not name_columns:
            name_columns = [col for col in opp_data.columns if 'opp' in col.lower() and 'name' in col.lower()]
        if not name_columns:
            name_columns = [col for col in opp_data.columns if 'campaign' in col.lower() and 'name' in col.lower()]
        
        # If we found a name column, extract the first word
        if name_columns:
            name_col = name_columns[0]
            # Get the first non-null opportunity name
            opp_names = opp_data[name_col].dropna()
            if len(opp_names) > 0:
                full_name = str(opp_names.iloc[0]).strip()
                # Get first word (until first space)
                first_word = full_name.split()[0] if full_name else f"opp{opp_id}"
                # Clean the first word for use in filename
                first_word = ''.join(c for c in first_word if c.isalnum() or c in '-_')
                return first_word
        
        # Fallback if no opportunity name found
        return f"opp{opp_id}"
    
    def _detect_outliers(self, coords, outlier_distance_m):
        """Detect outlier points using spatial indexing for efficiency"""
        
        if len(coords) < 2:
            return np.array([False] * len(coords))
        
        # Convert coordinates to radians for BallTree (uses haversine distance)
        coords_rad = np.radians(coords)
        
        # Build spatial index using BallTree with haversine metric
        tree = BallTree(coords_rad, metric='haversine')
        
        # Convert distance from meters to radians (Earth's radius = 6371000m)
        outlier_distance_rad = outlier_distance_m / 6371000.0
        
        # For each point, find distance to nearest neighbor
        # k=2 because first neighbor is the point itself
        distances, indices = tree.query(coords_rad, k=2)
        
        # distances[:, 1] contains distance to nearest neighbor (not self)
        nearest_neighbor_distances = distances[:, 1] * 6371000.0  # Convert back to meters
        
        # Points are outliers if nearest neighbor is farther than threshold
        is_outlier = nearest_neighbor_distances > outlier_distance_m
        
        return is_outlier
    
    def _cluster_kmeans_with_outliers(self, data, n_clusters, auto_clusters, outlier_distance_m):
        """Perform K-means clustering with outlier detection"""
        
        self.log(f"  Running outlier detection + K-means clustering...")
        
        # Prepare coordinates
        coords = data[['latitude', 'longitude']].values
        
        # Step 1: Detect outliers
        is_outlier = self._detect_outliers(coords, outlier_distance_m)
        n_outliers = is_outlier.sum()
        n_regular = len(coords) - n_outliers
        
        self.log(f"  Found {n_outliers} outliers (>{outlier_distance_m}m from nearest neighbor)")
        self.log(f"  Clustering {n_regular} regular points")
        
        # Initialize result
        result = data.copy()
        
        if n_regular < 2:
            # If too few non-outlier points, put everything in outliers
            result['cluster_id'] = 'outliers'
            self.log(f"  Too few non-outlier points, assigning all to outliers cluster")
            return result
        
        # Step 2: Cluster non-outlier points
        regular_coords = coords[~is_outlier]
        
        # Determine optimal number of clusters for non-outliers
        if auto_clusters:
            optimal_k = self._find_optimal_clusters(regular_coords, max_k=min(10, len(regular_coords)-1))
            actual_clusters = optimal_k
            self.log(f"  Auto-determined optimal clusters: {actual_clusters}")
        else:
            actual_clusters = min(n_clusters, len(regular_coords))
            self.log(f"  Using fixed number of clusters: {actual_clusters}")
        
        if actual_clusters < 1:
            actual_clusters = 1
        
        # Step 3: Run K-means on non-outlier points
        if actual_clusters == 1:
            # Single cluster for all non-outliers
            cluster_labels = np.zeros(len(regular_coords), dtype=int)
        else:
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(regular_coords)
            
            # Calculate silhouette score
            if actual_clusters > 1 and len(regular_coords) > actual_clusters:
                try:
                    silhouette_avg = silhouette_score(regular_coords, cluster_labels)
                    self.log(f"  Silhouette score: {silhouette_avg:.3f}")
                except:
                    pass
        
        # Step 4: Assign cluster IDs
        result['cluster_id'] = 'outliers'  # Default all to outliers
        
        # Assign regular clusters
        regular_indices = result.index[~is_outlier]
        for i, idx in enumerate(regular_indices):
            result.loc[idx, 'cluster_id'] = f"cluster_{cluster_labels[i] + 1}"
        
        # Count results
        cluster_counts = result['cluster_id'].value_counts()
        self.log(f"  Results: {actual_clusters} clusters + {n_outliers} outliers")
        for cluster_id, count in cluster_counts.items():
            self.log(f"    {cluster_id}: {count} points")
        
        return result
    
    def _cluster_simple_distance(self, data, max_distance_m, min_points):
        """Perform simple distance-based clustering"""
        
        self.log(f"  Running simple distance clustering...")
        
        coords = data[['latitude', 'longitude']].values
        n_points = len(coords)
        
        # Initialize cluster assignments
        cluster_labels = [-1] * n_points
        current_cluster = 0
        
        for i in range(n_points):
            if cluster_labels[i] != -1:  # Already assigned
                continue
            
            # Start new cluster
            cluster_points = [i]
            cluster_labels[i] = current_cluster
            
            # Find all points within distance
            for j in range(i + 1, n_points):
                if cluster_labels[j] != -1:  # Already assigned
                    continue
                
                # Calculate distance between points i and j
                dist = self._haversine_distance(
                    coords[i][1], coords[i][0],  # lon, lat
                    coords[j][1], coords[j][0]
                )
                
                if dist <= max_distance_m:
                    cluster_labels[j] = current_cluster
                    cluster_points.append(j)
            
            # Check if cluster meets minimum size requirement
            if len(cluster_points) < min_points:
                # Mark as outliers
                for point_idx in cluster_points:
                    cluster_labels[point_idx] = -1
            else:
                current_cluster += 1
        
        # Add results to data
        result = data.copy()
        result['cluster_id'] = cluster_labels
        
        # Convert labels to descriptive names
        result['cluster_id'] = result['cluster_id'].replace(-1, 'outliers')
        unique_clusters = [c for c in result['cluster_id'].unique() if c != 'outliers']
        cluster_mapping = {old_id: f"cluster_{i+1}" for i, old_id in enumerate(sorted(unique_clusters))}
        result['cluster_id'] = result['cluster_id'].replace(cluster_mapping)
        
        n_clusters = len(unique_clusters)
        n_outliers = (result['cluster_id'] == 'outliers').sum()
        
        self.log(f"  Simple distance results: {n_clusters} clusters, {n_outliers} outliers")
        
        return result
    
    def _cluster_growth(self, data, max_distance_m, min_points):
        """Perform cluster growth (connected components) clustering with spatial indexing"""
        
        self.log(f"  Running optimized cluster growth clustering...")
        
        coords = data[['latitude', 'longitude']].values
        n_points = len(coords)
        
        if n_points < 2:
            result = data.copy()
            result['cluster_id'] = 'outliers'
            return result
        
        # Convert coordinates to radians for BallTree
        coords_rad = np.radians(coords)
        
        # Convert distance from meters to radians
        max_distance_rad = max_distance_m / 6371000.0
        
        # Build spatial index and find all neighbors efficiently
        self.log(f"    Building spatial index for {n_points} points...")
        tree = BallTree(coords_rad, metric='haversine')
        
        # Query all neighbors within radius for each point
        neighbors_lists = tree.query_radius(coords_rad, r=max_distance_rad)
        
        # Convert to adjacency representation (remove self-references)
        neighbors = {}
        for i, neighbor_list in enumerate(neighbors_lists):
            # Remove self from neighbor list
            neighbors[i] = [j for j in neighbor_list if j != i]
        
        self.log(f"    Finding connected components using Union-Find...")
        
        # Use Union-Find for efficient connected components
        parent = list(range(n_points))
        rank = [0] * n_points
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
        
        # Union all connected points
        for point, neighbor_list in neighbors.items():
            for neighbor in neighbor_list:
                union(point, neighbor)
        
        # Group points by their root parent (connected component)
        components = {}
        for i in range(n_points):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        
        # Assign cluster labels
        cluster_labels = [-1] * n_points
        current_cluster = 0
        
        for component_points in components.values():
            if len(component_points) >= min_points:
                # Valid cluster
                for point_idx in component_points:
                    cluster_labels[point_idx] = current_cluster
                current_cluster += 1
            else:
                # Too small - mark as outliers
                for point_idx in component_points:
                    cluster_labels[point_idx] = -1
        
        # Add results to data
        result = data.copy()
        result['cluster_id'] = cluster_labels
        
        # Convert labels to descriptive names
        result['cluster_id'] = result['cluster_id'].replace(-1, 'outliers')
        unique_clusters = [c for c in result['cluster_id'].unique() if c != 'outliers']
        cluster_mapping = {old_id: f"cluster_{i+1}" for i, old_id in enumerate(sorted(unique_clusters))}
        result['cluster_id'] = result['cluster_id'].replace(cluster_mapping)
        
        n_clusters = len(unique_clusters)
        n_outliers = (result['cluster_id'] == 'outliers').sum()
        
        self.log(f"  Optimized cluster growth results: {n_clusters} clusters, {n_outliers} outliers")
        
        return result
    
    def _cluster_kmeans(self, data, n_clusters, auto_clusters):
        """Perform K-means clustering on visit data"""
        
        self.log(f"  Running K-means clustering...")
        
        # Prepare coordinates
        coords = data[['latitude', 'longitude']].values
        
        # Determine optimal number of clusters if auto mode is enabled
        if auto_clusters:
            optimal_k = self._find_optimal_clusters(coords, max_k=min(10, len(coords)-1))
            actual_clusters = optimal_k
            self.log(f"  Auto-determined optimal clusters: {actual_clusters}")
        else:
            # Ensure we don't have more clusters than data points
            actual_clusters = min(n_clusters, len(coords))
            self.log(f"  Using fixed number of clusters: {actual_clusters}")
        
        if actual_clusters < 2:
            self.log(f"  Warning: Using {actual_clusters} cluster (insufficient data for clustering)")
            # Create single cluster
            result = data.copy()
            result['cluster_id'] = 'cluster_1'
            return result
        
        # Run K-means
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Add cluster labels to data
        result = data.copy()
        result['cluster_id'] = [f"cluster_{i+1}" for i in cluster_labels]
        
        # Calculate silhouette score if we have enough clusters
        if actual_clusters > 1 and len(coords) > actual_clusters:
            try:
                silhouette_avg = silhouette_score(coords, cluster_labels)
                self.log(f"  Silhouette score: {silhouette_avg:.3f}")
            except:
                pass
        
        self.log(f"  K-means results: {actual_clusters} clusters")
        
        return result
    
    def _find_optimal_clusters(self, coords, max_k=10):
        """Find optimal number of clusters using silhouette analysis"""
        
        if len(coords) < 4:
            return 1
        
        max_k = min(max_k, len(coords) - 1)
        if max_k < 2:
            return 1
        
        best_score = -1
        best_k = 2
        
        # Test different numbers of clusters
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(coords)
                
                # Calculate silhouette score
                score = silhouette_score(coords, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception:
                # Skip if clustering fails for this k
                continue
        
        return best_k
    
    def _generate_single_opp_outputs(self, clustered_data, opp_id, cluster_dir, auto_clusters, buffer_distance, opp_name_prefix, outlier_distance, algorithm, polygon_type, alpha_param):
        """Generate output files for a specific OPP ID"""
        
        output_files = []
        
        # Create algorithm-specific file prefix with key parameters
        if algorithm == 'Growth':
            algorithm_prefix = f'growth_{int(outlier_distance)}m_{int(buffer_distance)}buf'
        else:
            algorithm_prefix = {
                'K-means': 'kmeans',
                'Simple': 'simpledist'
            }.get(algorithm, 'unknown')
        
        # Add polygon type to filename
        if polygon_type == "Alpha shape":
            if alpha_param > 0:
                polygon_suffix = f'_alpha{alpha_param:.1f}'.replace('.', 'p')
            else:
                polygon_suffix = '_alphaauto'
        else:
            polygon_suffix = '_convex'
        
        algorithm_prefix += polygon_suffix
        
        # Create both versions of the interactive map
        # Version 1: Full map with visits and counts
        full_map_file = self._create_cluster_map(clustered_data, opp_id, cluster_dir, auto_clusters, buffer_distance, opp_name_prefix, outlier_distance, algorithm, algorithm_prefix, polygon_type, show_visits=True)
        output_files.append(full_map_file)
        self.log(f"  Created: {os.path.basename(full_map_file)}")
        
        # Version 2: Clean map with only polygons (no visits or counts)
        clean_map_file = self._create_cluster_map(clustered_data, opp_id, cluster_dir, auto_clusters, buffer_distance, opp_name_prefix, outlier_distance, algorithm, algorithm_prefix, polygon_type, show_visits=False)
        output_files.append(clean_map_file)
        self.log(f"  Created: {os.path.basename(clean_map_file)}")
        
        return output_files

    def _create_cluster_map(self, clustered_data, opp_id, cluster_dir, auto_clusters, buffer_distance, opp_name_prefix, outlier_distance_m, algorithm, algorithm_prefix, polygon_type, show_visits=True):
        """Create interactive Leaflet map showing clusters with convex hull polygons
        
        Args:
            show_visits (bool): If True, shows visit points and counts. If False, shows only polygons.
        """
        
        # Generate colors for clusters (including outliers)
        cluster_ids = clustered_data['cluster_id'].unique()
        colors = self._generate_cluster_colors(len(cluster_ids))
        cluster_colors = dict(zip(cluster_ids, colors))
        
        # Make sure outliers get a distinct gray color
        if 'outliers' in cluster_colors:
            cluster_colors['outliers'] = '#888888'
        
        # Generate cluster polygons
        cluster_polygons = self._generate_cluster_polygons(clustered_data, buffer_distance, polygon_type)
        
        # Prepare data for map (only if showing visits)
        map_data = []
        if show_visits:
            for _, row in clustered_data.iterrows():
                map_data.append({
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'cluster_id': str(row['cluster_id']),
                    'visit_id': str(row.get('visit_id', 'Unknown')),
                    'color': cluster_colors[row['cluster_id']]
                })
        
        # Calculate map center
        center_lat = clustered_data['latitude'].mean()
        center_lon = clustered_data['longitude'].mean()
        
        # Create version-specific title and filename components
        if show_visits:
            version_title = "Visit Clusters (Full)"
            version_suffix = "_full"
            map_description = f"<strong>Clusters:</strong> {len(cluster_ids)} | "
        else:
            version_title = "Cluster Boundaries"
            version_suffix = "_clean"
            # Count only non-outlier clusters for clean version
            non_outlier_clusters = [c for c in cluster_ids if c != 'outliers']
            map_description = f"<strong>Cluster areas:</strong> {len(non_outlier_clusters)} | "
        
        # Generate HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{version_title} - OPP {opp_id}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS and JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, Helvetica, sans-serif;
        }}
        #map {{
            width: 100%;
            height: 90vh;
        }}
        .control-panel {{
            background: white;
            padding: 10px;
            height: 10vh;
            border-bottom: 1px solid #ccc;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .cluster-toggle {{
            display: inline-block;
            margin-right: 10px;
            cursor: pointer;
            padding: 5px 10px;
            border-radius: 3px;
            border: 1px solid #ddd;
            background: #f9f9f9;
        }}
        .cluster-toggle input {{
            margin-right: 5px;
        }}
        .color-swatch {{
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 5px;
            border: 1px solid #999;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div class="control-panel">
        <div>
            <strong>OPP ID:</strong> {opp_id} | 
            <strong>Algorithm:</strong> {algorithm} | 
            <strong>Polygon:</strong> {polygon_type} |
            {map_description}
            <strong>Buffer:</strong> {buffer_distance}m |
            <strong>Outlier threshold:</strong> {outlier_distance_m}m
        </div>
        <div>
            <strong>Toggle Clusters:</strong>
            <button onclick="toggleAllClusters(true)">Show All</button>
            <button onclick="toggleAllClusters(false)">Hide All</button>
        </div>
        <div id="cluster-toggles">
            <!-- Cluster toggles will be added here -->
        </div>
    </div>
    
    <div id="map"></div>
    
    <script>
        // Map data
        const visitData = {json.dumps(map_data)};
        const clusterColors = {json.dumps(cluster_colors)};
        const clusterPolygons = {json.dumps(cluster_polygons)};
        const showVisits = {json.dumps(show_visits)};
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 13);
        
        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);
        
        // Create layer groups for each cluster
        const clusterLayers = {{}};
        const clusterCounts = {{}};
        const polygonLayers = {{}};
        
        // Add cluster polygons first (so they appear behind points)
        Object.keys(clusterPolygons).forEach(clusterId => {{
            const polygonCoords = clusterPolygons[clusterId];
            if (polygonCoords && polygonCoords.length > 0) {{
                const polygon = L.polygon(polygonCoords, {{
                    color: clusterColors[clusterId],
                    fillColor: clusterColors[clusterId],
                    fillOpacity: showVisits ? 0.2 : 0.3,  // Slightly more opaque when no visits shown
                    weight: 2,
                    opacity: 0.8
                }});
                
                // Add popup with cluster info (but no visit count for clean version)
                if (showVisits) {{
                    polygon.bindPopup(`<strong>Cluster:</strong> ${{clusterId}}`);
                }} else {{
                    polygon.bindPopup(`<strong>Cluster:</strong> ${{clusterId}}<br><strong>Area boundary</strong>`);
                }}
                
                polygon.addTo(map);
                polygonLayers[clusterId] = polygon;
            }}
        }});
        
        // Only add visit points if showing visits
        if (showVisits) {{
            // Group visits by cluster
            visitData.forEach(visit => {{
                const clusterId = visit.cluster_id;
                
                if (!clusterLayers[clusterId]) {{
                    clusterLayers[clusterId] = L.layerGroup().addTo(map);
                    clusterCounts[clusterId] = 0;
                }}
                clusterCounts[clusterId]++;
                
                // Create marker
                const marker = L.circleMarker([visit.latitude, visit.longitude], {{
                    radius: 6,
                    fillColor: visit.color,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }});
                
                // Add popup
                marker.bindPopup(`
                    <strong>Visit ID:</strong> ${{visit.visit_id}}<br>
                    <strong>Cluster:</strong> ${{visit.cluster_id}}<br>
                    <strong>Coordinates:</strong> ${{visit.latitude.toFixed(6)}}, ${{visit.longitude.toFixed(6)}}
                `);
                
                clusterLayers[clusterId].addLayer(marker);
            }});
        }} else {{
            // For clean version, just initialize empty cluster counts for polygon-only clusters
            Object.keys(clusterPolygons).forEach(clusterId => {{
                clusterCounts[clusterId] = 0;  // No count shown in clean version
            }});
        }}
        
        // Create cluster toggles
        const togglesContainer = document.getElementById('cluster-toggles');
        Object.keys(clusterColors).sort().forEach(clusterId => {{
            // Skip outliers in clean version since they have no polygon
            if (!showVisits && clusterId === 'outliers') {{
                return;
            }}
            
            const label = document.createElement('label');
            label.className = 'cluster-toggle';
            
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = true;
            input.addEventListener('change', function() {{
                if (this.checked) {{
                    if (clusterLayers[clusterId]) {{
                        clusterLayers[clusterId].addTo(map);
                    }}
                    if (polygonLayers[clusterId]) {{
                        polygonLayers[clusterId].addTo(map);
                    }}
                }} else {{
                    if (clusterLayers[clusterId]) {{
                        map.removeLayer(clusterLayers[clusterId]);
                    }}
                    if (polygonLayers[clusterId]) {{
                        map.removeLayer(polygonLayers[clusterId]);
                    }}
                }}
            }});
            
            const colorSwatch = document.createElement('span');
            colorSwatch.className = 'color-swatch';
            colorSwatch.style.backgroundColor = clusterColors[clusterId];
            
            label.appendChild(input);
            label.appendChild(colorSwatch);
            
            // Show count only if showing visits, otherwise just cluster name
            if (showVisits && clusterCounts[clusterId] > 0) {{
                label.appendChild(document.createTextNode(`${{clusterId}} (${{clusterCounts[clusterId]}})`));
            }} else {{
                label.appendChild(document.createTextNode(clusterId));
            }}
            
            togglesContainer.appendChild(label);
        }});
        
        // Toggle all clusters function
        function toggleAllClusters(show) {{
            document.querySelectorAll('.cluster-toggle input').forEach(input => {{
                input.checked = show;
                const event = new Event('change');
                input.dispatchEvent(event);
            }});
        }}
        
        // Fit map to appropriate bounds
        if (showVisits && visitData.length > 0) {{
            // Fit to all visit points
            const group = new L.featureGroup(Object.values(clusterLayers));
            map.fitBounds(group.getBounds().pad(0.1));
        }} else if (Object.keys(clusterPolygons).length > 0) {{
            // Fit to polygon bounds
            const polygonGroup = new L.featureGroup(Object.values(polygonLayers));
            map.fitBounds(polygonGroup.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>"""
        
        # Save HTML file with version suffix
        output_filename = os.path.join(cluster_dir, f"visit_clusters_map_{algorithm_prefix}_{opp_name_prefix}_{opp_id}{version_suffix}.html")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_filename
    
    def _generate_cluster_statistics(self, clustered_data):
        """Generate summary statistics for clusters"""
        
        stats = []
        
        for cluster_id in clustered_data['cluster_id'].unique():
            cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
            
            # Calculate cluster center
            center_lat = cluster_data['latitude'].mean()
            center_lon = cluster_data['longitude'].mean()
            
            # Calculate bounding box
            min_lat, max_lat = cluster_data['latitude'].min(), cluster_data['latitude'].max()
            min_lon, max_lon = cluster_data['longitude'].min(), cluster_data['longitude'].max()
            
            # Convert bbox to meters (approximate)
            bbox_width_m = self._haversine_distance(min_lon, center_lat, max_lon, center_lat)
            bbox_height_m = self._haversine_distance(center_lon, min_lat, center_lon, max_lat)
            
            # Calculate maximum internal distance
            max_dist = 0
            coords = cluster_data[['latitude', 'longitude']].values
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = self._haversine_distance(
                        coords[i][1], coords[i][0],
                        coords[j][1], coords[j][0]
                    )
                    max_dist = max(max_dist, dist)
            
            stats.append({
                'cluster_id': cluster_id,
                'visit_count': len(cluster_data),
                'center_latitude': round(center_lat, 6),
                'center_longitude': round(center_lon, 6),
                'bbox_width_m': round(bbox_width_m, 2),
                'bbox_height_m': round(bbox_height_m, 2),
                'max_internal_distance_m': round(max_dist, 2)
            })
        
        return pd.DataFrame(stats).sort_values('cluster_id')
    
    def _generate_cluster_polygons(self, clustered_data, buffer_distance_m, polygon_type="Convex hull", alpha_param=0):
        """Generate polygons for each cluster using convex hull or alpha shapes"""
        
        cluster_polygons = {}
        
        for cluster_id in clustered_data['cluster_id'].unique():
            cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
            
            # Skip polygon generation for outliers cluster
            if cluster_id == 'outliers':
                continue
            
            if len(cluster_data) < 3:
                # Need at least 3 points for a polygon, skip if fewer
                continue
                
            try:
                # Create points for this cluster
                points = [Point(row['longitude'], row['latitude']) for _, row in cluster_data.iterrows()]
                
                # Create GeoDataFrame to handle coordinate system properly
                gdf = gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')
                
                # Convert to a projected coordinate system for accurate buffer calculation
                gdf_projected = gdf.to_crs('EPSG:3857')
                
                # Generate polygon based on type
                if polygon_type == "Alpha shape" and ALPHASHAPE_AVAILABLE:
                    # Use alpha shapes for more accurate boundaries
                    coords_2d = np.array([[geom.x, geom.y] for geom in gdf_projected.geometry])
                    
                    if alpha_param <= 0:
                        # Auto-determine alpha parameter
                        alpha_shape = alphashape.alphashape(coords_2d)
                    else:
                        # Use specified alpha parameter
                        alpha_shape = alphashape.alphashape(coords_2d, alpha_param)
                    
                    # Handle different return types from alphashape
                    if hasattr(alpha_shape, 'geom_type'):
                        if alpha_shape.geom_type == 'Polygon':
                            base_polygon = alpha_shape
                        elif alpha_shape.geom_type == 'MultiPolygon':
                            # Take the largest polygon if multiple
                            base_polygon = max(alpha_shape.geoms, key=lambda p: p.area)
                        else:
                            # Fallback to convex hull if alpha shape fails
                            all_points = unary_union(gdf_projected.geometry)
                            base_polygon = all_points.convex_hull
                    else:
                        # Fallback to convex hull if alpha shape returns invalid geometry
                        all_points = unary_union(gdf_projected.geometry)
                        base_polygon = all_points.convex_hull
                else:
                    # Use convex hull (default)
                    all_points = unary_union(gdf_projected.geometry)
                    base_polygon = all_points.convex_hull
                
                # Apply buffer
                buffered_polygon = base_polygon.buffer(buffer_distance_m)
                
                # Convert back to geographic coordinates
                buffered_gdf = gpd.GeoDataFrame([1], geometry=[buffered_polygon], crs='EPSG:3857')
                buffered_geographic = buffered_gdf.to_crs('EPSG:4326')
                
                # Extract coordinates for Leaflet (lat, lon format)
                polygon_geom = buffered_geographic.geometry.iloc[0]
                if polygon_geom.geom_type == 'Polygon':
                    # Get exterior coordinates and convert from (lon, lat) to [lat, lon]
                    coords = list(polygon_geom.exterior.coords)
                    leaflet_coords = [[lat, lon] for lon, lat in coords]
                    cluster_polygons[cluster_id] = leaflet_coords
                
            except Exception as e:
                # If polygon generation fails, skip this cluster
                poly_type_str = "alpha shape" if polygon_type == "Alpha shape" else "convex hull"
                self.log(f"  Warning: Could not generate {poly_type_str} for cluster {cluster_id}: {str(e)}")
                continue
        
        return cluster_polygons
    
    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        """Calculate haversine distance between two points in meters"""
        
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        
        return c * r
    
    def _generate_cluster_colors(self, n_clusters):
        """Generate distinct colors for clusters"""
        
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
            "#FF8000", "#FF0080", "#80FF00", "#00FF80", "#8000FF", "#0080FF",
            "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
            "#FF8080", "#80FF80", "#8080FF", "#FFFF80", "#FF80FF", "#80FFFF"
        ]
        
        if n_clusters <= len(colors):
            return colors[:n_clusters]
        else:
            # Generate additional random colors if needed
            import random
            additional_colors = []
            for i in range(n_clusters - len(colors)):
                color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                additional_colors.append(color)
            return colors + additional_colors
    
    def _create_summary_report(self, all_visits_data, cluster_dir):
        """Create a summary report across all OPP IDs"""
        
        try:
            summary_data = []
            
            # Get unique OPP IDs
            opp_ids = all_visits_data['opp_id'].unique()
            
            for opp_id in opp_ids:
                opp_data = all_visits_data[all_visits_data['opp_id'] == opp_id]
                
                # Get opportunity name prefix for this OPP ID
                opp_name_prefix = self._get_opportunity_name_prefix(opp_data, opp_id)
                
                # Try to load the clustering results for this OPP ID (only check for maps now since CSVs are not created)
                # We'll just track basic info for the summary
                summary_data.append({
                    'opp_id': opp_id,
                    'total_visits': len(opp_data),
                    'visits_clustered': 0,  # Not tracking individual files anymore
                    'n_clusters': 0,
                    'avg_cluster_size': 0,
                    'clustering_success': True,  # Assume success if we got here
                    'note': 'Individual CSVs not generated - see maps for results'
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = os.path.join(cluster_dir, "clustering_summary_all_opps.csv")
                summary_df.to_csv(summary_file, index=False)
                
                # Log summary statistics
                total_opps = len(summary_df)
                successful_opps = summary_df['clustering_success'].sum()
                total_visits = summary_df['total_visits'].sum()
                total_clusters = summary_df['n_clusters'].sum()
                
                self.log(f"Summary: {successful_opps}/{total_opps} OPP IDs successfully clustered")
                self.log(f"Total: {total_visits} visits ? {total_clusters} clusters")
                
                return summary_file
                
        except Exception as e:
            self.log(f"Error creating summary report: {str(e)}")
            
        return None
    
    def _extract_cluster_details(self, clustered_data, opp_id, opp_name_prefix, algorithm, buffer_distance, outlier_distance, polygon_type, alpha_param):
        """Extract cluster details from clustered data for this OPP ID"""
        
        cluster_details = []
        
        # Generate cluster polygons for this OPP's data
        cluster_polygons = self._generate_cluster_polygons(clustered_data, buffer_distance, polygon_type, alpha_param)
        
        # Process each cluster
        for cluster_id in clustered_data['cluster_id'].unique():
            cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
            
            # Calculate cluster statistics
            center_lat = cluster_data['latitude'].mean()
            center_lon = cluster_data['longitude'].mean()
            visit_count = len(cluster_data)
            
            # Calculate bounding box
            min_lat, max_lat = cluster_data['latitude'].min(), cluster_data['latitude'].max()
            min_lon, max_lon = cluster_data['longitude'].min(), cluster_data['longitude'].max()
            
            # Get polygon WKT if available
            polygon_wkt = None
            if cluster_id in cluster_polygons and cluster_id != 'outliers':
                # Convert Leaflet coordinates back to WKT
                coords = cluster_polygons[cluster_id]
                if coords and len(coords) > 2:
                    # Convert from [lat, lon] back to (lon, lat) for WKT
                    wkt_coords = [(coord[1], coord[0]) for coord in coords]
                    # Ensure polygon is closed
                    if wkt_coords[0] != wkt_coords[-1]:
                        wkt_coords.append(wkt_coords[0])
                    coord_pairs = [f"{lon} {lat}" for lon, lat in wkt_coords]
                    polygon_wkt = f"POLYGON (({', '.join(coord_pairs)}))"
            
            cluster_details.append({
                'opp_id': opp_id,
                'opp_name': opp_name_prefix,
                'cluster_id': cluster_id,
                'algorithm': algorithm,
                'polygon_type': polygon_type,
                'visit_count': visit_count,
                'center_latitude': round(center_lat, 6),
                'center_longitude': round(center_lon, 6),
                'bbox_min_lat': round(min_lat, 6),
                'bbox_max_lat': round(max_lat, 6),
                'bbox_min_lon': round(min_lon, 6),
                'bbox_max_lon': round(max_lon, 6),
                'polygon_wkt': polygon_wkt,
                'buffer_distance_m': buffer_distance,
                'outlier_distance_m': outlier_distance,
                'alpha_param': alpha_param if polygon_type == "Alpha shape" else None
            })
        
        return cluster_details
    
    def _save_cluster_details_file(self, all_cluster_data, cluster_dir, algorithm, buffer_distance, outlier_distance, polygon_type, alpha_param):
        """Save the collected cluster details to CSV file"""
        
        try:
            cluster_details_df = pd.DataFrame(all_cluster_data)
            
            # Sort by OPP ID and cluster ID
            cluster_details_df = cluster_details_df.sort_values(['opp_id', 'cluster_id'])
            
            # Create filename with parameters (matching map filename pattern)
            if algorithm == 'Growth':
                algorithm_suffix = f'growth_{int(outlier_distance)}m_{int(buffer_distance)}buf'
            else:
                algorithm_suffix = {
                    'K-means': 'kmeans',
                    'Simple': 'simpledist'
                }.get(algorithm, algorithm.lower())
            
            # Add polygon type to filename
            if polygon_type == "Alpha shape":
                if alpha_param > 0:
                    polygon_suffix = f'_alpha{alpha_param:.1f}'.replace('.', 'p')
                else:
                    polygon_suffix = '_alphaauto'
            else:
                polygon_suffix = '_convex'
            
            algorithm_suffix += polygon_suffix
            
            details_file = os.path.join(cluster_dir, f"cluster_details_{algorithm_suffix}.csv")
            cluster_details_df.to_csv(details_file, index=False)
            
            self.log(f"Created cluster details file: {os.path.basename(details_file)} ({len(cluster_details_df)} clusters)")
            return details_file
            
        except Exception as e:
            self.log(f"Error saving cluster details file: {str(e)}")
            return None
    
    def _create_cluster_details_file(self, all_visits_data, cluster_dir, algorithm, buffer_distance, outlier_distance, polygon_type, alpha_param):
        """Create a detailed cluster file with one row per cluster across all OPP IDs"""
        
        try:
            self.log("Creating cluster details file...")
            cluster_details = []
            
            # Look for map files to extract cluster information
            map_files = [f for f in os.listdir(cluster_dir) if f.startswith('visit_clusters_map_') and f.endswith('.html')]
            
            for map_file in map_files:
                # Extract OPP info from filename
                # Format: visit_clusters_map_{algorithm}_{opp_name}_{opp_id}.html
                parts = map_file.replace('visit_clusters_map_', '').replace('.html', '').split('_')
                if len(parts) >= 3:
                    opp_id = parts[-1]
                    opp_name = parts[-2]
                    
                    # Get data for this OPP ID
                    opp_data = all_visits_data[all_visits_data['opp_id'].astype(str) == str(opp_id)]
                    
                    if len(opp_data) > 0:
                        # Re-run clustering to get cluster assignments
                        if algorithm == "K-means":
                            # Get parameters from main function scope
                            n_clusters = int(self.get_parameter_value('n_clusters', '5'))
                            auto_clusters = self.get_parameter_value('auto_clusters', True)
                            outlier_distance = float(self.get_parameter_value('outlier_distance', '500'))
                            clustered_data = self._cluster_kmeans_with_outliers(opp_data, n_clusters, auto_clusters, outlier_distance)
                        elif algorithm == "Simple":
                            outlier_distance = float(self.get_parameter_value('outlier_distance', '500'))
                            min_points = int(self.get_parameter_value('min_points', '3'))
                            clustered_data = self._cluster_simple_distance(opp_data, outlier_distance, min_points)
                        else:  # Growth
                            outlier_distance = float(self.get_parameter_value('outlier_distance', '500'))
                            min_points = int(self.get_parameter_value('min_points', '3'))
                            clustered_data = self._cluster_growth(opp_data, outlier_distance, min_points)
                        
                        # Generate cluster polygons
                        cluster_polygons = self._generate_cluster_polygons(clustered_data, buffer_distance, polygon_type, alpha_param)
                        
                        # Process each cluster
                        for cluster_id in clustered_data['cluster_id'].unique():
                            cluster_data = clustered_data[clustered_data['cluster_id'] == cluster_id]
                            
                            # Calculate cluster statistics
                            center_lat = cluster_data['latitude'].mean()
                            center_lon = cluster_data['longitude'].mean()
                            visit_count = len(cluster_data)
                            
                            # Calculate bounding box
                            min_lat, max_lat = cluster_data['latitude'].min(), cluster_data['latitude'].max()
                            min_lon, max_lon = cluster_data['longitude'].min(), cluster_data['longitude'].max()
                            
                            # Get polygon WKT if available
                            polygon_wkt = None
                            if cluster_id in cluster_polygons and cluster_id != 'outliers':
                                # Convert Leaflet coordinates back to WKT
                                coords = cluster_polygons[cluster_id]
                                if coords and len(coords) > 2:
                                    # Convert from [lat, lon] back to (lon, lat) for WKT
                                    wkt_coords = [(coord[1], coord[0]) for coord in coords]
                                    # Ensure polygon is closed
                                    if wkt_coords[0] != wkt_coords[-1]:
                                        wkt_coords.append(wkt_coords[0])
                                    coord_pairs = [f"{lon} {lat}" for lon, lat in wkt_coords]
                                    polygon_wkt = f"POLYGON (({', '.join(coord_pairs)}))"
                            
                            cluster_details.append({
                                'opp_id': opp_id,
                                'opp_name': opp_name,
                                'cluster_id': cluster_id,
                                'algorithm': algorithm,
                                'visit_count': visit_count,
                                'center_latitude': round(center_lat, 6),
                                'center_longitude': round(center_lon, 6),
                                'bbox_min_lat': round(min_lat, 6),
                                'bbox_max_lat': round(max_lat, 6),
                                'bbox_min_lon': round(min_lon, 6),
                                'bbox_max_lon': round(max_lon, 6),
                                'polygon_wkt': polygon_wkt,
                                'buffer_distance_m': buffer_distance,
                                'outlier_distance_m': outlier_distance if 'outlier_distance' in locals() else None
                            })
            
            if cluster_details:
                cluster_details_df = pd.DataFrame(cluster_details)
                
                # Sort by OPP ID and cluster ID
                cluster_details_df = cluster_details_df.sort_values(['opp_id', 'cluster_id'])
                
                # Create filename with parameters (matching map filename pattern)
                if algorithm == 'Growth':
                    algorithm_suffix = f'growth_{int(outlier_distance)}m_{int(buffer_distance)}buf'
                else:
                    algorithm_suffix = {
                        'K-means': 'kmeans',
                        'Simple': 'simpledist'
                    }.get(algorithm, algorithm.lower())
                
                # Add polygon type to filename
                if polygon_type == "Alpha shape":
                    if alpha_param > 0:
                        polygon_suffix = f'_alpha{alpha_param:.1f}'.replace('.', 'p')
                    else:
                        polygon_suffix = '_alphaauto'
                else:
                    polygon_suffix = '_convex'
                
                algorithm_suffix += polygon_suffix
                
                details_file = os.path.join(cluster_dir, f"cluster_details_{algorithm_suffix}.csv")
                cluster_details_df.to_csv(details_file, index=False)
                
                self.log(f"Created cluster details file: {os.path.basename(details_file)} ({len(cluster_details_df)} clusters)")
                return details_file
                
        except Exception as e:
            self.log(f"Error creating cluster details file: {str(e)}")
            
        return None
    
    def _log_clustering_summary(self, clustered_data, cluster_stats, opp_id):
        """Log summary of clustering results for a specific OPP ID"""
        
        total_visits = len(clustered_data)
        outliers_count = (clustered_data['cluster_id'] == 'outliers').sum()
        regular_clusters = len([c for c in clustered_data['cluster_id'].unique() if c != 'outliers'])
        
        self.log(f"  OPP {opp_id} Summary:")
        self.log(f"    Total visits: {total_visits}")
        self.log(f"    Regular clusters: {regular_clusters}")
        self.log(f"    Outliers: {outliers_count} ({100*outliers_count/total_visits:.1f}%)")
        
        if regular_clusters > 0:
            # Get cluster sizes excluding outliers
            regular_stats = cluster_stats[cluster_stats['cluster_id'] != 'outliers']
            if len(regular_stats) > 0:
                cluster_sizes = regular_stats['visit_count']
                self.log(f"    Avg cluster size: {cluster_sizes.mean():.1f} visits")
                self.log(f"    Cluster size range: {cluster_sizes.min()}-{cluster_sizes.max()} visits")
