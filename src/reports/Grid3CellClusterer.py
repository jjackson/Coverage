# Grid3CellClusterer.py

"""
Grid3 Cell Clustering Module

Identifies spatially connected clusters of populated grid cells within wards.
Uses connected components algorithm to group adjacent cells (4-connectivity).
Only processes cells with population > 0.

Key features:
- Ward-level clustering (clusters don't cross ward boundaries)
- Multi-resolution support (100m, 200m, 300m, 500m, 700m, 1000m)
- Efficient connected components algorithm
- Generates both cell-level and cluster-level outputs
- Integration with existing Grid3WardAnalysis workflow
- Building statistics per cluster

Usage:
    clusterer = Grid3CellClusterer()
    enhanced_cells, cluster_summaries = clusterer.add_clustering_to_cells(cells_df, resolution)
"""

import pandas as pd
import numpy as np
from collections import deque, defaultdict


class Grid3CellClusterer:
    """Handles spatial clustering of populated grid cells within wards"""
    
    def __init__(self, log_func=None):
        """
        Initialize clusterer
        
        Args:
            log_func: Optional logging function
        """
        self.log = log_func if log_func else lambda x: print(x)
    
    def add_clustering_to_cells(self, cells_df, resolution):
        """
        Add clustering information to cells dataframe
        
        Args:
            cells_df: DataFrame with grid cells (must have 'row', 'col', 'ward_id', 'population')
            resolution: Grid resolution in meters (for logging)
            
        Returns:
            tuple: (enhanced_cells_df, cluster_summaries_df)
                - enhanced_cells_df: Original cells with cluster columns added
                - cluster_summaries_df: One row per cluster with statistics
        """
        if len(cells_df) == 0:
            return cells_df.copy(), pd.DataFrame()
        
        self.log(f"[clusterer] Starting clustering for {resolution}m resolution...")
        
        # Initialize cluster columns
        enhanced_cells = cells_df.copy()
        enhanced_cells['cluster_id'] = None
        enhanced_cells['cluster_size'] = 0
        enhanced_cells['cluster_population'] = 0.0
        
        cluster_summaries = []
        total_clusters = 0
        
        # Process each ward separately
        for ward_id in cells_df['ward_id'].unique():
            ward_cells = cells_df[cells_df['ward_id'] == ward_id]
            
            # Only cluster populated cells
            populated_cells = ward_cells[ward_cells['population'] > 0]
            
            if len(populated_cells) == 0:
                continue
            
            # Find clusters for this ward
            ward_clusters = self._find_clusters_in_ward(populated_cells, ward_id)
            ward_cluster_count = len(ward_clusters)
            total_clusters += ward_cluster_count
            
            # Update cells with cluster information
            for cluster_info in ward_clusters:
                cluster_id = cluster_info['cluster_id']
                cell_indices = cluster_info['cell_indices']
                cluster_size = cluster_info['cluster_size']
                cluster_population = cluster_info['cluster_population']
                
                # Update the enhanced_cells dataframe
                mask = enhanced_cells.index.isin(cell_indices)
                enhanced_cells.loc[mask, 'cluster_id'] = cluster_id
                enhanced_cells.loc[mask, 'cluster_size'] = cluster_size
                enhanced_cells.loc[mask, 'cluster_population'] = cluster_population
                
                # Add to cluster summaries
                cluster_summaries.append(cluster_info)
        
        # Convert cluster summaries to DataFrame
        cluster_summaries_df = pd.DataFrame(cluster_summaries) if cluster_summaries else pd.DataFrame()
        
        self.log(f"[clusterer] {resolution}m clustering complete: {total_clusters} total clusters across {cells_df['ward_id'].nunique()} wards")
        
        return enhanced_cells, cluster_summaries_df
    
    def _find_clusters_in_ward(self, populated_cells, ward_id):
        """
        Find all clusters within a single ward using connected components
        
        Args:
            populated_cells: DataFrame of cells with population > 0 in this ward
            ward_id: Ward identifier
            
        Returns:
            list: List of cluster dictionaries with statistics
        """
        if len(populated_cells) == 0:
            return []
        
        # Create coordinate to index mapping for fast neighbor lookup
        coord_to_idx = {}
        for idx, row in populated_cells.iterrows():
            coord_to_idx[(row['row'], row['col'])] = idx
        
        # Track visited cells and find connected components
        visited = set()
        clusters = []
        cluster_counter = 1
        
        for idx, cell_row in populated_cells.iterrows():
            if idx in visited:
                continue
            
            # Start new cluster with BFS
            cluster_cells = self._bfs_cluster(
                cell_row['row'], cell_row['col'], 
                coord_to_idx, visited
            )
            
            if cluster_cells:
                # Calculate cluster statistics
                cluster_id = f"{ward_id}_cluster_{cluster_counter}"
                cluster_size = len(cluster_cells)
                
                # Get population and visits for all cells in cluster
                cluster_data = populated_cells.loc[cluster_cells]
                cluster_population = cluster_data['population'].sum()
                cluster_visits = cluster_data['visits_in_cell'].sum() if 'visits_in_cell' in cluster_data.columns else 0
                
                # Building statistics for cluster
                cluster_buildings = int(cluster_data['num_buildings'].sum()) if 'num_buildings' in cluster_data.columns else 0
                cluster_building_area = cluster_data['total_building_area_m2'].sum() if 'total_building_area_m2' in cluster_data.columns else 0.0
                
                # Get ward info from first cell
                first_cell = populated_cells.loc[cluster_cells[0]]
                ward_name = first_cell.get('ward_name', 'Unknown')
                state_name = first_cell.get('state_name', 'Unknown')
                
                # Calculate cluster bounds
                cluster_data = populated_cells.loc[cluster_cells]
                min_row, max_row = cluster_data['row'].min(), cluster_data['row'].max()
                min_col, max_col = cluster_data['col'].min(), cluster_data['col'].max()
                
                # Calculate centroid (population-weighted)
                total_pop = cluster_data['population'].sum()
                if total_pop > 0:
                    centroid_lat = (cluster_data['center_latitude'] * cluster_data['population']).sum() / total_pop
                    centroid_lon = (cluster_data['center_longitude'] * cluster_data['population']).sum() / total_pop
                else:
                    centroid_lat = cluster_data['center_latitude'].mean()
                    centroid_lon = cluster_data['center_longitude'].mean()
                
                cluster_info = {
                    'cluster_id': cluster_id,
                    'ward_id': ward_id,
                    'ward_name': ward_name,
                    'state_name': state_name,
                    'cluster_size': cluster_size,
                    'cluster_visits': cluster_visits,
                    'cluster_population': cluster_population,
                    'cluster_buildings': cluster_buildings,
                    'cluster_building_area_m2': cluster_building_area,
                    'min_row': min_row,
                    'max_row': max_row,
                    'min_col': min_col,
                    'max_col': max_col,
                    'centroid_latitude': centroid_lat,
                    'centroid_longitude': centroid_lon,
                    'cell_indices': cluster_cells  # Store for updating main dataframe
                }
                
                clusters.append(cluster_info)
                cluster_counter += 1
        
        return clusters
    
    def _bfs_cluster(self, start_row, start_col, coord_to_idx, visited):
        """
        Use breadth-first search to find all connected cells in a cluster
        
        Args:
            start_row, start_col: Starting cell coordinates
            coord_to_idx: Dictionary mapping (row, col) -> dataframe index
            visited: Set of already visited indices
            
        Returns:
            list: List of dataframe indices in this cluster
        """
        start_coord = (start_row, start_col)
        if start_coord not in coord_to_idx:
            return []
        
        start_idx = coord_to_idx[start_coord]
        if start_idx in visited:
            return []
        
        # BFS to find all connected cells
        cluster_cells = []
        queue = deque([start_coord])
        local_visited = set()
        
        while queue:
            row, col = queue.popleft()
            coord = (row, col)
            
            if coord in local_visited or coord not in coord_to_idx:
                continue
            
            idx = coord_to_idx[coord]
            if idx in visited:
                continue
            
            # Add to cluster
            local_visited.add(coord)
            visited.add(idx)
            cluster_cells.append(idx)
            
            # Check 4-connected neighbors
            neighbors = [
                (row - 1, col),  # North
                (row + 1, col),  # South
                (row, col - 1),  # West
                (row, col + 1)   # East
            ]
            
            for neighbor in neighbors:
                if neighbor not in local_visited and neighbor in coord_to_idx:
                    queue.append(neighbor)
        
        return cluster_cells
    
    def create_cluster_summary_file(self, cluster_summaries_df, output_file):
        """
        Write cluster summaries to CSV file
        
        Args:
            cluster_summaries_df: DataFrame with cluster statistics
            output_file: Path for output CSV file
        """
        if len(cluster_summaries_df) == 0:
            self.log(f"[clusterer] No clusters to write to {output_file}")
            return
        
        # Select columns for output (exclude cell_indices which is for internal use)
        output_cols = [col for col in cluster_summaries_df.columns if col != 'cell_indices']
        cluster_summaries_df[output_cols].to_csv(output_file, index=False)
        
        self.log(f"[clusterer] Wrote cluster summaries: {output_file}")
    
    def get_cluster_statistics(self, cluster_summaries_df):
        """
        Generate overall clustering statistics
        
        Args:
            cluster_summaries_df: DataFrame with cluster information
            
        Returns:
            dict: Dictionary with clustering statistics
        """
        if len(cluster_summaries_df) == 0:
            return {
                'total_clusters': 0,
                'total_wards_with_clusters': 0,
                'avg_cluster_size': 0,
                'avg_cluster_population': 0,
                'largest_cluster_size': 0,
                'largest_cluster_population': 0
            }
        
        stats = {
            'total_clusters': len(cluster_summaries_df),
            'total_wards_with_clusters': cluster_summaries_df['ward_id'].nunique(),
            'avg_cluster_size': cluster_summaries_df['cluster_size'].mean(),
            'avg_cluster_population': cluster_summaries_df['cluster_population'].mean(),
            'largest_cluster_size': cluster_summaries_df['cluster_size'].max(),
            'largest_cluster_population': cluster_summaries_df['cluster_population'].max(),
            'median_cluster_size': cluster_summaries_df['cluster_size'].median(),
            'median_cluster_population': cluster_summaries_df['cluster_population'].median()
        }
        
        return stats
    
    def create_hamlet_analysis(self, cluster_summaries_df, ward_cells_df, population_thresholds=[50, 100, 200, 500]):
        """
        Create hamlet analysis report - one row per ward showing cluster statistics by population thresholds
        
        Args:
            cluster_summaries_df: DataFrame with cluster statistics
            ward_cells_df: DataFrame with all ward cells (for total population calculation)
            population_thresholds: List of population thresholds to analyze
            
        Returns:
            DataFrame: One row per ward with cluster statistics by population threshold
        """
        if len(cluster_summaries_df) == 0:
            return pd.DataFrame()
        
        hamlet_analysis = []
        
        # Process each ward
        for ward_id in cluster_summaries_df['ward_id'].unique():
            ward_clusters = cluster_summaries_df[cluster_summaries_df['ward_id'] == ward_id]
            ward_cells = ward_cells_df[ward_cells_df['ward_id'] == ward_id] if len(ward_cells_df) > 0 else pd.DataFrame()
            
            # Basic ward info
            first_cluster = ward_clusters.iloc[0]
            ward_name = first_cluster['ward_name']
            state_name = first_cluster['state_name']
            
            # Total ward population
            total_ward_population = ward_cells['population'].sum() if len(ward_cells) > 0 else ward_clusters['cluster_population'].sum()
            
            # Basic cluster statistics
            clusters_total = len(ward_clusters)
            total_population_in_clusters = ward_clusters['cluster_population'].sum()
            total_visits_in_clusters = ward_clusters['cluster_visits'].sum()
            median_cluster_size = ward_clusters['cluster_size'].median()
            median_cluster_population = ward_clusters['cluster_population'].median()
            
            # Initialize ward info dictionary
            ward_info = {
                'ward_id': ward_id,
                'ward_name': ward_name,
                'state_name': state_name,
                'clusters_total': clusters_total,
                'total_population_in_clusters': total_population_in_clusters,
                'total_visits_in_clusters': total_visits_in_clusters,
                'median_cluster_size': median_cluster_size,
                'median_cluster_population': median_cluster_population
            }
            
            # Calculate statistics for each population threshold
            for threshold in population_thresholds:
                threshold_clusters = ward_clusters[ward_clusters['cluster_population'] <= threshold]
                
                clusters_count = len(threshold_clusters)
                threshold_total_pop = threshold_clusters['cluster_population'].sum()
                percent_pop = (threshold_total_pop / total_population_in_clusters * 100) if total_population_in_clusters > 0 else 0
                median_threshold_size = threshold_clusters['cluster_size'].median() if len(threshold_clusters) > 0 else 0
                
                visited_clusters = threshold_clusters[threshold_clusters['cluster_visits'] > 0]
                clusters_visited_count = len(visited_clusters)
                visited_population = visited_clusters['cluster_population'].sum()
                percent_visited_pop = (visited_population / threshold_total_pop * 100) if threshold_total_pop > 0 else 0
                
                ward_info.update({
                    f'clusters_pop_le_{threshold}': clusters_count,
                    f'total_pop_in_clusters_le_{threshold}': threshold_total_pop,
                    f'percent_pop_in_clusters_le_{threshold}': percent_pop,
                    f'median_cluster_size_le_{threshold}': median_threshold_size,
                    f'clusters_le_{threshold}_visited': clusters_visited_count,
                    f'percent_pop_visited_in_clusters_le_{threshold}': percent_visited_pop
                })
            
            # Medium-sized clusters (>100 and <=500)
            medium_clusters = ward_clusters[(ward_clusters['cluster_population'] > 100) & (ward_clusters['cluster_population'] <= 500)]
            medium_clusters_count = len(medium_clusters)
            medium_total_pop = medium_clusters['cluster_population'].sum()
            medium_percent_pop = (medium_total_pop / total_population_in_clusters * 100) if total_population_in_clusters > 0 else 0
            medium_median_size = medium_clusters['cluster_size'].median() if len(medium_clusters) > 0 else 0
            
            medium_visited_clusters = medium_clusters[medium_clusters['cluster_visits'] > 0]
            medium_clusters_visited_count = len(medium_visited_clusters)
            medium_visited_population = medium_visited_clusters['cluster_population'].sum()
            medium_percent_visited_pop = (medium_visited_population / medium_total_pop * 100) if medium_total_pop > 0 else 0
            
            ward_info.update({
                'clusters_pop_100_to_500': medium_clusters_count,
                'total_pop_in_clusters_100_to_500': medium_total_pop,
                'percent_pop_in_clusters_100_to_500': medium_percent_pop,
                'median_cluster_size_100_to_500': medium_median_size,
                'clusters_100_to_500_visited': medium_clusters_visited_count,
                'percent_pop_visited_in_clusters_100_to_500': medium_percent_visited_pop
            })
            
            # Large clusters (>500)
            high_clusters = ward_clusters[ward_clusters['cluster_population'] > 500]
            high_clusters_count = len(high_clusters)
            high_total_pop = high_clusters['cluster_population'].sum()
            high_percent_pop = (high_total_pop / total_population_in_clusters * 100) if total_population_in_clusters > 0 else 0
            high_median_size = high_clusters['cluster_size'].median() if len(high_clusters) > 0 else 0
            
            high_visited_clusters = high_clusters[high_clusters['cluster_visits'] > 0]
            high_clusters_visited_count = len(high_visited_clusters)
            high_visited_population = high_visited_clusters['cluster_population'].sum()
            high_percent_visited_pop = (high_visited_population / high_total_pop * 100) if high_total_pop > 0 else 0
            
            ward_info.update({
                'clusters_pop_greater_500': high_clusters_count,
                'total_pop_in_clusters_greater_500': high_total_pop,
                'percent_pop_in_clusters_greater_500': high_percent_pop,
                'median_cluster_size_greater_500': high_median_size,
                'clusters_greater_500_visited': high_clusters_visited_count,
                'percent_pop_visited_in_clusters_greater_500': high_percent_visited_pop
            })

            hamlet_analysis.append(ward_info)
        
        return pd.DataFrame(hamlet_analysis)
    
    def create_hamlet_analysis_file(self, hamlet_analysis_df, output_file):
        """
        Write hamlet analysis to CSV file
        
        Args:
            hamlet_analysis_df: DataFrame with hamlet analysis
            output_file: Path for output CSV file
        """
        if len(hamlet_analysis_df) == 0:
            self.log(f"[clusterer] No population threshold analysis to write to {output_file}")
            return
        
        hamlet_analysis_df.to_csv(output_file, index=False)
        self.log(f"[clusterer] Wrote population threshold analysis: {output_file}")
        
        total_wards = len(hamlet_analysis_df)
        self.log(f"[clusterer] Population threshold analysis: {total_wards} wards analyzed")


def add_clustering_to_analysis(cells_df, resolution, output_dir, data_tag, log_func=None):
    """
    Convenience function to add clustering to existing analysis workflow
    
    Args:
        cells_df: DataFrame with grid cells
        resolution: Grid resolution in meters
        output_dir: Directory for output files
        data_tag: Unique identifier for this dataset
        log_func: Optional logging function
        
    Returns:
        tuple: (enhanced_cells_df, cluster_csv_file_path)
    """
    clusterer = Grid3CellClusterer(log_func)
    
    enhanced_cells, cluster_summaries = clusterer.add_clustering_to_cells(cells_df, resolution)
    
    cluster_file = f"{output_dir}/cluster_summary_{resolution}m_{data_tag}.csv"
    clusterer.create_cluster_summary_file(cluster_summaries, cluster_file)
    
    if log_func:
        stats = clusterer.get_cluster_statistics(cluster_summaries)
        log_func(f"[clusterer] {resolution}m cluster stats: {stats['total_clusters']} clusters, "
                f"avg size {stats['avg_cluster_size']:.1f} cells, "
                f"largest {stats['largest_cluster_size']} cells")
    
    return enhanced_cells, cluster_file
