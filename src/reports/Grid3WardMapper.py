# grid3_ward_mapper.py

"""
Grid3 Ward Mapper

Generates interactive Leaflet maps showing grid cells with population data for each ward.
Called from Grid3WardAnalysis to create HTML map visualizations.

Features:
- One map per ward showing 300m grid cells
- Population density color coding
- Interactive hover details
- Ward boundary overlay
- Visit markers as orange circles
- Cluster boundary outlines in purple
- Building footprints as gray squares
- Clean HTML output with embedded JavaScript
"""

import os
import json
import numpy as np
import pandas as pd
from pyproj import Transformer
from affine import Affine


class Grid3WardMapper:
    """Generates interactive ward maps showing Grid3 population data"""
    
    @staticmethod
    def generate_ward_maps(cells_300m_df, ward_boundaries_gdf, enriched_visits, output_dir, buildings_df=None, log_func=None):
        """
        Generate HTML maps for each ward showing 300m grid cells
        
        Args:
            cells_300m_df: DataFrame with 300m cell data (from global_cells[300])
            ward_boundaries_gdf: GeoDataFrame with ward boundaries 
            enriched_visits: DataFrame with enriched visit data for visit markers
            output_dir: Main output directory
            buildings_df: Optional DataFrame with building data
            log_func: Optional logging function
            
        Returns:
            List of created HTML map file paths
        """
        
        def log(message):
            if log_func:
                log_func(f"[ward_mapper] {message}")
            else:
                print(f"[ward_mapper] {message}")
        
        if cells_300m_df is None or len(cells_300m_df) == 0:
            log("No 300m cell data provided, skipping map generation")
            return []
        
        log(f"Generating ward maps for {len(cells_300m_df):,} cells across {cells_300m_df['ward_id'].nunique()} wards")
        
        # Create maps subdirectory
        maps_dir = os.path.join(output_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        
        output_files = []
        
        # Process each ward
        for ward_id in cells_300m_df['ward_id'].unique():
            try:
                ward_cells = cells_300m_df[cells_300m_df['ward_id'] == ward_id].copy()
                ward_boundary = ward_boundaries_gdf[ward_boundaries_gdf['ward_id'] == ward_id].iloc[0]
                
                # Filter visits by ward_id
                ward_visits = enriched_visits[enriched_visits['ward_id'] == ward_id].copy()
                
                # Prepare building markers for this ward
                building_markers = Grid3WardMapper._prepare_building_markers(buildings_df, ward_id) if buildings_df is not None else []
                
                ward_name = ward_boundary.get('ward_name', 'Unknown')
                state_name = ward_boundary.get('state_name', 'Unknown')
                
                log(f"Creating map for ward {ward_id} ({ward_name}, {state_name}) - {len(ward_cells)} cells, {len(ward_visits)} visits, {len(building_markers)} buildings")
                
                # Generate map
                map_file = Grid3WardMapper._create_ward_map(
                    ward_cells, ward_boundary, ward_visits, building_markers, ward_id, ward_name, state_name, maps_dir
                )
                
                if map_file:
                    output_files.append(map_file)
                    log(f"Created: {os.path.basename(map_file)}")
                
            except Exception as e:
                log(f"Error creating map for ward {ward_id}: {str(e)}")
                continue
        
        log(f"Generated {len(output_files)} ward maps in maps/ subdirectory")
        return output_files
    
    @staticmethod
    def _prepare_building_markers(buildings_df, ward_id):
        """
        Prepare building marker data for a specific ward
        
        Args:
            buildings_df: DataFrame with all building data
            ward_id: Ward to filter buildings for
            
        Returns:
            List of building dicts with lat, lon, area, confidence
        """
        if buildings_df is None or len(buildings_df) == 0:
            return []
        
        ward_buildings = buildings_df[buildings_df['ward_id'] == ward_id]
        
        building_markers = []
        for _, bldg in ward_buildings.iterrows():
            building_markers.append({
                'lat': float(bldg['latitude']),
                'lon': float(bldg['longitude']),
                'area': float(bldg['area_in_meters']),
                'confidence': float(bldg['confidence'])
            })
        
        return building_markers
    
    @staticmethod
    def _create_ward_map(ward_cells, ward_boundary, ward_visits, building_markers, ward_id, ward_name, state_name, maps_dir):
        """Create HTML map for a single ward"""
        
        # Calculate cell rectangles
        cell_rectangles = Grid3WardMapper._calculate_cell_rectangles(ward_cells)
        
        if len(cell_rectangles) == 0:
            return None
        
        # Calculate cluster boundaries
        cluster_boundaries = Grid3WardMapper._calculate_cluster_boundaries(ward_cells)
        
        # Calculate map center from cells
        center_lat = ward_cells['center_latitude'].mean()
        center_lon = ward_cells['center_longitude'].mean()
        
        # Get ward boundary coordinates
        ward_boundary_coords = Grid3WardMapper._extract_boundary_coords(ward_boundary.geometry)
        
        # Calculate population color scale
        pop_values = ward_cells['population'].values
        pop_values = pop_values[~np.isnan(pop_values)]
        
        if len(pop_values) > 0:
            pop_min = np.min(pop_values[pop_values > 0]) if np.any(pop_values > 0) else 0
            pop_max = np.max(pop_values)
        else:
            pop_min = pop_max = 0
        
        # Create HTML content
        html_content = Grid3WardMapper._generate_html_content(
            cell_rectangles, cluster_boundaries, ward_boundary_coords, ward_visits, building_markers,
            center_lat, center_lon, ward_id, ward_name, state_name, pop_min, pop_max
        )
        
        # Save HTML file
        safe_ward_name = Grid3WardMapper._safe_filename(ward_name)
        safe_ward_id = Grid3WardMapper._safe_filename(ward_id)
        filename = f"ward_grid_map_300m_{safe_ward_name}_{safe_ward_id}.html"
        output_path = os.path.join(maps_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    @staticmethod
    def _calculate_cluster_boundaries(ward_cells):
        """Calculate cluster boundary polygons using grid-based edge detection"""
        
        populated_cells = ward_cells[
            (ward_cells['population'] > 0) & 
            (ward_cells['cluster_id'].notna())
        ].copy()
        
        if len(populated_cells) == 0:
            return []
        
        cluster_boundaries = []
        
        for cluster_id in populated_cells['cluster_id'].unique():
            cluster_cells = populated_cells[populated_cells['cluster_id'] == cluster_id]
            
            if len(cluster_cells) == 1:
                cell = cluster_cells.iloc[0]
                boundary = Grid3WardMapper._create_single_cell_boundary(cell)
                cluster_boundaries.append({
                    'cluster_id': cluster_id,
                    'boundary': boundary,
                    'cell_count': 1
                })
            else:
                boundary = Grid3WardMapper._find_cluster_perimeter(cluster_cells)
                if boundary:
                    cluster_boundaries.append({
                        'cluster_id': cluster_id,
                        'boundary': boundary,
                        'cell_count': len(cluster_cells)
                    })
        
        return cluster_boundaries
    
    @staticmethod
    def _create_single_cell_boundary(cell):
        """Create boundary for a single cell"""
        cell_size_deg = 0.0027
        half_size = cell_size_deg / 2
        
        center_lat = cell['center_latitude']
        center_lon = cell['center_longitude']
        
        boundary = [
            [center_lat - half_size, center_lon - half_size],
            [center_lat - half_size, center_lon + half_size],
            [center_lat + half_size, center_lon + half_size],
            [center_lat + half_size, center_lon - half_size],
            [center_lat - half_size, center_lon - half_size]
        ]
        
        return boundary
    
    @staticmethod
    def _find_cluster_perimeter(cluster_cells):
        """Find the perimeter of a cluster using grid-based edge detection"""
        
        cell_lookup = {}
        for _, cell in cluster_cells.iterrows():
            key = (int(cell['row']), int(cell['col']))
            cell_lookup[key] = {
                'center_lat': cell['center_latitude'],
                'center_lon': cell['center_longitude']
            }
        
        external_edges = []
        cell_size_deg = 0.0027
        half_size = cell_size_deg / 2
        
        for (row, col), cell_info in cell_lookup.items():
            center_lat = cell_info['center_lat']
            center_lon = cell_info['center_lon']
            
            neighbors = [
                (row - 1, col),  # North
                (row + 1, col),  # South
                (row, col + 1),  # East
                (row, col - 1)   # West
            ]
            
            directions = ['north', 'south', 'east', 'west']
            
            for i, neighbor_key in enumerate(neighbors):
                if neighbor_key not in cell_lookup:
                    direction = directions[i]
                    
                    if direction == 'north':
                        edge = [
                            [center_lat + half_size, center_lon - half_size],
                            [center_lat + half_size, center_lon + half_size]
                        ]
                    elif direction == 'south':
                        edge = [
                            [center_lat - half_size, center_lon + half_size],
                            [center_lat - half_size, center_lon - half_size]
                        ]
                    elif direction == 'east':
                        edge = [
                            [center_lat + half_size, center_lon + half_size],
                            [center_lat - half_size, center_lon + half_size]
                        ]
                    elif direction == 'west':
                        edge = [
                            [center_lat - half_size, center_lon - half_size],
                            [center_lat + half_size, center_lon - half_size]
                        ]
                    
                    external_edges.append(edge)
        
        if external_edges:
            boundary = Grid3WardMapper._edges_to_polygon(external_edges)
            return boundary
        
        return None
    
    @staticmethod
    def _edges_to_polygon(edges):
        """Convert a list of edges to a continuous polygon boundary"""
        
        if not edges:
            return None
        
        vertices = []
        for edge in edges:
            vertices.extend(edge)
        
        unique_vertices = []
        seen = set()
        for vertex in vertices:
            vertex_tuple = (round(vertex[0], 8), round(vertex[1], 8))
            if vertex_tuple not in seen:
                seen.add(vertex_tuple)
                unique_vertices.append(vertex)
        
        if len(unique_vertices) < 3:
            return None
        
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        start = min(unique_vertices, key=lambda p: (p[1], p[0]))
        
        hull = []
        current = start
        
        while True:
            hull.append(current)
            next_point = unique_vertices[0]
            
            for candidate in unique_vertices[1:]:
                if next_point == current or cross_product(current, next_point, candidate) < 0:
                    next_point = candidate
            
            current = next_point
            if current == start:
                break
        
        if len(hull) > 0 and hull[0] != hull[-1]:
            hull.append(hull[0])
        
        return hull if len(hull) >= 4 else None
    
    @staticmethod
    def _calculate_cell_rectangles(ward_cells):
        """Calculate geographic rectangles for each grid cell"""
        
        rectangles = []
        
        for _, cell in ward_cells.iterrows():
            row = cell['row']
            col = cell['col'] 
            population = cell['population']
            center_lat = cell['center_latitude']
            center_lon = cell['center_longitude']
            cell_id = cell['cell_id']
            
            cell_size_deg = 0.0027
            half_size = cell_size_deg / 2
            
            bounds = {
                'north': center_lat + half_size,
                'south': center_lat - half_size,
                'east': center_lon + half_size,
                'west': center_lon - half_size
            }
            
            rectangles.append({
                'bounds': bounds,
                'population': float(population) if not np.isnan(population) else 0,
                'cell_id': cell_id,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'row': int(row),
                'col': int(col),
                'visits_in_cell': int(cell.get('visits_in_cell', 0)) if 'visits_in_cell' in cell else 0,
                'unique_flws': int(cell.get('unique_flws', 0)) if 'unique_flws' in cell else 0,
                'cluster_id': cell.get('cluster_id', None),
                'cluster_size': int(cell.get('cluster_size', 0)) if 'cluster_size' in cell else 0,
                'cluster_population': float(cell.get('cluster_population', 0)) if 'cluster_population' in cell else 0,
                'num_buildings': int(cell.get('num_buildings', 0)) if 'num_buildings' in cell else 0,
                'total_building_area_m2': float(cell.get('total_building_area_m2', 0)) if 'total_building_area_m2' in cell else 0,
                'avg_building_confidence': float(cell.get('avg_building_confidence', 0)) if 'avg_building_confidence' in cell else 0
            })
        
        return rectangles
    
    @staticmethod
    def _extract_boundary_coords(geometry):
        """Extract ward boundary coordinates for Leaflet"""
        
        coords = []
        
        if geometry.geom_type == 'Polygon':
            exterior_coords = list(geometry.exterior.coords)
            coords = [[lat, lon] for lon, lat in exterior_coords]
            
        elif geometry.geom_type == 'MultiPolygon':
            largest_poly = max(geometry.geoms, key=lambda p: p.area)
            exterior_coords = list(largest_poly.exterior.coords)
            coords = [[lat, lon] for lon, lat in exterior_coords]
        
        return coords
    
    @staticmethod
    def _get_population_color(population, min_pop, max_pop):
        """Get color for population value"""
        
        if population == 0 or np.isnan(population):
            return '#f0f0f0'
        
        if max_pop <= min_pop:
            return '#3182bd'
        
        normalized = (population - min_pop) / (max_pop - min_pop)
        normalized = max(0, min(1, normalized))
        
        light_blue = np.array([173, 216, 230])
        dark_blue = np.array([0, 100, 200])
        
        color_rgb = light_blue + normalized * (dark_blue - light_blue)
        color_rgb = color_rgb.astype(int)
        
        return f'#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}'
    
    @staticmethod
    def _safe_filename(name):
        """Create safe filename from ward name"""
        import re
    
        safe = name.replace('/', '_').replace('\\', '_')
        safe = re.sub(r'[<>:"|?*]', '_', safe)
        safe = ''.join(c for c in safe if c.isalnum() or c in '-_ ')
        safe = re.sub(r'[_\s]+', '_', safe)
        safe = safe.strip('_')
    
        return safe[:50]

    @staticmethod
    def _generate_html_content(cell_rectangles, cluster_boundaries, ward_boundary_coords, ward_visits, building_markers,
                          center_lat, center_lon, ward_id, ward_name, state_name, pop_min, pop_max):
        """Generate complete HTML content for the ward map"""
        
        # Prepare cell data for JavaScript
        cells_data = []
        for rect in cell_rectangles:
            color = Grid3WardMapper._get_population_color(rect['population'], pop_min, pop_max)
            cells_data.append({
                'bounds': rect['bounds'],
                'population': rect['population'],
                'cell_id': rect['cell_id'],
                'color': color,
                'center_lat': rect['center_lat'],
                'center_lon': rect['center_lon'],
                'row': rect['row'],
                'col': rect['col'],
                'visits_in_cell': rect['visits_in_cell'],
                'unique_flws': rect['unique_flws'],
                'cluster_id': rect['cluster_id'],
                'cluster_size': rect['cluster_size'],
                'cluster_population': rect['cluster_population'],
                'num_buildings': rect['num_buildings'],
                'total_building_area_m2': rect['total_building_area_m2'],
                'avg_building_confidence': rect['avg_building_confidence']
            })
        
        # Prepare visit data for JavaScript
        visits_data = []
        if len(ward_visits) > 0:
            for _, visit in ward_visits.iterrows():
                visits_data.append({
                    'lat': float(visit['latitude']),
                    'lon': float(visit['longitude']),
                    'visit_id': str(visit.get('visit_id', 'unknown')),
                    'opportunity_id': str(visit.get('opportunity_id', 'unknown')),
                    'flw_id': str(visit.get('flw_id', 'unknown')),
                    'cell_100m_id': str(visit.get('cell_100m_id', 'unknown')),
                    'cell_300m_id': str(visit.get('cell_300m_id', 'unknown'))
                })
        
        # Building markers already in correct format
        buildings_data = building_markers
        
        # Prepare cluster data with population information
        clusters_data = []

        # Count clusters by size category (do this ONCE before the loop)
        small_clusters = 0
        medium_clusters = 0
        large_clusters = 0

        for cluster in cluster_boundaries:
            cluster_id = cluster['cluster_id']
            # Find any cell in this cluster to get the cluster_population
            cluster_cell = next((rect for rect in cell_rectangles if rect['cluster_id'] == cluster_id), None)
            cluster_pop = cluster_cell['cluster_population'] if cluster_cell else 0
    
            # Count this cluster in the appropriate size category
            if cluster_pop < 100:
                small_clusters += 1
            elif cluster_pop <= 500:
                medium_clusters += 1
            else:
                large_clusters += 1
    
            clusters_data.append({
                'cluster_id': cluster_id,
                'boundary': cluster['boundary'],
                'cell_count': cluster['cell_count'],
                'cluster_population': cluster_pop
            })

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Grid3 Population Map - {ward_name} (Ward {ward_id})</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
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
        .info-panel {{
            background: white;
            padding: 10px;
            height: 10vh;
            border-bottom: 1px solid #ccc;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            left: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 5px;
            border: 2px solid rgba(0,0,0,0.2);
            z-index: 1000;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 2px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 12px;
            margin-right: 5px;
            border: 1px solid #ccc;
        }}
        .toggle-controls {{
            margin-left: auto;
        }}
        .toggle-controls button {{
            margin: 0 2px;
            padding: 5px 10px;
            border: 1px solid #ccc;
            background: #f9f9f9;
            cursor: pointer;
            border-radius: 3px;
        }}
        .toggle-controls button:hover {{
            background: #e9e9e9;
        }}
    </style>
</head>
<body>
    <div class="info-panel">
        <div>
            <strong>Ward:</strong> {ward_name} (ID: {ward_id}) | 
            <strong>State:</strong> {state_name} | 
            <strong>Grid Resolution:</strong> 300m | 
            <strong>Cells:</strong> {len(cells_data):,} |
            <strong>Clusters:</strong> {len(cluster_boundaries)} |
            <strong>Buildings:</strong> {len(building_markers):,} |
            <strong>Population Range:</strong> {pop_min:.0f} - {pop_max:.0f}
        </div>
        <div class="toggle-controls">
            <button onclick="toggleGridCells()">Toggle Grid</button>
            <button onclick="toggleClustersSmall()">Clusters <100 ({small_clusters})</button>
            <button onclick="toggleClustersMedium()">Clusters 100-500 ({medium_clusters})</button>
            <button onclick="toggleClustersLarge()">Clusters >500 ({large_clusters})</button>
            <button onclick="toggleWardBoundary()">Toggle Boundary</button>
            <button onclick="toggleVisits()">Toggle Visits ({len(visits_data)})</button>
            <button onclick="toggleBuildings()">Toggle Buildings ({len(building_markers)})</button>
            <button onclick="resetView()">Reset View</button>
        </div>
    </div>
    
    <div id="map"></div>
    
    <div class="legend">
        <strong>Population Density</strong><br>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #f0f0f0;"></div>
            No population
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #add8e6;"></div>
            Low ({pop_min:.0f})
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #0064c8;"></div>
            High ({pop_max:.0f})
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ff7800;"></div>
            Visit locations
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ff69b4; border: 2px solid #ff69b4;"></div>
            Clusters <100 pop
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #8a2be2; border: 2px solid #8a2be2;"></div>
            Clusters 100-500 pop
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #228b22; border: 2px solid #228b22;"></div>
            Clusters >500 pop
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #666666;"></div>
            Building footprints
        </div>
    </div>
    
    <script>
        const cellsData = {json.dumps(cells_data)};
        const clustersData = {json.dumps(clusters_data)};
        const wardBoundary = {json.dumps(ward_boundary_coords)};
        const visitsData = {json.dumps(visits_data)};
        const buildingsData = {json.dumps(buildings_data)};
        
        const map = L.map('map').setView([{center_lat}, {center_lon}], 14);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);
        
        const boundaryLayer = L.layerGroup().addTo(map);
        const gridLayer = L.layerGroup().addTo(map);
        const visitsLayer = L.layerGroup().addTo(map);
        const clustersSmallLayer = L.layerGroup().addTo(map);
        const clustersMediumLayer = L.layerGroup().addTo(map);
        const clustersLargeLayer = L.layerGroup().addTo(map);
        const buildingsLayer = L.layerGroup().addTo(map);
        
        cellsData.forEach(cell => {{
            const bounds = [
                [cell.bounds.south, cell.bounds.west],
                [cell.bounds.north, cell.bounds.east]
            ];
            
            const rectangle = L.rectangle(bounds, {{
                color: cell.color,
                fillColor: cell.color,
                fillOpacity: 0.7,
                weight: 1,
                opacity: 0.8
            }});
            
            rectangle.bindPopup(`
                <strong>Cell:</strong> ${{cell.cell_id}}<br>
                <strong>Population:</strong> ${{cell.population.toFixed(1)}}<br>
                <strong>Buildings:</strong> ${{cell.num_buildings}}<br>
                <strong>Building Area:</strong> ${{cell.total_building_area_m2.toFixed(1)}} m²<br>
                <strong>Avg Confidence:</strong> ${{cell.avg_building_confidence.toFixed(2)}}<br>
                <strong>Visits:</strong> ${{cell.visits_in_cell}}<br>
                <strong>Unique FLWs:</strong> ${{cell.unique_flws}}<br>
                <strong>Cluster ID:</strong> ${{cell.cluster_id !== null ? cell.cluster_id : 'N/A'}}<br>
                <strong>Cluster Size:</strong> ${{cell.cluster_size !== undefined ? cell.cluster_size : 'N/A'}}<br>
                <strong>Cluster Population:</strong> ${{cell.cluster_population !== undefined ? cell.cluster_population.toFixed(1) : 'N/A'}}<br>
                <strong>Grid Position:</strong> Row ${{cell.row}}, Col ${{cell.col}}<br>
                <strong>Center:</strong> ${{cell.center_lat.toFixed(6)}}, ${{cell.center_lon.toFixed(6)}}
            `);
            
            rectangle.on('mouseover', function(e) {{
                this.setStyle({{ weight: 3, opacity: 1 }});
            }});
            
            rectangle.on('mouseout', function(e) {{
                this.setStyle({{ weight: 1, opacity: 0.8 }});
            }});
            
            gridLayer.addLayer(rectangle);
        }});
        
        clustersData.forEach(cluster => {{
            if (cluster.boundary && cluster.boundary.length > 0) {{
                let color, targetLayer;
                
                if (cluster.cluster_population < 100) {{
                    color = '#ff69b4';  // Hot pink for small clusters
                    targetLayer = clustersSmallLayer;
                }} else if (cluster.cluster_population <= 500) {{
                    color = '#8a2be2';  // Blue-violet for medium clusters
                    targetLayer = clustersMediumLayer;
                }} else {{
                    color = '#228b22';  // Forest green for large clusters
                    targetLayer = clustersLargeLayer;
                }}
                
                const boundary = L.polygon(cluster.boundary, {{
                    color: color,
                    fillOpacity: 0,
                    weight: 3,
                    opacity: 0.8,
                    interactive: false
                }});
                
                targetLayer.addLayer(boundary);
            }}
        }});
        
        visitsData.forEach(visit => {{
            const marker = L.circleMarker([visit.lat, visit.lon], {{
                radius: 6,
                fillColor: '#ff7800',
                color: '#ff7800',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }});
            
            marker.bindPopup(`
                <strong>Visit:</strong> ${{visit.visit_id}}<br>
                <strong>Opportunity:</strong> ${{visit.opportunity_id}}<br>
                <strong>FLW:</strong> ${{visit.flw_id}}<br>
                <strong>300m Cell:</strong> ${{visit.cell_300m_id}}<br>
                <strong>Location:</strong> ${{visit.lat.toFixed(6)}}, ${{visit.lon.toFixed(6)}}
            `);
            
            marker.on('mouseover', function(e) {{
                this.setStyle({{ radius: 8, weight: 3 }});
            }});
            
            marker.on('mouseout', function(e) {{
                this.setStyle({{ radius: 6, weight: 2 }});
            }});
            
            visitsLayer.addLayer(marker);
        }});
        
        buildingsData.forEach(building => {{
            const sizeInDeg = Math.sqrt(building.area) / 111320 * 2;
            const halfSize = sizeInDeg / 2;
            
            const bounds = [
                [building.lat - halfSize, building.lon - halfSize],
                [building.lat + halfSize, building.lon + halfSize]
            ];
            
            const buildingSquare = L.rectangle(bounds, {{
                color: '#666666',
                fillColor: '#666666',
                fillOpacity: 0.6,
                weight: 1,
                opacity: 0.8
            }});
            
            buildingSquare.bindPopup(`
                <strong>Building</strong><br>
                <strong>Area:</strong> ${{building.area.toFixed(1)}} m²<br>
                <strong>Confidence:</strong> ${{building.confidence.toFixed(2)}}<br>
                <strong>Location:</strong> ${{building.lat.toFixed(6)}}, ${{building.lon.toFixed(6)}}
            `);
            
            buildingSquare.on('mouseover', function(e) {{
                this.setStyle({{ weight: 2, opacity: 1, fillOpacity: 0.8 }});
            }});
            
            buildingSquare.on('mouseout', function(e) {{
                this.setStyle({{ weight: 1, opacity: 0.8, fillOpacity: 0.6 }});
            }});
            
            buildingsLayer.addLayer(buildingSquare);
        }});
        
        if (wardBoundary && wardBoundary.length > 0) {{
            const boundary = L.polygon(wardBoundary, {{
                color: '#ff0000',
                fillOpacity: 0,
                weight: 3,
                opacity: 1,
                interactive: false
            }});
            
            boundaryLayer.addLayer(boundary);
        }}
        
        let gridVisible = true;
        let boundaryVisible = true;
        let visitsVisible = true;
        let clustersSmallVisible = true;
        let clustersMediumVisible = true;
        let clustersLargeVisible = true;
        let buildingsVisible = true;
        
        function toggleGridCells() {{
            if (gridVisible) {{
                map.removeLayer(gridLayer);
                gridVisible = false;
            }} else {{
                gridLayer.addTo(map);
                gridVisible = true;
            }}
        }}
        
        function toggleClustersSmall() {{
            if (clustersSmallVisible) {{
                map.removeLayer(clustersSmallLayer);
                clustersSmallVisible = false;
            }} else {{
                clustersSmallLayer.addTo(map);
                clustersSmallVisible = true;
            }}
        }}
        
        function toggleClustersMedium() {{
            if (clustersMediumVisible) {{
                map.removeLayer(clustersMediumLayer);
                clustersMediumVisible = false;
            }} else {{
                clustersMediumLayer.addTo(map);
                clustersMediumVisible = true;
            }}
        }}
        
        function toggleClustersLarge() {{
            if (clustersLargeVisible) {{
                map.removeLayer(clustersLargeLayer);
                clustersLargeVisible = false;
            }} else {{
                clustersLargeLayer.addTo(map);
                clustersLargeVisible = true;
            }}
        }}
        
        function toggleWardBoundary() {{
            if (boundaryVisible) {{
                map.removeLayer(boundaryLayer);
                boundaryVisible = false;
            }} else {{
                boundaryLayer.addTo(map);
                boundaryVisible = true;
            }}
        }}
        
        function toggleVisits() {{
            if (visitsVisible) {{
                map.removeLayer(visitsLayer);
                visitsVisible = false;
            }} else {{
                visitsLayer.addTo(map);
                visitsVisible = true;
            }}
        }}
        
        function toggleBuildings() {{
            if (buildingsVisible) {{
                map.removeLayer(buildingsLayer);
                buildingsVisible = false;
            }} else {{
                buildingsLayer.addTo(map);
                buildingsVisible = true;
            }}
        }}
        
        function resetView() {{
            const allLayers = [];
            if (gridVisible) allLayers.push(...gridLayer.getLayers());
            if (visitsVisible) allLayers.push(...visitsLayer.getLayers());
            if (buildingsVisible) allLayers.push(...buildingsLayer.getLayers());
            
            if (allLayers.length > 0) {{
                const group = new L.featureGroup(allLayers);
                map.fitBounds(group.getBounds().pad(0.1));
            }}
        }}
        
        setTimeout(resetView, 100);
    </script>
</body>
</html>"""
        
        return html_content
