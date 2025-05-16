import pandas as pd
import geopandas as gpd
from shapely import wkt
import json
import numpy as np
import random
import os
import glob
import argparse

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .models import CoverageData, DeliveryUnit, ServiceDeliveryPoint
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData, DeliveryUnit, ServiceDeliveryPoint

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def generate_contrasting_colors(n):
    # Predefined high-contrast color palette
    base_colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",  # Primary and secondary
        "#FF8000", "#FF0080", "#80FF00", "#00FF80", "#8000FF", "#0080FF",  # Tertiary
        "#804000", "#FF4080", "#8080FF", "#80FF80", "#804080", "#408080",  # Mixed tones
        "#FF8080", "#80FF40", "#4080FF", "#FF4040", "#40FF40", "#4040FF",  # Pastel-ish
        "#000000", "#555555", "#888888", "#BBBBBB",                        # Grayscale
        "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",  # Dark tones
    ]
    
    # If we need more colors than we have predefined
    if n > len(base_colors):
        # Generate random distinct colors for the remaining ones
        existing_colors = set(base_colors)
        while len(existing_colors) < n:
            # Generate random RGB with sufficient distance from existing colors
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            # Convert to hex
            color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            
            # Check if the color is sufficiently different from existing ones
            # by measuring Euclidean distance in RGB space
            min_distance = float('inf')
            for existing in existing_colors:
                # Convert hex to RGB
                er = int(existing[1:3], 16)
                eg = int(existing[3:5], 16)
                eb = int(existing[5:7], 16)
                # Calculate distance
                distance = ((r - er) ** 2 + (g - eg) ** 2 + (b - eb) ** 2) ** 0.5
                min_distance = min(min_distance, distance)
            
            # Add if sufficiently different
            if min_distance > 100:  # Threshold for difference
                existing_colors.add(color)
        
        colors = list(existing_colors)
    else:
        colors = base_colors[:n]
    
    # Shuffle to ensure adjacent items in the list get different colors
    random.shuffle(colors)
    return colors

def create_leaflet_map(excel_file=None, service_delivery_csv=None, coverage_data=None):
    """
    Create a leaflet map from the DU Export Excel file and service delivery GPS coordinates
    or directly from a CoverageData object.
    
    Args:
        excel_file: Path to the DU Export Excel file (not needed if coverage_data is provided)
        service_delivery_csv: Optional path to service delivery GPS coordinates CSV (not needed if coverage_data is provided)
        coverage_data: Optional CoverageData object containing the already loaded data
    """
    # Either use provided coverage_data or load from files
    if coverage_data is None:
        if excel_file is None:
            raise ValueError("Either coverage_data or excel_file must be provided")
        # Load data using the CoverageData model
        coverage_data = CoverageData.from_excel_and_csv(excel_file, service_delivery_csv)
    
    # Use CoverageData's convenience methods to create GeoDataFrames
    gdf = coverage_data.create_delivery_units_geodataframe()
    service_points = coverage_data.create_service_points_geodataframe()
    
    # Debug prints to check the geodataframe content
    # print(f"Delivery units GeoDataFrame shape: {gdf.shape}")
    # print(f"Delivery units GeoDataFrame columns: {gdf.columns.tolist()}")
    # print(f"Sample of first row: {gdf.iloc[0] if len(gdf) > 0 else 'Empty DataFrame'}")
    # print(f"Number of geometries: {sum(1 for g in gdf['geometry'] if g is not None)}")
    
    # Use precomputed data from CoverageData object
    service_areas = coverage_data.unique_service_area_ids
    flws = coverage_data.unique_flw_names
    status_values = coverage_data.unique_status_values
    
    # Debug print
    # print(f"Unique status values: {status_values}")
    
    # Create a color for each FLW
    flw_colors = {flw: color for flw, color in zip(flws, generate_contrasting_colors(len(flws)))}
    
    # Convert to GeoJSON and handle numpy types
    geojson_str = gdf.to_json()
    geojson_data = json.loads(geojson_str)
    
    # Add color property based on FLW
    for i, feature in enumerate(geojson_data['features']):
        # Get FLW for this feature
        flw = feature['properties']['flw']
        feature['properties']['color'] = flw_colors[flw]
        
        # Convert all numpy types in properties to Python native types
        for key, value in list(feature['properties'].items()):
            feature['properties'][key] = convert_to_serializable(value)
    
    # Prepare service points GeoJSON if available
    service_points_geojson = None
    if service_points is not None:
        service_points_geojson = json.loads(service_points.to_json())
        
        # Add color property to service points based on FLW
        for feature in service_points_geojson['features']:
            flw_name = feature['properties']['flw_name']
            flw_id = str(feature['properties']['flw_id'])
            
            # Try to match by name first
            if flw_name in flw_colors:
                feature['properties']['color'] = flw_colors[flw_name]
            # Then try to match by ID if it's in the flw list
            elif flw_id in flw_colors:
                feature['properties']['color'] = flw_colors[flw_id]
            else:
                # Assign a new color if this FLW isn't in our dictionary yet
                new_color = generate_contrasting_colors(1)[0]
                flw_colors[flw_name] = new_color
                feature['properties']['color'] = new_color
            
            # Convert all numpy types in properties to Python native types
            for key, value in list(feature['properties'].items()):
                feature['properties'][key] = convert_to_serializable(value)
    
    
    # Create HTML with embedded Leaflet map
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nigeria Delivery Units Map</title>
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
                height: 92vh;
            }}
            .info {{
                padding: 6px 8px;
                font: 14px/16px Arial, Helvetica, sans-serif;
                background: white;
                background: rgba(255,255,255,0.8);
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                max-width: 250px;
            }}
            .info h4 {{
                margin: 0 0 5px;
                color: #777;
            }}
            .legend {{
                line-height: 18px;
                color: #555;
                max-height: 300px;
                overflow-y: auto;
            }}
            .legend i {{
                width: 18px;
                height: 18px;
                float: left;
                margin-right: 8px;
                opacity: 0.7;
            }}
            .controls {{
                padding: 10px;
                background: white;
                background: rgba(255,255,255,0.8);
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                margin-bottom: 10px;
            }}
            #control-panel {{
                background: white;
                padding: 10px;
                height: auto;
                min-height: 8vh;
                max-height: 25vh;
                display: flex;
                flex-wrap: wrap;
                border-bottom: 1px solid #ccc;
                overflow-y: auto;
            }}
            .row {{
                display: flex;
                flex-wrap: wrap;
                width: 100%;
                margin-bottom: 10px;
            }}
            .control-section {{
                margin-right: 15px;
                margin-bottom: 10px;
                min-width: 200px;
                flex: 1;
            }}
            .section-title {{
                margin-top: 0;
                margin-bottom: 5px;
                font-weight: bold;
            }}
            .flw-toggle {{
                display: inline-block;
                margin-right: 5px;
                margin-bottom: 5px;
                cursor: pointer;
                padding: 2px 5px;
                border-radius: 3px;
                border: 1px solid #ddd;
                background: #f9f9f9;
            }}
            .flw-toggle:hover {{
                background: #f0f0f0;
            }}
            .flw-toggle input {{
                margin-right: 5px;
                vertical-align: middle;
            }}
            #flw-toggles {{
                max-height: 8vh;
                overflow-y: auto;
                display: flex;
                flex-wrap: wrap;
                align-content: flex-start;
                margin-bottom: 5px;
            }}
            .flw-toggle-controls {{
                display: flex;
                margin-bottom: 5px;
                gap: 10px;
            }}
            .toggle-all-btn {{
                padding: 3px 8px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                cursor: pointer;
                font-size: 12px;
            }}
            .toggle-all-btn:hover {{
                background-color: #e0e0e0;
            }}
            #service-area-filter {{
                min-width: 200px;
                flex: 1;
            }}
            .color-swatch {{
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 5px;
                border: 1px solid #999;
                vertical-align: middle;
            }}
            select {{
                padding: 5px;
                border-radius: 3px;
                border: 1px solid #ccc;
                background: #f9f9f9;
                width: 100%;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 3px;
            }}
            .legend-color {{
                width: 15px;
                height: 15px;
                margin-right: 8px;
                display: inline-block;
                border: 1px solid rgba(0,0,0,0.2);
            }}
            .layer-toggle {{
                margin-top: 10px;
                padding: 5px 0;
                border-top: 1px solid #eee;
            }}
            .status-toggle-container {{
                margin-top: 10px;
                padding-top: 5px;
            }}
            #status-toggles {{
                display: flex;
                flex-wrap: wrap;
            }}
            .status-toggle {{
                display: inline-block;
                margin-right: 10px;
                margin-bottom: 5px;
                cursor: pointer;
                padding: 4px 8px;
                border-radius: 3px;
                border: 1px solid #ddd;
                background: #f0f0f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div id="control-panel">
            <div class="row">
                <div class="control-section" style="width: 100%;">
                    <div class="section-title">FLW Toggle Controls</div>
                    <div class="flw-toggle-controls">
                        <button id="select-all-flw" class="toggle-all-btn">Select All FLWs</button>
                        <button id="deselect-all-flw" class="toggle-all-btn">Deselect All FLWs</button>
                    </div>
                    <div id="flw-toggles">
                        <!-- FLW toggles will be added here -->
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="control-section">
                    <div class="section-title">Status Filters</div>
                    <div id="status-toggles">
                        <!-- Status toggles will be added here -->
                    </div>
                </div>
                
                <div class="control-section" id="service-area-filter">
                    <div class="section-title">Filter by Service Area</div>
                    <select id="service-area-select">
                        <option value="all">All Service Areas</option>
                        <!-- Service area options will be added here -->
                    </select>
                </div>
                
                <div class="control-section">
                    <div class="section-title">Layer Controls</div>
                    <div class="layer-toggle">
                        <label>
                            <input type="checkbox" id="toggle-delivery-points" checked> 
                            Show Service Delivery Points
                        </label>
                    </div>
                </div>
            </div>
        </div>
        <div id="map"></div>
        
        <script>
            // Define the GeoJSON data with all delivery units
            const geojsonData = {json.dumps(geojson_data)};
            
            // Define service points data if available
            const servicePointsData = {json.dumps(service_points_geojson) if service_points_geojson else 'null'};
            
            // Define unique service areas
            const serviceAreas = {json.dumps(service_areas)};
            
            // Define unique FLWs
            const flws = {json.dumps(flws)};
            
            // Define unique statuses
            const statuses = {json.dumps(status_values)};
            
            // Define colors for FLWs
            const flwColors = {json.dumps(flw_colors)};
            
            // Initialize the map
            const map = L.map('map').setView([11.8, 13.15], 13);  // Centered on Nigeria (approximate)
            
            // Add OpenStreetMap tile layer
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);
            
            // No need for FLW or status layer groups anymore
            // Each feature has its own layer that we'll manage directly
            
            // Create a layer for service points
            const servicePointsLayer = L.layerGroup().addTo(map);
            
            // Function to style features based on FLW and status
            function style(feature) {{
                const status = feature.properties.du_status || 'unvisited';
                
                // Base styling
                const baseStyle = {{
                    fillColor: feature.properties.color,
                    weight: 2,
                    opacity: 1,
                    color: 'white',
                    dashArray: '3'
                }};
                
                // Apply different styling based on status
                if (status === 'completed') {{
                    // More solid/vibrant for completed
                    baseStyle.fillOpacity = 0.8;
                    baseStyle.weight = 3;
                    baseStyle.dashArray = null;
                }} else if (status === 'visited') {{
                    // Medium transparency for visited
                    baseStyle.fillOpacity = 0.5;
                    baseStyle.dashArray = '5';
                }} else {{
                    // More transparent for unvisited
                    baseStyle.fillOpacity = 0.3;
                    baseStyle.dashArray = '5,3';
                }}
                
                return baseStyle;
            }}
            
            // Create individual GeoJSON layers for each feature
            const allFeatureLayers = [];
            
            // Process all features and create individual layers
            geojsonData.features.forEach(feature => {{
                // Set default status if missing
                if (!feature.properties.du_status) {{
                    feature.properties.du_status = 'unvisited';
                }}
                
                const flw = feature.properties.flw;
                const status = feature.properties.du_status;
                
                // Create GeoJSON layer for this feature
                const geoJsonLayer = L.geoJSON(feature, {{
                    style: style,
                    onEachFeature: function(feature, layer) {{
                        // Popup content
                        let popupContent = `
                            <strong>Name:</strong> ${{feature.properties.name}}<br>
                            <strong>Service Area:</strong> ${{feature.properties.service_area_id}}<br>
                            <strong>FLW (Owner ID):</strong> ${{feature.properties.flw}}<br>
                            <strong>Status:</strong> ${{feature.properties.du_status}}<br>
                            <strong>Delivery Target:</strong> ${{feature.properties.delivery_target}}<br>
                            <strong>Delivery Count:</strong> ${{feature.properties.delivery_count}}<br>
                            <strong>Buildings:</strong> ${{feature.properties['#Buildings']}}<br>
                            <strong>Surface Area:</strong> ${{feature.properties['Surface Area (sq. meters)'].toFixed(2)}} m²<br>
                            <strong>Checkout Remark:</strong> ${{feature.properties.du_checkout_remark || 'N/A'}}<br>
                            <strong>Checked Out Date:</strong> ${{feature.properties.checked_out_date || 'N/A'}}
                        `;
                        layer.bindPopup(popupContent);
                    }}
                }}).addTo(map);
                
                // Store layer reference with metadata
                allFeatureLayers.push({{
                    layer: geoJsonLayer,
                    flw: flw,
                    status: status,
                    serviceArea: feature.properties.service_area_id
                }});
            }});
            
            // Add service points if available
            if (servicePointsData) {{
                servicePointsData.features.forEach(feature => {{
                    const flwName = feature.properties.flw_name;
                    const markerColor = feature.properties.color || '#FF0000';
                    
                    // Create a regular marker for each service point with a custom icon for better visibility
                    const marker = L.marker(
                        [feature.geometry.coordinates[1], feature.geometry.coordinates[0]], 
                        {{
                            icon: L.divIcon({{
                                className: 'service-point-marker',
                                html: `<div style="background-color: ${{markerColor}}; width: 10px; height: 10px; border-radius: 50%; border: 2px solid #000; z-index: 9999;"></div>`,
                                iconSize: [14, 14],
                                iconAnchor: [7, 7]
                            }}),
                            zIndexOffset: 9000  // Extremely high z-index to ensure it stays on top
                        }}
                    );
                    
                    // Add popup with info
                    marker.bindPopup(`
                        <strong>FLW:</strong> ${{flwName}}<br>
                        <strong>Visit Date:</strong> ${{feature.properties.visit_date || 'N/A'}}<br>
                        <strong>Visit ID:</strong> ${{feature.properties.visit_id || 'N/A'}}<br>
                        <strong>Accuracy:</strong> ${{feature.properties.accuracy_in_m || 'N/A'}} m<br>
                        <strong>Status:</strong> ${{feature.properties.status || 'N/A'}}<br>
                        <strong>Delivery Unit:</strong> ${{feature.properties.du_name || 'N/A'}}<br>
                        <strong>Flagged:</strong> ${{feature.properties.flagged ? 'Yes' : 'No'}}<br>
                        <strong>Flag Reason:</strong> ${{feature.properties.flag_reason || 'N/A'}}
                    `);
                    
                    // Add to service points layer
                    marker.addTo(servicePointsLayer);
                }});
            }}
            
            // Fit map to all features
            map.fitBounds(L.geoJSON(geojsonData).getBounds());
            
            // Add FLW toggle controls
            const flwTogglesContainer = document.getElementById('flw-toggles');
            flws.forEach(flw => {{
                const label = document.createElement('label');
                label.className = 'flw-toggle';
                
                const input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = true;
                input.dataset.flw = flw;
                
                const colorSwatch = document.createElement('span');
                colorSwatch.className = 'color-swatch';
                colorSwatch.style.backgroundColor = flwColors[flw];
                
                input.addEventListener('change', function() {{
                    const flw = this.dataset.flw;
                    if (this.checked) {{
                        allFeatureLayers.forEach(feature => {{
                            if (feature.flw === flw) {{
                                feature.layer.addTo(map);
                            }}
                        }});
                    }} else {{
                        allFeatureLayers.forEach(feature => {{
                            if (feature.flw === flw) {{
                                map.removeLayer(feature.layer);
                            }}
                        }});
                    }}
                    
                    // Update status toggles to reflect current state
                    updateStatusToggles();
                }});
                
                label.appendChild(input);
                label.appendChild(colorSwatch);
                label.appendChild(document.createTextNode(flw));
                
                flwTogglesContainer.appendChild(label);
            }});
            
            // Add status toggle controls
            const statusTogglesContainer = document.getElementById('status-toggles');
            
            // Create status toggle function
            function createStatusToggles() {{
                // Clear existing toggles
                statusTogglesContainer.innerHTML = '';
                
                // Add a toggle for each status
                statuses.forEach(status => {{
                    const label = document.createElement('label');
                    label.className = 'status-toggle';
                    
                    const input = document.createElement('input');
                    input.type = 'checkbox';
                    input.checked = true;
                    input.dataset.status = status;
                    
                    input.addEventListener('change', function() {{
                        filterByStatus();
                    }});
                    
                    label.appendChild(input);
                    label.appendChild(document.createTextNode(status));
                    
                    statusTogglesContainer.appendChild(label);
                }});
            }}
            
            // Create initial status toggles
            createStatusToggles();
            
            // Add event handlers for select all and deselect all buttons
            document.getElementById('select-all-flw').addEventListener('click', function() {{
                // Check all FLW toggles
                document.querySelectorAll('.flw-toggle input').forEach(input => {{
                    if (!input.checked) {{
                        input.checked = true;
                        // Add the corresponding FLW layers to the map
                        const flw = input.dataset.flw;
                        allFeatureLayers.forEach(feature => {{
                            if (feature.flw === flw) {{
                                feature.layer.addTo(map);
                            }}
                        }});
                    }}
                }});
                // Update status filters
                filterByStatus();
            }});
            
            document.getElementById('deselect-all-flw').addEventListener('click', function() {{
                // Uncheck all FLW toggles
                document.querySelectorAll('.flw-toggle input').forEach(input => {{
                    if (input.checked) {{
                        input.checked = false;
                        // Remove the corresponding FLW layers from the map
                        const flw = input.dataset.flw;
                        allFeatureLayers.forEach(feature => {{
                            if (feature.flw === flw) {{
                                map.removeLayer(feature.layer);
                            }}
                        }});
                    }}
                }});
                // Update status filters
                filterByStatus();
            }});
            
            // Function to update status toggles based on visible FLWs
            function updateStatusToggles() {{
                // This function will be called when FLW toggles change
                // For now, we'll just recreate all status toggles
                createStatusToggles();
            }}
            
            // Function to filter features by status
            function filterByStatus() {{
                // Get all checked status toggles
                const checkedStatuses = Array.from(document.querySelectorAll('.status-toggle input:checked'))
                    .map(input => input.dataset.status);
                
                // Get all checked FLWs
                const checkedFlws = Array.from(document.querySelectorAll('.flw-toggle input:checked'))
                    .map(input => input.dataset.flw);
                
                // Get the selected service area
                const selectedServiceArea = document.getElementById('service-area-select').value;
                
                // Apply filtering to each feature layer
                allFeatureLayers.forEach(feature => {{
                    const flw = feature.flw;
                    const status = feature.status;
                    const serviceArea = feature.serviceArea;
                    
                    // Check if this feature meets all criteria:
                    // 1. FLW is selected
                    // 2. Status is selected
                    // 3. Either all service areas are selected OR this feature is in the selected service area
                    const shouldBeVisible = 
                        checkedFlws.includes(flw) && 
                        checkedStatuses.includes(status) &&
                        (selectedServiceArea === 'all' || serviceArea === selectedServiceArea);
                    
                    // Add or remove from map accordingly
                    if (shouldBeVisible) {{
                        // Make visible if it should be shown
                        feature.layer.addTo(map);
                    }} else {{
                        // Hide if it shouldn't be shown
                        map.removeLayer(feature.layer);
                    }}
                }});
            }}
            
            // Add service points toggle control
            const toggleServicePoints = document.getElementById('toggle-delivery-points');
            toggleServicePoints.addEventListener('change', function() {{
                if (this.checked) {{
                    servicePointsLayer.addTo(map);
                }} else {{
                    map.removeLayer(servicePointsLayer);
                }}
            }});
            
            // Add service area options
            const serviceAreaSelect = document.getElementById('service-area-select');
            serviceAreas.forEach(id => {{
                const option = document.createElement('option');
                option.value = id;
                option.textContent = `Service Area ${{id}}`;
                serviceAreaSelect.appendChild(option);
            }});
            
            // Function to filter by service area
            serviceAreaSelect.addEventListener('change', function() {{
                const selectedServiceArea = this.value;
                
                // First hide all FLW layers
                allFeatureLayers.forEach(feature => {{
                    map.removeLayer(feature.layer);
                }});
                
                // Uncheck all FLW toggles
                document.querySelectorAll('.flw-toggle input').forEach(input => {{
                    input.checked = false;
                }});
                
                if (selectedServiceArea === 'all') {{
                    // Show all FLWs
                    document.querySelectorAll('.flw-toggle input').forEach(input => {{
                        input.checked = true;
                    }});
                    
                    // Add all feature layers to the map
                    allFeatureLayers.forEach(feature => {{
                        feature.layer.addTo(map);
                    }});
                }} else {{
                    // Show only delivery units in the selected service area and check their FLWs
                    const relevantFlws = new Set();
                    
                    // Add only delivery units from the selected service area
                    allFeatureLayers.forEach(feature => {{
                        if (feature.serviceArea === selectedServiceArea) {{
                            feature.layer.addTo(map);
                            relevantFlws.add(feature.flw);
                        }}
                    }});
                    
                    // Check toggles for FLWs that have delivery units in this service area
                    relevantFlws.forEach(flw => {{
                        const toggle = document.querySelector(`.flw-toggle input[data-flw="${{flw}}"]`);
                        if (toggle) {{
                            toggle.checked = true;
                        }}
                    }});
                }}
                
                // Reset status filters when service area changes
                document.querySelectorAll('.status-toggle input').forEach(input => {{
                    input.checked = true;
                }});
                
                // Apply status filters with the new selection
                filterByStatus();
            }});
            
            // Add legend
            const legend = L.control({{position: 'bottomright'}});
            legend.onAdd = function(map) {{
                const div = L.DomUtil.create('div', 'info legend');
                

                
                // Add legend for service points
                if (servicePointsData) {{
                    div.innerHTML += '<h4 style="margin-top: 10px;">Map Symbols</h4>';
                    div.innerHTML += `
                        <div class="legend-item">
                            <svg height="20" width="20" style="margin-right: 8px;">
                                <circle cx="10" cy="10" r="5" stroke="black" stroke-width="1" fill="#FF0000" />
                            </svg>
                            Service Delivery Point
                        </div>
                    `;
                }}
                
                return div;
            }};
            legend.addTo(map);
        </script>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    output_filename = "nigeria_delivery_units_map.html"
    with open(output_filename, "w") as f:
        f.write(html_content)
    
    print(f"Map has been created: {output_filename}")
    return output_filename

# Execute the function if run directly
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create a delivery map from Excel and CSV data")
    parser.add_argument("--excel", help="Excel file containing delivery unit data")
    parser.add_argument("--csv", help="CSV file containing service delivery data")
    args = parser.parse_args()
    
    excel_file = None
    delivery_csv = None
    
    # If arguments are provided, use them
    if args.excel and args.csv:
        excel_file = args.excel
        delivery_csv = args.csv
        
        print(f"\nCreating map using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the map
        create_leaflet_map(excel_file=excel_file, service_delivery_csv=delivery_csv)
    else:
        # Original interactive selection
        # Get all files in the current directory
        files = glob.glob('*.*')
        
        # Filter for Excel and CSV files
        excel_files = [f for f in files if f.lower().endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        
        # Handle Excel file selection
        if len(excel_files) == 0:
            print("No Excel files found in the current directory.")
            exit(1)
        elif len(excel_files) == 1:
            excel_choice = 1
            print(f"\nAutomatically selected the only available Excel file: {excel_files[0]}")
        else:
            # Display Excel files
            print("\nAvailable Excel files for microplanning:")
            for i, file in enumerate(excel_files, 1):
                print(f"{i}. {file}")
            
            # Get user selection for Excel file
            excel_choice = 0
            while excel_choice < 1 or excel_choice > len(excel_files):
                try:
                    excel_choice = int(input(f"\nEnter the number for the microplanning Excel file (1-{len(excel_files)}): "))
                except ValueError:
                    print("Please enter a valid number.")
        
        # Handle CSV file selection
        if len(csv_files) == 0:
            print("No CSV files found in the current directory.")
            exit(1)
        elif len(csv_files) == 1:
            csv_choice = 1
            print(f"\nAutomatically selected the only available CSV file: {csv_files[0]}")
        else:
            # Display CSV files
            print("\nAvailable CSV files for service delivery data:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            
            # Get user selection for CSV file
            csv_choice = 0
            while csv_choice < 1 or csv_choice > len(csv_files):
                try:
                    csv_choice = int(input(f"\nEnter the number for the service delivery CSV file (1-{len(csv_files)}): "))
                except ValueError:
                    print("Please enter a valid number.")
        
        # Get selected files
        excel_file = excel_files[excel_choice - 1]
        delivery_csv = csv_files[csv_choice - 1]
        
        print(f"\nCreating map using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the map
        create_leaflet_map(excel_file=excel_file, service_delivery_csv=delivery_csv)
