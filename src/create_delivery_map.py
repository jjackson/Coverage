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
        flw = feature['properties']['flw_commcare_id']
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
            flw_commcare_id = str(feature['properties']['flw_commcare_id'])
            
            # Try to match by name first
            if flw_name in flw_colors:
                feature['properties']['color'] = flw_colors[flw_name]
            # Then try to match by ID if it's in the flw list
            elif flw_commcare_id in flw_colors:
                feature['properties']['color'] = flw_colors[flw_commcare_id]
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
            
            /* Info Panel Styles */
            #info-panel {{
                position: absolute;
                top: 10px;
                right: 10px;
                width: 300px;
                max-width: 35%;
                background: white;
                border-radius: 4px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                z-index: 1000;
                overflow: hidden;
                transition: all 0.3s ease;
            }}
            #info-panel.collapsed {{
                width: 40px;
                height: 40px;
            }}
            #info-panel-toggle {{
                position: absolute;
                top: 0;
                right: 0;
                width: 40px;
                height: 40px;
                background: #fff;
                color: #555;
                border: none;
                border-left: 1px solid #ccc;
                border-bottom: 1px solid #ccc;
                border-bottom-left-radius: 4px;
                cursor: pointer;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1001;
            }}
            #info-panel-toggle:hover {{
                background: #f0f0f0;
            }}
            #info-panel-content {{
                padding: 15px;
                max-height: 70vh;
                overflow-y: auto;
                display: block;
            }}
            .info-panel-section {{
                margin-bottom: 20px;
                border-bottom: 1px solid #eee;
                padding-bottom: 15px;
            }}
            .info-panel-section:last-child {{
                border-bottom: none;
                margin-bottom: 0;
            }}
            .info-panel-section h3 {{
                margin-top: 0;
                margin-bottom: 10px;
                color: #333;
                font-size: 16px;
                font-weight: bold;
            }}
            .info-panel-section p {{
                margin: 5px 0;
                color: #666;
                font-size: 14px;
            }}
            .info-panel-label {{
                font-weight: bold;
                display: inline-block;
                min-width: 120px;
            }}
            .info-panel-value {{
                display: inline-block;
            }}
            .info-panel-header {{
                padding: 10px 15px;
                background: #f5f5f5;
                border-bottom: 1px solid #ddd;
                font-weight: bold;
                position: relative;
            }}
            .info-panel-empty {{
                padding: 20px;
                text-align: center;
                color: #999;
                font-style: italic;
            }}
            .info-panel-section.hidden {{
                display: none;
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
        
        <div id="info-panel" class="collapsed">
            <div id="info-panel-header" class="info-panel-header">
                Information Panel
            </div>
            <button id="info-panel-toggle">&#8249;</button>
            <div id="info-panel-content">
                <div class="info-panel-empty">
                    Click on a Delivery Unit or Service Delivery Point to view details
                </div>
                
                <div id="service-point-section" class="info-panel-section hidden">
                    <h3>Service Delivery Point</h3>
                    <p><span class="info-panel-label">FLW:</span> <span id="sp-flw" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Visit Date:</span> <span id="sp-date" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Status:</span> <span id="sp-status" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Accuracy:</span> <span id="sp-accuracy" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Flagged:</span> <span id="sp-flagged" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Flag Reason:</span> <span id="sp-flag-reason" class="info-panel-value">-</span></p>
                </div>
                
                <div id="flw-section" class="info-panel-section hidden">
                    <h3>FLW (Field Worker)</h3>
                    <p><span class="info-panel-label">ID/Name:</span> <span id="flw-name" class="info-panel-value">-</span></p>
                </div>
                
                <div id="delivery-unit-section" class="info-panel-section hidden">
                    <h3>Delivery Unit</h3>
                    <p><span class="info-panel-label">Name:</span> <span id="du-name" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Status:</span> <span id="du-status" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Delivery Target:</span> <span id="du-target" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Delivery Count:</span> <span id="du-count" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Buildings:</span> <span id="du-buildings" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Checked Out:</span> <span id="du-checkout" class="info-panel-value">-</span></p>
                </div>
                
                <div id="service-area-section" class="info-panel-section hidden">
                    <h3>Service Area</h3>
                    <p><span class="info-panel-label">ID:</span> <span id="service-area-id" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Total DUs:</span> <span id="service-area-total-dus" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Completed DUs:</span> <span id="service-area-completed-dus" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Total Buildings:</span> <span id="service-area-buildings" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Completion Rate:</span> <span id="service-area-completion" class="info-panel-value">-</span></p>
                    <p><span class="info-panel-label">Total Service Deliveries:</span> <span id="service-area-deliveries" class="info-panel-value">-</span></p>
                </div>
            </div>
        </div>
        
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
            
            // Track the currently selected service area for highlighting
            let selectedServiceAreaId = null;
            
            // Create a layer group for boundary highlights
            const boundaryHighlightLayer = L.layerGroup().addTo(map);
            
            // Function to style features based on FLW, status, and if it's in the selected service area
            function style(feature) {{
                if (!feature || !feature.properties) return {{}}; // Safety check
                
                const status = feature.properties.du_status || 'unvisited';
                const inSelectedServiceArea = selectedServiceAreaId && 
                                             feature.properties.service_area_id === selectedServiceAreaId;
                
                // Base styling
                const baseStyle = {{
                    fillColor: feature.properties.color,
                    weight: 3, // Same width for all borders (3px)
                    opacity: 1,
                    color: inSelectedServiceArea ? 'black' : 'white',
                    dashArray: null, // Always solid borders for cleaner look
                    zIndex: inSelectedServiceArea ? 1000 : 500 // Higher z-index for selected areas
                }};
                
                // Apply different styling based on status
                if (status === 'completed') {{
                    // More solid/vibrant for completed
                    baseStyle.fillOpacity = 0.8;
                }} else if (status === 'visited') {{
                    // Medium transparency for visited
                    baseStyle.fillOpacity = 0.5;
                }} else {{
                    // More transparent for unvisited
                    baseStyle.fillOpacity = 0.3;
                }}
                
                return baseStyle;
            }}
            
            // Function to extract boundary coordinates from a GeoJSON feature
            function extractBoundaryCoordinates(feature) {{
                if (!feature || !feature.geometry || !feature.geometry.coordinates) return [];
                
                const coordinates = [];
                
                // Function to process coordinates array
                const processCoordinates = (coords) => {{
                    if (coords.length === 0) return;
                    
                    // Polygons in GeoJSON are arrays of linear rings
                    // The first ring is the outer boundary, the rest are holes
                    if (Array.isArray(coords[0]) && Array.isArray(coords[0][0])) {{
                        // This is a MultiPolygon or Polygon with holes
                        coords.forEach(ring => processCoordinates(ring));
                    }} else if (Array.isArray(coords[0]) && typeof coords[0][0] === 'number') {{
                        // This is a single linear ring (array of positions)
                        // Each position is [longitude, latitude]
                        coordinates.push(coords.map(pos => [pos[1], pos[0]])); // Swap to Leaflet's [lat, lng] format
                    }}
                }};
                
                processCoordinates(feature.geometry.coordinates);
                return coordinates;
            }}
            
            // Function to draw outer boundary for a service area
            function highlightServiceAreaBoundary(serviceAreaId) {{
                // Clear previous highlights
                boundaryHighlightLayer.clearLayers();
                
                if (!serviceAreaId || serviceAreaId === 'all') return;
                
                // Get all features in the selected service area
                const serviceAreaFeatures = geojsonData.features.filter(
                    feature => feature.properties.service_area_id === serviceAreaId
                );
                
                // Extract and draw boundaries for each feature
                serviceAreaFeatures.forEach(feature => {{
                    const boundaryRings = extractBoundaryCoordinates(feature);
                    
                    boundaryRings.forEach(ring => {{
                        // Create polyline for each boundary ring
                        const boundaryLine = L.polyline(ring, {{
                            color: 'black',
                            weight: 3.5, // Slightly thicker to ensure visibility
                            opacity: 1,
                            lineCap: 'square',
                            lineJoin: 'miter'
                        }});
                        
                        boundaryHighlightLayer.addLayer(boundaryLine);
                    }});
                }});
            }}
            
            // Function to update styles when service area selection changes
            function updateDUStyles() {{
                allFeatureLayers.forEach(feature => {{
                    if (feature.layer && feature.layer.getLayers) {{
                        feature.layer.getLayers().forEach(layer => {{
                            if (layer.setStyle) {{
                                layer.setStyle(style(layer.feature));
                            }}
                        }});
                    }}
                }});
                
                // Add the outer boundary highlight
                highlightServiceAreaBoundary(selectedServiceAreaId);
            }}
            
            // Create individual GeoJSON layers for each feature
            const allFeatureLayers = [];
            
            // Process all features and create individual layers
            geojsonData.features.forEach(feature => {{
                // Set default status if missing
                if (!feature.properties.du_status) {{
                    feature.properties.du_status = 'unvisited';
                }}
                
                const flw = feature.properties.flw_commcare_id;
                const status = feature.properties.du_status;
                
                // Create GeoJSON layer for this feature
                const geoJsonLayer = L.geoJSON(feature, {{
                    style: style,
                    onEachFeature: function(feature, layer) {{
                        // Popup content (we'll keep simple popup for quick reference)
                        let popupContent = `
                            <strong>Name:</strong> ${{feature.properties.name}}<br>
                            <strong>Service Area:</strong> ${{feature.properties.service_area_id}}<br>
                            <strong>FLW:</strong> ${{feature.properties.flw_commcare_id}}
                        `;
                        layer.bindPopup(popupContent);
                        
                        // Add click handler to update the info panel
                        layer.on('click', function() {{
                            // Expand the info panel if collapsed
                            if (document.getElementById('info-panel').classList.contains('collapsed')) {{
                                toggleInfoPanel();
                            }}
                            
                            // Update Delivery Unit section
                            document.getElementById('du-name').textContent = feature.properties.name;
                            document.getElementById('du-status').textContent = feature.properties.du_status || 'Unknown';
                            document.getElementById('du-target').textContent = feature.properties.delivery_target;
                            document.getElementById('du-count').textContent = feature.properties.delivery_count;
                            document.getElementById('du-buildings').textContent = feature.properties['#Buildings'];
                            document.getElementById('du-checkout').textContent = feature.properties.checked_out_date || 'Not checked out';
                            
                            // Update FLW section
                            document.getElementById('flw-name').textContent = feature.properties.flw_commcare_id;
                            
                            // Update Service Area section
                            const serviceAreaId = feature.properties.service_area_id;
                            document.getElementById('service-area-id').textContent = serviceAreaId;
                            
                            // Highlight this service area on the map
                            selectedServiceAreaId = serviceAreaId;
                            
                            // Update styles of all DUs to highlight this service area
                            updateDUStyles();
                            
                            // Calculate and display service area statistics
                            const areaStats = calculateServiceAreaStats(serviceAreaId);
                            document.getElementById('service-area-total-dus').textContent = areaStats.totalDUs;
                            document.getElementById('service-area-completed-dus').textContent = areaStats.completedDUs;
                            document.getElementById('service-area-buildings').textContent = areaStats.totalBuildings;
                            document.getElementById('service-area-completion').textContent = areaStats.completionRate;
                            document.getElementById('service-area-deliveries').textContent = areaStats.totalServiceDeliveries;
                            
                            // Show all sections
                            document.getElementById('service-point-section').classList.remove('hidden');
                            document.getElementById('flw-section').classList.remove('hidden');
                            document.getElementById('delivery-unit-section').classList.remove('hidden');
                            document.getElementById('service-area-section').classList.remove('hidden');
                            
                            // Reset service point values since we're viewing a DU
                            document.getElementById('sp-flw').textContent = '-';
                            document.getElementById('sp-date').textContent = '-';
                            document.getElementById('sp-status').textContent = '-';
                            document.getElementById('sp-accuracy').textContent = '-';
                            document.getElementById('sp-flagged').textContent = '-';
                            document.getElementById('sp-flag-reason').textContent = '-';
                            
                            // Hide the empty message
                            document.querySelector('.info-panel-empty').style.display = 'none';
                        }});
                    }}
                }}).addTo(map);
                
                // Store layer reference with metadata
                allFeatureLayers.push({{
                    layer: geoJsonLayer,
                    flw_commcare_id: flw,
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
                                html: '<div style="background-color: ' + markerColor + '; width: 10px; height: 10px; border-radius: 50%; border: 2px solid #000; z-index: 9999;"></div>',
                                iconSize: [14, 14],
                                iconAnchor: [7, 7]
                            }}),
                            zIndexOffset: 9000  // Extremely high z-index to ensure it stays on top
                        }}
                    );
                    
                    // Add popup with info
                    marker.bindPopup(
                        '<strong>FLW:</strong> ' + flwName + '<br>' +
                        '<strong>Visit Date:</strong> ' + (feature.properties.visit_date || 'N/A') + '<br>' +
                        '<strong>Visit ID:</strong> ' + (feature.properties.visit_id || 'N/A') + '<br>' +
                        '<strong>Accuracy:</strong> ' + (feature.properties.accuracy_in_m || 'N/A') + ' m<br>' +
                        '<strong>Status:</strong> ' + (feature.properties.status || 'N/A') + '<br>' +
                        '<strong>Delivery Unit:</strong> ' + (feature.properties.du_name || 'N/A') + '<br>' +
                        '<strong>Flagged:</strong> ' + (feature.properties.flagged ? 'Yes' : 'No') + '<br>' +
                        '<strong>Flag Reason:</strong> ' + (feature.properties.flag_reason || 'N/A')
                    );
                    
                    // Add to service points layer
                    marker.addTo(servicePointsLayer);
                    
                    // Add click handler to update info panel for service points
                    marker.on('click', function() {{
                        // Expand the info panel if collapsed
                        if (document.getElementById('info-panel').classList.contains('collapsed')) {{
                            toggleInfoPanel();
                        }}
                        
                                                    // Update Service Point section
                            document.getElementById('sp-flw').textContent = flwName;
                            document.getElementById('sp-date').textContent = feature.properties.visit_date || 'N/A';
                            document.getElementById('sp-status').textContent = feature.properties.status || 'N/A';
                            document.getElementById('sp-accuracy').textContent = (feature.properties.accuracy_in_m || 'N/A') + " m";
                            document.getElementById('sp-flagged').textContent = feature.properties.flagged ? 'Yes' : 'No';
                            document.getElementById('sp-flag-reason').textContent = feature.properties.flag_reason || 'N/A';
                            
                            // If this service point is in the selected service area, highlight it
                            if (feature.properties.service_area_id && feature.properties.service_area_id === selectedServiceAreaId) {{
                                // Add a special highlight class or styling if needed
                            }}
                        
                        // Variables to track if we found DU and service area information
                        let foundDU = false;
                        let serviceAreaId = null;
                        
                        // Update related Delivery Unit info if available
                        if (feature.properties.du_name) {{
                            const duName = feature.properties.du_name;
                            document.getElementById('du-name').textContent = duName;
                            foundDU = true;
                            
                            // Look for the DU in the geojson data to get full DU information
                            // and its service area
                            for (const du of geojsonData.features) {{
                                if (du.properties.name === duName) {{
                                    // Found the DU, update DU fields with full information
                                    document.getElementById('du-status').textContent = du.properties.du_status || 'Unknown';
                                    document.getElementById('du-target').textContent = du.properties.delivery_target || '-';
                                    document.getElementById('du-count').textContent = du.properties.delivery_count || '-';
                                    document.getElementById('du-buildings').textContent = du.properties['#Buildings'] || '-';
                                    document.getElementById('du-checkout').textContent = du.properties.checked_out_date || 'Not checked out';
                                    
                                    // Get the service area ID
                                    serviceAreaId = du.properties.service_area_id;
                                    break;
                                }}
                            }}
                        }}
                        
                        if (!foundDU) {{
                            // Reset DU values if no DU is associated
                            document.getElementById('du-name').textContent = '-';
                            document.getElementById('du-status').textContent = '-';
                            document.getElementById('du-target').textContent = '-';
                            document.getElementById('du-count').textContent = '-';
                            document.getElementById('du-buildings').textContent = '-';
                            document.getElementById('du-checkout').textContent = '-';
                        }}
                        
                        // Update FLW section
                        document.getElementById('flw-name').textContent = flwName;
                        
                        // First, try to use service_area_id from the feature itself
                        if (feature.properties.service_area_id) {{
                            serviceAreaId = feature.properties.service_area_id;
                        }}
                        
                        // Update Service Area section if we have a service area ID, otherwise reset
                        if (serviceAreaId) {{
                            document.getElementById('service-area-id').textContent = serviceAreaId;
                            
                            // Calculate and display service area statistics
                            const areaStats = calculateServiceAreaStats(serviceAreaId);
                            document.getElementById('service-area-total-dus').textContent = areaStats.totalDUs;
                            document.getElementById('service-area-completed-dus').textContent = areaStats.completedDUs;
                            document.getElementById('service-area-buildings').textContent = areaStats.totalBuildings;
                            document.getElementById('service-area-completion').textContent = areaStats.completionRate;
                            document.getElementById('service-area-deliveries').textContent = areaStats.totalServiceDeliveries;
                        }} else {{
                            // Reset all service area fields
                            document.getElementById('service-area-id').textContent = '-';
                            document.getElementById('service-area-total-dus').textContent = '-';
                            document.getElementById('service-area-completed-dus').textContent = '-';
                            document.getElementById('service-area-buildings').textContent = '-';
                            document.getElementById('service-area-completion').textContent = '-';
                            document.getElementById('service-area-deliveries').textContent = '-';
                        }}
                        
                        // Show all sections
                        document.getElementById('service-point-section').classList.remove('hidden');
                        document.getElementById('flw-section').classList.remove('hidden');
                        document.getElementById('delivery-unit-section').classList.remove('hidden');
                        document.getElementById('service-area-section').classList.remove('hidden');
                        
                        // Hide the empty message
                        document.querySelector('.info-panel-empty').style.display = 'none';
                    }});
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
                            if (feature.flw_commcare_id === flw) {{
                                feature.layer.addTo(map);
                            }}
                        }});
                    }} else {{
                        allFeatureLayers.forEach(feature => {{
                            if (feature.flw_commcare_id === flw) {{
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
                    const flw = feature.flw_commcare_id;
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
                
                // Update the global selected service area ID
                selectedServiceAreaId = selectedServiceArea === 'all' ? null : selectedServiceArea;
                
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
                    
                    // If showing all, fit to all features
                    map.fitBounds(L.geoJSON(geojsonData).getBounds());
                }} else {{
                    // Create a bounds object to calculate the extent of the selected service area
                    const bounds = L.latLngBounds();
                    let hasValidBounds = false;
                    
                    // Show only delivery units in the selected service area and check their FLWs
                    const relevantFlws = new Set();
                    
                    // Add only delivery units from the selected service area
                    allFeatureLayers.forEach(feature => {{
                        if (feature.serviceArea === selectedServiceArea) {{
                            feature.layer.addTo(map);
                            relevantFlws.add(feature.flw_commcare_id);
                            
                            // Add to bounds for zooming
                            if (feature.layer.getBounds) {{
                                bounds.extend(feature.layer.getBounds());
                                hasValidBounds = true;
                            }}
                        }}
                    }});
                    
                    // Check toggles for FLWs that have delivery units in this service area
                    relevantFlws.forEach(flw => {{
                        const toggle = document.querySelector(`.flw-toggle input[data-flw="${{flw}}"]`);
                        if (toggle) {{
                            toggle.checked = true;
                        }}
                    }});
                    
                    // Zoom to the selected service area if we have valid bounds
                    if (hasValidBounds) {{
                        map.fitBounds(bounds, {{ padding: [50, 50] }});
                    }}
                }}
                
                // Reset status filters when service area changes
                document.querySelectorAll('.status-toggle input').forEach(input => {{
                    input.checked = true;
                }});
                
                // Apply status filters with the new selection
                filterByStatus();
                
                // Update styles for all features and draw boundary highlights
                updateDUStyles();
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
            
            // Function to calculate service area statistics
            function calculateServiceAreaStats(serviceAreaId) {{
                // Initialize stats
                let totalDUs = 0;
                let totalBuildings = 0;
                let totalSurfaceArea = 0;
                let completedDUs = 0;
                let flwsInArea = new Set();
                let totalServiceDeliveries = 0;
                
                // Loop through all features to find those in this service area
                geojsonData.features.forEach(feature => {{
                    if (feature.properties.service_area_id === serviceAreaId) {{
                        totalDUs++;
                        
                        // Add to total buildings if available
                        if (feature.properties['#Buildings']) {{
                            totalBuildings += parseInt(feature.properties['#Buildings'], 10);
                        }}
                        
                        // Add to total surface area if available
                        if (feature.properties['Surface Area (sq. meters)']) {{
                            totalSurfaceArea += parseFloat(feature.properties['Surface Area (sq. meters)']);
                        }}
                        
                        // Count completed DUs
                        if (feature.properties.du_status === 'completed') {{
                            completedDUs++;
                        }}
                        
                        // Add delivery count if available
                        if (feature.properties.delivery_count) {{
                            totalServiceDeliveries += parseInt(feature.properties.delivery_count, 10);
                        }}
                        
                        // Add FLW to unique set
                        if (feature.properties.flw_commcare_id) {{
                            flwsInArea.add(feature.properties.flw_commcare_id);
                        }}
                    }}
                }});
                
                // Calculate completion rate
                const completionRate = totalDUs > 0 ? ((completedDUs / totalDUs) * 100).toFixed(1) : 0;
                
                // Return the stats object
                return {{
                    totalDUs: totalDUs,
                    completedDUs: completedDUs,
                    totalBuildings: totalBuildings,
                    totalSurfaceArea: totalSurfaceArea.toFixed(2),
                    totalServiceDeliveries: totalServiceDeliveries,
                    completionRate: completionRate + "%",
                    uniqueFLWs: flwsInArea.size
                }};
            }}
            
            // Info panel toggle functionality
            function toggleInfoPanel() {{
                const panel = document.getElementById('info-panel');
                const toggleBtn = document.getElementById('info-panel-toggle');
                
                panel.classList.toggle('collapsed');
                
                // Change toggle button text based on state
                if (panel.classList.contains('collapsed')) {{
                    toggleBtn.innerHTML = '&#8250;'; // Right arrow
                }} else {{
                    toggleBtn.innerHTML = '&#8249;'; // Left arrow
                }}
            }}
            
            // Add event listener to toggle button
            document.getElementById('info-panel-toggle').addEventListener('click', toggleInfoPanel);
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
