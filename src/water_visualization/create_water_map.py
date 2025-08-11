"""
Water Points Map Generator
Creates interactive Leaflet maps from CommCare water survey data with on-demand image loading.
"""
import os
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from waterpoint import WaterPoint


def find_projects(water_data_path: str = "../../water_data") -> Dict[str, Dict[str, str]]:
    """Find all water survey projects in the water_data directory."""
    projects = {}
    
    water_data_dir = Path(water_data_path)
    if not water_data_dir.exists():
        return projects
    
    # Look for Excel files and matching image directories
    for excel_file in water_data_dir.glob("*.xlsx"):
        filename = excel_file.stem
        project_name = None
        
        # Handle new filename patterns: "{PROJECT} Final Waterbody Data.xlsx" or "{PROJECT} Final Waterbody data.xlsx"
        if "Final Waterbody" in filename:
            if "Final Waterbody Data" in filename:
                project_name = filename.split(" Final Waterbody Data")[0]
            elif "Final Waterbody data" in filename:
                project_name = filename.split(" Final Waterbody data")[0]
        # Handle old filename pattern: "{PROJECT} CCC Waterbody Survey - August 7.xlsx"
        elif "CCC Waterbody Survey" in filename:
            project_name = filename.split(" CCC Waterbody Survey")[0]
        
        if project_name:
            # Find matching image directory - try multiple patterns
            image_dirs = []
            
            # Try new pattern: "{PROJECT} Waterbody Survey/" or "{PROJECT} Waterbody survey/"
            image_dirs.extend(water_data_dir.glob(f"{project_name} Waterbody Survey*"))
            image_dirs.extend(water_data_dir.glob(f"{project_name} Waterbody survey*"))
            
            # Try old pattern: "{PROJECT} Pics*"
            if not image_dirs:
                image_dirs.extend(water_data_dir.glob(f"{project_name} Pics*"))
            
            if image_dirs:
                projects[project_name] = {
                    "excel_path": str(excel_file),
                    "images_path": str(image_dirs[0])
                }
    
    return projects


def load_water_points(project_name: str, water_data_path: str = "../../water_data") -> List[WaterPoint]:
    """Load water points from Excel file for a specific project."""
    projects = find_projects(water_data_path)
    
    if project_name not in projects:
        raise ValueError(f"Project '{project_name}' not found. Available projects: {list(projects.keys())}")
    
    project_info = projects[project_name]
    excel_path = project_info["excel_path"]
    images_path = project_info["images_path"]
    
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Convert each row to WaterPoint
    water_points = []
    for _, row in df.iterrows():
        try:
            water_point = WaterPoint.from_excel_row(row, images_path)
            water_points.append(water_point)
        except Exception as e:
            print(f"Error processing row {row.get('number', '?')}: {e}")
            continue
    
    return water_points



def create_small_thumbnail(image_path: str, max_width: int = 60) -> Optional[str]:
    """Create a small base64-encoded thumbnail for popup preview."""
    try:
        if not os.path.exists(image_path):
            return None
            
        from PIL import Image
        import io
        import base64
        
        with Image.open(image_path) as img:
            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            if width > max_width:
                new_height = int(height * max_width / width)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save to bytes with heavy compression for small size
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=60, optimize=True)
            
            # Encode as base64
            img_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_data}"
            
    except Exception as e:
        print(f"Warning: Could not create thumbnail for {os.path.basename(image_path)}: {e}")
        return None


def generate_popup_html(water_point: WaterPoint, output_dir: str) -> str:
    """Generate HTML content for marker popup."""
    
    # Create image gallery with small thumbnails
    images_html = ""
    available_photos = water_point.available_photos
    if available_photos:
        photo_items = []
        for i, photo_path in enumerate(available_photos):
            photo_filename = os.path.basename(photo_path)
            # Copy image to output directory and use relative path
            relative_path = f"images/{photo_filename}"
            
            # Ensure images directory exists
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy image file
            import shutil
            dest_path = os.path.join(images_dir, photo_filename)
            if not os.path.exists(dest_path):
                try:
                    shutil.copy2(photo_path, dest_path)
                except Exception as e:
                    print(f"Warning: Could not copy {photo_path}: {e}")
                    continue
            
            # Create small thumbnail for popup
            thumbnail_data = create_small_thumbnail(photo_path, max_width=60)
            
            if thumbnail_data:
                photo_items.append(f'''
                    <div class="photo-item" onclick="openLightbox('{relative_path}', '{photo_filename}')">
                        <img src="{thumbnail_data}" class="photo-thumbnail" alt="Photo {i+1}">
                        <div class="photo-name">Photo {i+1}</div>
                    </div>
                ''')
            else:
                # Fallback to placeholder if thumbnail creation fails
                photo_items.append(f'''
                    <div class="photo-item" onclick="openLightbox('{relative_path}', '{photo_filename}')">
                        <div class="photo-placeholder">
                            📷 Photo {i+1}
                        </div>
                        <div class="photo-name">Photo {i+1}</div>
                    </div>
                ''')
        
        if photo_items:
            images_html = f'''
                <div class="image-gallery">
                    <strong>Photos ({len(photo_items)}):</strong><br>
                    <div class="photo-grid">
                        {"".join(photo_items)}
                    </div>
                </div>
            '''
    
    # Create characteristics list
    characteristics = []
    characteristics.append(f"<strong>Type:</strong> {water_point.water_point_type_display}")
    characteristics.append(f"<strong>Usage:</strong> {water_point.usage_level_display}")
    
    if water_point.is_piped:
        characteristics.append("<strong>Piped:</strong> Yes")
    
    if water_point.has_dispenser:
        characteristics.append("<strong>Has Dispenser:</strong> Yes")
        if water_point.chlorine_dispenser_functional is not None:
            functional = "Yes" if water_point.chlorine_dispenser_functional else "No"
            characteristics.append(f"<strong>Dispenser Functional:</strong> {functional}")
    
    if water_point.other_treatment:
        characteristics.append(f"<strong>Other Treatment:</strong> {water_point.other_treatment}")
    
    characteristics_html = "<br>".join(characteristics)
    
    # Create notes section
    notes_html = ""
    if water_point.notes:
        notes_html = f'''
            <div class="notes">
                <strong>Notes:</strong><br>
                <em>{water_point.notes}</em>
            </div>
        '''
    
    popup_html = f'''
        <div class="water-point-popup">
            <h3>{water_point.community}</h3>
            <div class="location-breadcrumb">
                📍 {water_point.location_breadcrumb}
            </div>
            <hr>
            <div class="characteristics">
                {characteristics_html}
            </div>
            {images_html}
            {notes_html}
            <div class="metadata">
                <small>
                    <strong>Survey:</strong> {water_point.time_of_visit.strftime('%Y-%m-%d %H:%M')}<br>
                    <strong>Collector:</strong> {water_point.username} ({water_point.project_name})
                </small>
            </div>
        </div>
    '''
    
    return popup_html


def get_marker_color(water_point: WaterPoint) -> str:
    """Get marker color based on water point type."""
    color_map = {
        'piped_water': '#2E86AB',               # Blue
        'borehole_hand_pump': '#A23B72',       # Purple
        'borehole_motorized_pump': '#8E44AD',   # Dark Purple
        'protected_wells': '#F18F01',           # Orange
        'well': '#E67E22',                      # Light Orange
        'surface_water': '#8B4513',             # Brown
        'storage_tank_tap_stand': '#27AE60',    # Green
        'other': '#95A5A6'                      # Gray
    }
    return color_map.get(water_point.water_point_type, '#666666')


def generate_html_map(water_points: List[WaterPoint], output_dir: str, title: str = "Water Points Map") -> str:
    """Generate complete HTML map with embedded data and styling."""
    
    # Prepare data for JavaScript
    markers_data = []
    for wp in water_points:
        marker_data = wp.to_dict()
        marker_data['popup_html'] = generate_popup_html(wp, output_dir)
        marker_data['marker_color'] = get_marker_color(wp)
        markers_data.append(marker_data)
    
    # Calculate map center
    if water_points:
        center_lat = sum(wp.latitude for wp in water_points) / len(water_points)
        center_lon = sum(wp.longitude for wp in water_points) / len(water_points)
    else:
        center_lat, center_lon = 9.0765, 7.3986  # Nigeria center
    
    # Generate statistics
    total_points = len(water_points)
    projects = list(set(wp.project_name for wp in water_points))
    
    # Generate dynamic legend based on actual data
    unique_types = list(set(wp.water_point_type for wp in water_points))
    legend_items = []
    for water_type in sorted(unique_types):  # Sort for consistent ordering
        # Get a sample water point of this type to get display name and color
        sample_wp = next(wp for wp in water_points if wp.water_point_type == water_type)
        color = get_marker_color(sample_wp)
        display_name = sample_wp.water_point_type_display
        legend_items.append(f'''
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <span>{display_name}</span>
        </div>''')
    
    legend_html = "".join(legend_items)
    
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        body {{
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }}
        
        #map {{
            height: 100vh;
            width: 100%;
        }}
        
        .map-header {{
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .map-header h1 {{
            margin: 0 0 5px 0;
            font-size: 1.5em;
            color: #333;
        }}
        
        .map-header .stats {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }}
        
        .legend h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid rgba(255,255,255,0.8);
        }}
        
        /* Popup styling */
        .water-point-popup {{
            max-width: 300px;
            font-family: inherit;
        }}
        
        .water-point-popup h3 {{
            margin: 0 0 8px 0;
            color: #2c3e50;
            font-size: 1.2em;
        }}
        
        .location-breadcrumb {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-bottom: 10px;
        }}
        
        .characteristics {{
            margin: 10px 0;
            line-height: 1.4;
        }}
        
        .image-gallery {{
            margin: 10px 0;
        }}
        
        .photo-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 5px;
        }}
        
        .photo-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            background: #f9f9f9;
            min-width: 80px;
            max-width: 90px;
        }}
        
        .photo-item:hover {{
            background: #e9e9e9;
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .photo-thumbnail {{
            width: 60px;
            height: 45px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        
        .photo-placeholder {{
            font-size: 18px;
            margin-bottom: 4px;
            width: 60px;
            height: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f0f0f0;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}
        
        .photo-placeholder:hover {{
            background: #e0e0e0;
        }}
        
        .photo-name {{
            font-size: 0.7em;
            color: #666;
            text-align: center;
            word-break: break-word;
            margin-top: 2px;
        }}
        
        .notes {{
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-left: 3px solid #3498db;
            border-radius: 4px;
        }}
        
        .metadata {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #eee;
        }}
        
        /* Lightbox */
        .lightbox {{
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        
        .lightbox-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }}
        
        .lightbox img {{
            max-width: 100%;
            max-height: 100%;
            border-radius: 8px;
        }}
        
        .close-lightbox {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        
        .close-lightbox:hover {{
            color: #ccc;
        }}
        
        .lightbox-controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 15px;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 25px;
        }}
        
        .nav-button {{
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            font-size: 18px;
            padding: 8px 12px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s;
            min-width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .nav-button:hover {{
            background: rgba(255,255,255,0.3);
            transform: scale(1.1);
        }}
        
        .nav-button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
            transform: none;
        }}
        
        .image-counter {{
            color: white;
            font-size: 14px;
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="map-header">
        <h1>{title}</h1>
        <div class="stats">
            {total_points} water points across {len(projects)} projects
        </div>
    </div>
    
    <div id="map"></div>
    
    <div class="legend">
        <h4>Water Point Types</h4>
        {legend_html}
    </div>
    
    <!-- Lightbox -->
    <div id="lightbox" class="lightbox" onclick="closeLightbox()">
        <span class="close-lightbox" onclick="closeLightbox()">&times;</span>
        <div class="lightbox-content">
            <img id="lightbox-img" src="" alt="">
            <div class="lightbox-controls">
                <button class="nav-button prev-button" onclick="event.stopPropagation(); prevImage();" title="Previous image (Left arrow)">❮</button>
                <div id="image-counter" class="image-counter"></div>
                <button class="nav-button next-button" onclick="event.stopPropagation(); nextImage();" title="Next image (Right arrow)">❯</button>
            </div>
        </div>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Water points data
        const waterPoints = {json.dumps(markers_data, indent=2)};
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 8);
        
        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }}).addTo(map);
        
        // Add markers
        waterPoints.forEach(function(point) {{
            const marker = L.circleMarker([point.latitude, point.longitude], {{
                radius: 8,
                fillColor: point.marker_color,
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }});
            
            marker.bindPopup(point.popup_html, {{
                maxWidth: 350,
                className: 'custom-popup'
            }});
            

            
            marker.addTo(map);
        }});
        
        // Add scale control
        L.control.scale({{
            position: 'bottomleft'
        }}).addTo(map);
        

        
        // Lightbox state
        let currentImageIndex = 0;
        let currentImageSet = [];
        
        // Enhanced lightbox functions with navigation
        function openLightbox(imageSrc, imageAlt) {{
            // Find all images in the current popup
            const currentPopup = document.querySelector('.leaflet-popup-content');
            if (currentPopup) {{
                const photoItems = currentPopup.querySelectorAll('.photo-item');
                currentImageSet = [];
                
                photoItems.forEach(function(item, index) {{
                    const img = item.querySelector('.photo-thumbnail');
                    if (img) {{
                        const fullImageSrc = item.getAttribute('onclick').match(/'([^']+)'/)[1];
                        const fullImageAlt = item.getAttribute('onclick').match(/'[^']+',\\s*'([^']+)'/)[1];
                        currentImageSet.push({{
                            src: fullImageSrc,
                            alt: fullImageAlt
                        }});
                        
                        // Set current index if this is the clicked image
                        if (fullImageSrc === imageSrc) {{
                            currentImageIndex = index;
                        }}
                    }}
                }});
            }}
            
            document.getElementById('lightbox').style.display = 'block';
            updateLightboxImage();
        }}
        
        function updateLightboxImage() {{
            if (currentImageSet.length > 0) {{
                const currentImage = currentImageSet[currentImageIndex];
                document.getElementById('lightbox-img').src = currentImage.src;
                document.getElementById('lightbox-img').alt = currentImage.alt;
                
                // Update counter
                const counter = document.getElementById('image-counter');
                if (counter) {{
                    if (currentImageSet.length > 1) {{
                        counter.textContent = `${{currentImageIndex + 1}} of ${{currentImageSet.length}}`;
                        counter.style.display = 'block';
                    }} else {{
                        counter.style.display = 'none';
                    }}
                }}
                
                // Update button states
                const prevButton = document.querySelector('.prev-button');
                const nextButton = document.querySelector('.next-button');
                
                if (prevButton && nextButton) {{
                    if (currentImageSet.length <= 1) {{
                        prevButton.style.display = 'none';
                        nextButton.style.display = 'none';
                    }} else {{
                        prevButton.style.display = 'flex';
                        nextButton.style.display = 'flex';
                        
                        // Optional: disable buttons at ends (or keep cycling)
                        // prevButton.disabled = currentImageIndex === 0;
                        // nextButton.disabled = currentImageIndex === currentImageSet.length - 1;
                    }}
                }}
            }}
        }}
        
        function nextImage() {{
            if (currentImageSet.length > 1) {{
                currentImageIndex = (currentImageIndex + 1) % currentImageSet.length;
                updateLightboxImage();
            }}
        }}
        
        function prevImage() {{
            if (currentImageSet.length > 1) {{
                currentImageIndex = (currentImageIndex - 1 + currentImageSet.length) % currentImageSet.length;
                updateLightboxImage();
            }}
        }}
        
        function closeLightbox() {{
            document.getElementById('lightbox').style.display = 'none';
            currentImageSet = [];
            currentImageIndex = 0;
        }}
        
        // Enhanced keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (document.getElementById('lightbox').style.display === 'block') {{
                switch(event.key) {{
                    case 'Escape':
                        closeLightbox();
                        break;
                    case 'ArrowLeft':
                        event.preventDefault();
                        prevImage();
                        break;
                    case 'ArrowRight':
                        event.preventDefault();
                        nextImage();
                        break;
                }}
            }}
        }});
        
        console.log('Water Points Map loaded with', waterPoints.length, 'points');
    </script>
</body>
</html>'''
    
    return html_template


def create_water_points_map(
    project_name: Optional[str] = None, 
    output_dir: Optional[str] = None,
    water_data_path: str = "../../water_data"
) -> str:
    """
    Create interactive water points map.
    
    Args:
        project_name: Specific project to map (None for all projects)
        output_dir: Output directory (None for auto-generated)
        water_data_path: Path to water_data directory
    
    Returns:
        Path to generated HTML file
    """
    
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create output directory at project root level (same level as src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from water_visualization -> src -> project_root
        output_dir = os.path.join(project_root, f"water_map_output_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load water points
    all_water_points = []
    projects = find_projects(water_data_path)
    
    if project_name:
        if project_name not in projects:
            raise ValueError(f"Project '{project_name}' not found. Available: {list(projects.keys())}")
        water_points = load_water_points(project_name, water_data_path)
        all_water_points.extend(water_points)
        title = f"CommCare Connect Water Source Research - {project_name}"
    else:
        # Load all projects
        for proj_name in projects:
            water_points = load_water_points(proj_name, water_data_path)
            all_water_points.extend(water_points)
        title = "CommCare Connect Water Source Research"
    
    print(f"Loaded {len(all_water_points)} water points from {len(projects)} projects")
    
    # Generate HTML
    html_content = generate_html_map(all_water_points, output_dir, title)
    
    # Write to file
    filename = "index.html"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Map generated: {output_path}")
    print(f"Images copied to: {os.path.join(output_dir, 'images')}")
    
    # Open the file in the default browser
    try:
        webbrowser.open(f'file://{os.path.abspath(output_path)}')
        print(f"Opening map in default browser...")
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
        print(f"Please manually open: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Generate map for all projects
    try:
        output_file = create_water_points_map()
        print(f"Success! Map should open automatically in your browser.")
        print(f"If it doesn't open, manually open: {output_file}")
    except Exception as e:
        print(f"Error generating map: {e}")
        import traceback
        traceback.print_exc()
