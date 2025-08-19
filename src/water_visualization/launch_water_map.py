#!/usr/bin/env python3
"""
Water Points Map Launcher
Simple script to generate and open the water points map.
"""
import os
import sys

# Import from the same directory
try:
    from create_water_map import create_water_points_map
    
    print("🗺️  Water Points Map Generator")
    print("=" * 40)
    print("Generating interactive map from water survey data...")
    print()
    
    # Generate the map (use path relative to project root)
    water_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'water_data')
    output_file = create_water_points_map(water_data_path=water_data_path)
    
    print()
    print("✅ Map generation complete!")
    print(f"📂 Output: {output_file}")
    print("🌐 Map should open automatically in your browser")
    print(f"📁 Self-contained directory ready for deployment!")
    print()
    print("Features:")
    print("  • Interactive markers for 141 water points")
    print("  • Color-coded by water point type")
    print("  • Click markers for detailed information")
    print("  • Click photo thumbnails to view full-size images")
    print("  • Navigate between images with ← → arrow keys")
    print("  • ESC key to close image viewer")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error generating map: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
