import os
import glob
import argparse
import subprocess
import webbrowser
from datetime import datetime

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .models import CoverageData
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData

def get_available_files():
    """Get available Excel and CSV files in the data subdirectory."""
    # Ensure data directory exists
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Look for files in the data directory
    files = glob.glob(os.path.join(data_dir, '*.*'))
    excel_files = [f for f in files if f.lower().endswith(('.xlsx', '.xls')) and not os.path.basename(f).startswith('~')]
    csv_files = [f for f in files if f.lower().endswith('.csv')]
    return excel_files, csv_files

def select_file(file_list, file_type, args=None):
    """Allow user to select a file interactively or use command line argument."""
    if args and file_type == 'excel' and args.excel_file:
        # Use command line argument if provided
        if args.excel_file in file_list:
            return args.excel_file
        else:
            print(f"Warning: Specified Excel file '{args.excel_file}' not found.")
    
    if args and file_type == 'csv' and args.csv_file:
        # Use command line argument if provided
        if args.csv_file in file_list:
            return args.csv_file
        else:
            print(f"Warning: Specified CSV file '{args.csv_file}' not found.")
    
    # If there's only one file, use it automatically
    if len(file_list) == 1:
        print(f"\nAutomatically selected the only available {file_type.upper()} file: {file_list[0]}")
        return file_list[0]
    
    # Interactive selection if no valid argument was provided and multiple files exist
    print(f"\nAvailable {file_type.upper()} files:")
    for i, file in enumerate(file_list, 1):
        print(f"{i}. {file}")
    
    choice = 0
    while choice < 1 or choice > len(file_list):
        try:
            choice = int(input(f"\nEnter the number for the {file_type} file (1-{len(file_list)}): "))
        except ValueError:
            print("Please enter a valid number.")
    
    return file_list[choice - 1]

def create_output_directory():
    """Create a timestamped output directory to store all generated files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"coverage_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_index_html(output_dir, map_file, stats_file, flw_views_file=None):
    """Generate an index HTML file that links to the map and statistics pages."""
    # Get relative paths to the output files - these are just the filenames since we're in the output directory
    map_path = map_file
    stats_path = stats_file
    flw_views_path = flw_views_file
    
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage Analysis Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .card {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
            margin: 15px 0;
            padding: 15px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .card h2 {{
            margin-top: 0;
            color: #444;
        }}
        .btn {{
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
            transition: background-color 0.3s;
        }}
        .btn:hover {{
            background-color: #45a049;
        }}
        .timestamp {{
            color: #777;
            font-size: 0.9em;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Coverage Analysis Dashboard</h1>
        
        <div class="card-container">
            <div class="card">
                <div>
                    <h2>Delivery Coverage Map</h2>
                    <p>Interactive map showing delivery units, service areas, and delivery points.</p>
                </div>
                <a href="{map_path}" class="btn">View Map</a>
            </div>
            
            <div class="card">
                <div>
                    <h2>Coverage Statistics</h2>
                    <p>Statistical analysis of delivery coverage and performance metrics.</p>
                </div>
                <a href="{stats_path}" class="btn">View Statistics</a>
            </div>
            
            {f'''
            <div class="card">
                <div>
                    <h2>FLW Analysis</h2>
                    <p>Detailed analysis of Field-Level Worker performance and activities.</p>
                </div>
                <a href="{flw_views_path}" class="btn">View FLW Analysis</a>
            </div>
            ''' if flw_views_path else ''}
        </div>
        
        <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>"""
    
    # Write the index HTML file directly to the current directory (which should be the output directory)
    with open("index.html", "w") as f:
        f.write(index_html)
    
    return "index.html"

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Master command for generating coverage analysis")
    parser.add_argument("--excel-file", help="Excel file containing delivery unit data")
    parser.add_argument("--csv-file", help="CSV file containing service delivery data")
    parser.add_argument("--output-dir", help="Custom output directory name")
    args = parser.parse_args()
    
    # Get available files
    excel_files, csv_files = get_available_files()
    
    # Check if files are available
    if not excel_files:
        print("Error: No Excel files found in the data directory.")
        return
    
    if not csv_files:
        print("Error: No CSV files found in the data directory.")
        return
    
    # Select input files
    excel_file = select_file(excel_files, "excel", args)
    csv_file = select_file(csv_files, "csv", args)
    
    # Create output directory
    output_dir = args.output_dir if args.output_dir else create_output_directory()
    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nInput files selected:")
    print(f"Excel file: {excel_file}")
    print(f"CSV file: {csv_file}")
    print(f"Output directory: {output_dir}")
    
    # Load the data using the CoverageData model
    print("\nLoading data from input files...")
    coverage_data = CoverageData.from_excel_and_csv(excel_file, csv_file)
    
    # Run the delivery map creation script
    print("\nGenerating delivery map...")
    current_dir = os.getcwd()
    os.chdir(output_dir)
    
    # Import the map creation function
    try:
        # When used as a module
        from .create_delivery_map import create_leaflet_map
    except ImportError:
        # When run as a script
        from src.create_delivery_map import create_leaflet_map
    
    map_file = create_leaflet_map(coverage_data=coverage_data)
    
    # Check if map file was created successfully
    if not os.path.exists(map_file):
        print(f"Warning: Expected map file '{map_file}' was not created.")
    
    # Run the statistics script
    print("\nGenerating statistics...")
    
    # Import the statistics report function
    try:
        # When used as a module
        from .create_statistics import create_statistics_report
    except ImportError:
        # When run as a script
        from src.create_statistics import create_statistics_report
    
    stats_file = create_statistics_report(coverage_data=coverage_data)
    
    # Check if stats file was created successfully
    if not os.path.exists(stats_file):
        print(f"Warning: Expected statistics file '{stats_file}' was not created.")
    
    # Generate FLW views
    print("\nGenerating FLW views...")
    
    # Import the FLW views function
    try:
        # When used as a module
        from .create_flw_views import create_flw_views_report
    except ImportError:
        # When run as a script
        from src.create_flw_views import create_flw_views_report
    
    flw_views_file = create_flw_views_report(coverage_data=coverage_data)
    
    # Check if FLW views file was created successfully
    if not os.path.exists(flw_views_file):
        print(f"Warning: Expected FLW views file '{flw_views_file}' was not created.")
        flw_views_file = None
    
    # Generate index HTML
    print("\nCreating dashboard index...")
    index_file = "index.html"
    generate_index_html(output_dir, map_file, stats_file, flw_views_file)
    
    # Construct the full path to the index.html file
    full_path = os.path.join(current_dir, output_dir, index_file)
    
    # Change back to original directory
    os.chdir(current_dir)
    
    print(f"\nAll done! Open the dashboard at: {full_path}")
    
    # Open the dashboard in the default web browser
    print("Launching dashboard in your default browser...")
    try:
        webbrowser.open(f"file://{full_path}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please open the file manually.")

if __name__ == "__main__":
    main()