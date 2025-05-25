import os
import glob
import argparse
import subprocess
import webbrowser
from datetime import datetime
from dotenv import load_dotenv
from . import utils_data_loader
from .models import CoverageData

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

def generate_coverage_outputs(output_dir, coverage_data, project_key):
    """Generate all output files for a single coverage data object."""
    print(f"\nGenerating outputs for {project_key}...")
    
    # Create subdirectory for this project
    project_dir = os.path.join(output_dir, project_key)
    os.makedirs(project_dir, exist_ok=True)
    
    # Change to project directory
    current_dir = os.getcwd()
    os.chdir(project_dir)
    
    # Import the required functions
    try:
        # When used as a module
        from .create_delivery_map import create_leaflet_map
        from .create_statistics import create_statistics_report
        from .create_flw_views import create_flw_views_report
    except ImportError:
        # When run as a script
        from src.create_delivery_map import create_leaflet_map
        from src.create_statistics import create_statistics_report
        from src.create_flw_views import create_flw_views_report
    
    # Generate delivery map
    print(f"  Generating delivery map for {project_key}...")
    map_file = create_leaflet_map(coverage_data=coverage_data)
    if not os.path.exists(map_file):
        print(f"  Warning: Expected map file '{map_file}' was not created.")
        map_file = None
    
    # Generate statistics
    print(f"  Generating statistics for {project_key}...")
    stats_file = create_statistics_report(coverage_data=coverage_data)
    if not os.path.exists(stats_file):
        print(f"  Warning: Expected statistics file '{stats_file}' was not created.")
        stats_file = None
    
    # Generate FLW views
    print(f"  Generating FLW views for {project_key}...")
    flw_views_file = create_flw_views_report(coverage_data=coverage_data)
    if not os.path.exists(flw_views_file):
        print(f"  Warning: Expected FLW views file '{flw_views_file}' was not created.")
        flw_views_file = None
    
    # Change back to original directory
    os.chdir(current_dir)
    
    return {
        'project_key': project_key,
        'project_dir': project_key,  # relative path from output_dir
        'map_file': map_file,
        'stats_file': stats_file,
        'flw_views_file': flw_views_file,
        'opportunity_name': getattr(coverage_data, 'opportunity_name', project_key)
    }

def generate_index_html(output_dir, project_outputs):
    """Generate an index HTML file that links to all project outputs."""
    
    # Generate cards for each project
    project_cards = ""
    for output_info in project_outputs:
        project_key = output_info['project_key']
        project_dir = output_info['project_dir']
        opportunity_name = output_info['opportunity_name']
        map_file = output_info['map_file']
        stats_file = output_info['stats_file']
        flw_views_file = output_info['flw_views_file']
        
        # Create links with proper paths
        map_link = f"{project_dir}/{map_file}" if map_file else None
        stats_link = f"{project_dir}/{stats_file}" if stats_file else None
        flw_views_link = f"{project_dir}/{flw_views_file}" if flw_views_file else None
        
        project_cards += f"""
        <div class="project-section">
            <h2>{opportunity_name}</h2>
            <div class="card-container">
                {f'''
                <div class="card">
                    <div>
                        <h3>Delivery Coverage Map</h3>
                        <p>Interactive map showing delivery units, service areas, and delivery points.</p>
                    </div>
                    <a href="{map_link}" class="btn">View Map</a>
                </div>
                ''' if map_link else ''}
                
                {f'''
                <div class="card">
                    <div>
                        <h3>Coverage Statistics</h3>
                        <p>Statistical analysis of delivery coverage and performance metrics.</p>
                    </div>
                    <a href="{stats_link}" class="btn">View Statistics</a>
                </div>
                ''' if stats_link else ''}
                
                {f'''
                <div class="card">
                    <div>
                        <h3>FLW Analysis</h3>
                        <p>Detailed analysis of Field-Level Worker performance and activities.</p>
                    </div>
                    <a href="{flw_views_link}" class="btn">View FLW Analysis</a>
                </div>
                ''' if flw_views_link else ''}
            </div>
        </div>
        """
    
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
        h2 {{
            color: #444;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            margin-top: 0;
            color: #555;
        }}
        .project-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
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
        <p>Analysis results for {len(project_outputs)} project(s)</p>
        
        {project_cards}
        
        <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>"""
    
    # Write the index HTML file to the output directory
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(index_html)
    
    return "index.html"

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Master command for generating coverage analysis")
    parser.add_argument("--excel-file", help="Excel file containing delivery unit data")
    parser.add_argument("--csv-file", help="CSV file containing service delivery data")
    parser.add_argument("--output-dir", help="Custom output directory name")
    args = parser.parse_args()
    
    # Hardcoded mapping to change opportunity names to domain names
    opportunity_to_domain_mapping = {
        "ZEGCAWIS | CHC Givewell Scale Up" : "ccc-chc-zegcawis-2024-25",
        "COWACDI | CHC Givewell Scale Up" : "ccc-chc-cowacdi-2024-25",
        "ADD NEXT HERE" : "ADD NEXT HERE" 
    }

    # Load environment variables from .env file
    load_dotenv()
    
    coverage_data_objects = {}
    use_api = os.environ.get('USE_API', '').upper() == 'TRUE'
    if use_api:
        print("Using API Method")
        excel_files, csv_files = get_available_files()
           
        if not csv_files:
            print("Error: No CSV files found in the data directory.")
            return
        
        csv_file = select_file(csv_files, "csv", args)
        service_delivery_by_opportunity_df = utils_data_loader.load_service_delivery_df_by_opportunity(csv_file)

        print("Loading Delivery Units from API, oppurtinty names found in CSV:" + service_delivery_by_opportunity_df.keys().__str__())
        user = os.environ.get('COMMCARE_USERNAME')
        api_key = os.environ.get('COMMCARE_API_KEY')
        
        # Loop through each opportunity and create CoverageData objects                
        for opportunity_name, service_df in service_delivery_by_opportunity_df.items():
            print(f"\nProcessing opportunity: {opportunity_name}")
            print(f"Service points for this opportunity: {len(service_df)}")

            # Use mapped domain name if available, otherwise use opportunity name
            domain_name = opportunity_to_domain_mapping.get(opportunity_name)
            
            coverage_data = CoverageData.from_du_api_and_service_dataframe(
                domain=domain_name,
                user=user,
                api_key=api_key,
                service_df=service_df)

            coverage_data.project_space = domain_name
            coverage_data.opportunity_name = opportunity_name
            
            # Store with a combined key
            key = domain_name
            coverage_data_objects[key] = coverage_data
            print(f"Successfully loaded coverage data for {key}")
                        
        print(f"\nTotal coverage data objects created: {len(coverage_data_objects)}")
        
    else:     
        print("Loading from Local Files")
        # Get available files
        excel_files, csv_files = get_available_files()
        
        # Check if files are available
        if not excel_files:
            print("Error: No Excel files found in the data directory")
            return
    
        if not csv_files:
            print("Error: No CSV files found in the data directory.")
            return
    
        # Select input files
        excel_file = select_file(excel_files, "excel", args)
        csv_file = select_file(csv_files, "csv", args)
    
        print(f"\nInput files selected:")
        print(f"Excel file: {excel_file}")
        print(f"CSV file: {csv_file}")
        
        # Load the data using the CoverageData model
        print("\nLoading data from input files...")
        coverage_data = CoverageData.from_excel_and_csv(excel_file, csv_file)
        coverage_data.project_space = opportunity_to_domain_mapping.get(coverage_data.opportunity_name)

        # Use project_space if available, otherwise use opportunity_name as key
        key = coverage_data.project_space if coverage_data.project_space else coverage_data.opportunity_name
        coverage_data_objects[key] = coverage_data
    
    # Create output directory
    output_dir = args.output_dir if args.output_dir else create_output_directory()
    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate outputs for each coverage data object
    print(f"\nGenerating outputs for {len(coverage_data_objects)} project(s)...")
    project_outputs = []
    
    for key, coverage_data in coverage_data_objects.items():
        output_info = generate_coverage_outputs(output_dir, coverage_data, key)
        project_outputs.append(output_info)
    
    # Generate main index HTML
    print("\nCreating main dashboard index...")
    generate_index_html(output_dir, project_outputs)
    
    # Construct the full path to the index.html file
    full_path = os.path.join(os.getcwd(), output_dir, "index.html")
    
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