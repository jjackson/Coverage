import os
import glob
import argparse
import subprocess
import webbrowser
import json
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv, find_dotenv
from .utils import data_loader
import pickle  # Add this import at the top
import logging
log_dir = '../../'
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
log_file_path = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    filename= log_file_path,           # Log file name
    filemode='a',                 # Append mode ('w' to overwrite)
    level=logging.INFO,           # Minimum log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    from .opportunity_comparison_statistics import create_opportunity_comparison_report
except ImportError:
    from src.opportunity_comparison_statistics import create_opportunity_comparison_report

# At the top of the file, after imports
coverage_data_objects = None  # Module-level variable to store coverage data

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
        'project_dir': project_key,
        'map_file': map_file,
        'stats_file': stats_file,
        'flw_views_file': flw_views_file,
        'opportunity_name': getattr(coverage_data, 'opportunity_name', project_key)
    }

def generate_index_html(output_dir, output_info_list):
    """Generate an index HTML file that links to all project outputs."""
    
    # Generate comparison report section if available
    comparison_section = ""
    if len(output_info_list) > 1:
        section_title = "Multi-Project Analysis"
        card_title = "Opportunity Comparison Report"
        card_description = f"Comparative analysis across all {len(output_info_list)} projects including statistics, performance metrics, and cross-project insights."
        
        comparison_section = f"""
        <div class="comparison-section">
            <h2>{section_title}</h2>
            <div class="card-container">
                <div class="card comparison-card">
                    <div>
                        <h3>{card_title}</h3>
                        <p>{card_description}</p>
                    </div>
                    <a href="opportunity_comparison_report.html" class="btn btn-comparison">View Analysis</a>
                </div>
            </div>
        </div>
        """
    
    # Generate cards for each project
    project_cards = ""
    for output_info in output_info_list:
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
        .btn-comparison {{
            background-color: #2196F3;
        }}
        .btn-comparison:hover {{
            background-color: #1976D2;
        }}
        .comparison-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #e3f2fd;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        .comparison-card {{
            background-color: white;
            border-left: 4px solid #2196F3;
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
        <p>Analysis results for {len(output_info_list)} project(s)</p>
        
        {comparison_section}
        
        {project_cards}
        
        <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>"""
    
    # Write the index HTML file to the output directory
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    
    return index_path

def load_opportunity_domain_mapping() -> Dict[str, str]:
    """Load opportunity to domain mapping from environment variable."""
    mapping_str = os.environ.get('OPPORTUNITY_DOMAIN_MAPPING', '')
    
    if not mapping_str:
        # Return default mapping if no environment variable is set
        return {
           "ZEGCAWIS | CHC Givewell Scale Up": "ccc-chc-zegcawis-2024-25",
           "COWACDI | CHC Givewell Scale Up": "ccc-chc-cowacdi-2024-25"
        }
    
    try:
        # Try to parse as JSON first
        return json.loads(mapping_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, try pipe-separated format
        try:
            mapping = {}
            pairs = mapping_str.split('|')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)  # Split only on first colon
                    mapping[key.strip()] = value.strip()
            return mapping
        except Exception as e:
            raise ValueError(f"Could not parse OPPORTUNITY_DOMAIN_MAPPING from environment variable. Please check the format. Error: {e}")

def main():
    global coverage_data_objects
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Master command for generating coverage analysis")
    parser.add_argument("--excel-file", help="Excel file containing delivery unit data")
    parser.add_argument("--csv-file", help="CSV file containing service delivery data")
    parser.add_argument("--output-dir", help="Custom output directory name")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    find_dotenv()
    load_dotenv(override=True,verbose=True)
    
    # Load opportunity to domain mapping from environment
    opportunity_to_domain_mapping = load_opportunity_domain_mapping()
    coverage_data_objects = {}
    use_api = os.environ.get('USE_API', '').upper() == 'TRUE'

    if use_api:
        print("Using API Method")

        
        # csv_file = select_file(csv_files, "csv", args)
        # service_delivery_by_opportunity_df = data_loader.load_service_delivery_df_by_opportunity(csv_file)

        # Use Superset API to get the service delivery data   
        # Get Superset configuration from environment variables
        superset_url = os.environ.get('SUPERSET_URL')
        superset_username = os.environ.get('SUPERSET_USERNAME')
        superset_password = os.environ.get('SUPERSET_PASSWORD')

        # Validate environment variables
        missing_vars = []
        if not superset_url:
            missing_vars.append('SUPERSET_URL')
        if not superset_username:
            missing_vars.append('SUPERSET_USERNAME')
        if not superset_password:
            missing_vars.append('SUPERSET_PASSWORD')


        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        print("Loading Service Delivery Points from Superset API")
        
        service_delivery_by_opportunity_df = data_loader.load_service_delivery_df_by_opportunity_from_superset(
            superset_url, superset_username, superset_password
        )
        

        print("Loading Delivery Units from API, opportunity names found in CSV:")
        for key, value in service_delivery_by_opportunity_df.items():
            print(f"  {key}: {len(value)} service points" if hasattr(value, '__len__') else f"  {key}: {value}")
        user = os.environ.get('COMMCARE_USERNAME')
        api_key = os.environ.get('COMMCARE_API_KEY')
        
        # Loop through each opportunity and create CoverageData objects                
        for opportunity_name, service_df in service_delivery_by_opportunity_df.items():
            print(f"\nProcessing opportunity: {opportunity_name}")
            print(f"Service points for this opportunity: {len(service_df)}")
            # Use mapped domain name if available, otherwise use opportunity name
            domain_name = opportunity_to_domain_mapping.get(opportunity_name) 
            #-------Changing domain list to get data from env variables only -----#
            if(domain_name is not None and domain_name != ""):
                print("------domain_name------")
                print(domain_name)
                coverage_data = data_loader.get_coverage_data_from_du_api_and_service_dataframe(
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
        
        # Get available CSV files
        csv_files = data_loader.get_available_files('data', 'csv')
        
        if not csv_files:
            print("Error: No CSV files found in the data directory.")
            return
        
        # Select CSV file (use existing select_file logic)
        csv_file = select_file(csv_files, "csv", args)
        
        print(f"\nSelected CSV file: {csv_file}")
        
        # Load service delivery data and group by opportunity
        print("Loading Service Delivery Points from CSV...")
        service_delivery_by_opportunity_df = data_loader.load_service_delivery_df_by_opportunity_from_csv(csv_file)
        
        print("Opportunity names found in CSV:")
        for key, value in service_delivery_by_opportunity_df.items():
            print(f"  {key}: {len(value)} service points" if hasattr(value, '__len__') else f"  {key}: {value}")
        
        # Get available Excel files
        excel_files = data_loader.get_available_files('data', 'xlsx')
        
        if not excel_files:
            print("Error: No Excel files found in the data directory")
            return
        
        print(f"\nAvailable Excel files: {excel_files}")
        
        # Loop through each opportunity and try to find matching Excel file
        for opportunity_name, service_df in service_delivery_by_opportunity_df.items():
            print(f"\nProcessing opportunity: {opportunity_name}")
            print(f"Service points for this opportunity: {len(service_df)}")

            # Use mapped domain name if available, otherwise use opportunity name
            domain_name = opportunity_to_domain_mapping.get(opportunity_name)
            
            if not domain_name:
                print(f"  Warning: No domain mapping found for opportunity '{opportunity_name}', skipping")
                continue
            
            print(f"  Looking for Excel file containing domain: {domain_name}")
            
            # Look for Excel file that contains the domain name
            matching_excel_file = None
            for excel_file in excel_files:
                if domain_name in excel_file:
                    matching_excel_file = excel_file
                    break
            
            if not matching_excel_file:
                print(f"  Warning: No Excel file found containing domain '{domain_name}', skipping opportunity")
                continue
            
            print(f"  Found matching Excel file: {matching_excel_file}")
            
            # Load the data using the CoverageData model
            print(f"  Loading data from Excel file and CSV data...")
            coverage_data = data_loader.get_coverage_data_from_excel_and_csv(matching_excel_file, None)
  
            # Load service delivery data from the dataframe for this opportunity
            coverage_data.load_service_delivery_from_datafame(service_df)
            
            coverage_data.project_space = domain_name
            coverage_data.opportunity_name = opportunity_name
            
            # Store with domain name as key
            coverage_data_objects[domain_name] = coverage_data
            print(f"  Successfully loaded coverage data for {domain_name}")
        
        print(f"\nTotal coverage data objects created: {len(coverage_data_objects)}")
    
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
    
    # Generate opportunity comparison report if multiple projects exist
    comparison_report_file = None
    if len(coverage_data_objects) >= 1:
        print("\nGenerating opportunity comparison report...")
        # Change to output directory to generate the comparison report there
        current_dir = os.getcwd()
        os.chdir(output_dir)
        
        comparison_report_file = create_opportunity_comparison_report(coverage_data_objects)
        
        # Change back to original directory
        os.chdir(current_dir)
    
    # Generate main index HTML
    print("\nCreating main dashboard index...")
    generate_index_html(output_dir, project_outputs)


    # Open the index file in browser
    index_path = os.path.join(output_dir, "index.html")
    webbrowser.open(f"file://{os.path.abspath(index_path)}")
    
    # Save coverage data to a file
    with open('coverage_data.pkl', 'wb') as f:
        pickle.dump(coverage_data_objects, f)
    
    return coverage_data_objects

if __name__ == "__main__":
    main()