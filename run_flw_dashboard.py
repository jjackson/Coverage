# run_flw_dashboard.py

import os
from src.utils import data_loader
from src.coverage_master import load_opportunity_domain_mapping
from src.flw_summary_dashboard import create_flw_dashboard
from src.models import CoverageData
from src.create_flw_views import create_flw_views_report
from src.create_statistics import create_statistics_report
from src.create_delivery_map import create_leaflet_map
from src.opportunity_comparison_statistics import create_opportunity_comparison_report
from dotenv import load_dotenv
import webbrowser
import argparse
from datetime import datetime

def load_coverage_data_objects():
    opportunity_to_domain_mapping = load_opportunity_domain_mapping()

    # Get latest CSV
    def get_latest_csv_file(folder="data"):
        csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in 'data/' folder.")

        # Sort by modification time descending, return first
        csv_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = csv_files[0]
        print(f"📁 Latest CSV file found: {latest_file}")
        return latest_file

    csv_path = get_latest_csv_file()
    excel_files = data_loader.get_available_files("data", "xlsx")
    service_delivery_by_opportunity_df = data_loader.load_service_delivery_df_by_opportunity(csv_path)

    coverage_data_objects = {}

    for opportunity_name, service_df in service_delivery_by_opportunity_df.items():
        domain_name = opportunity_to_domain_mapping.get(opportunity_name)
        if not domain_name:
            continue

        matching_excels = [f for f in excel_files if domain_name in f]
        if matching_excels:
            # Sort matching files by modification time, latest first
            matching_excels.sort(key=os.path.getmtime, reverse=True)
            matching_excel = matching_excels[0]
        else:
            matching_excel = None
        if not matching_excel:
            continue

        coverage_data = data_loader.get_coverage_data_from_excel_and_csv(matching_excel, None)
        coverage_data.load_service_delivery_from_datafame(service_df)
        coverage_data.project_space = domain_name
        coverage_data.opportunity_name = opportunity_name

        coverage_data_objects[domain_name] = coverage_data

    return coverage_data_objects

def main():
    # Load environment variables
    load_dotenv()
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Generate FLW dashboard")
    parser.add_argument("--excel-file", help="Excel file containing delivery unit data")
    parser.add_argument("--csv-file", help="CSV file containing service delivery data")
    args = parser.parse_args()
    
    # Get available Excel files
    excel_files = data_loader.get_available_files('data', 'xlsx')
    
    if not excel_files:
        print("Error: No Excel files found in the data directory")
        return
    
    # Get available CSV files
    csv_files = data_loader.get_available_files('data', 'csv')
    
    if not csv_files:
        print("Error: No CSV files found in the data directory")
        return
    
    # Select Excel file
    if args.excel_file:
        matching_excel = args.excel_file
    else:
        print("\nAvailable Excel files:")
        for i, file in enumerate(excel_files, 1):
            print(f"{i}. {file}")
        
        choice = 0
        while choice < 1 or choice > len(excel_files):
            try:
                choice = int(input(f"\nEnter the number for the Excel file (1-{len(excel_files)}): "))
            except ValueError:
                print("Please enter a valid number.")
        
        matching_excel = excel_files[choice - 1]
    
    print(f"\nSelected Excel file: {matching_excel}")
    
    # Select CSV file
    if args.csv_file:
        matching_csv = args.csv_file
    else:
        print("\nAvailable CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        
        choice = 0
        while choice < 1 or choice > len(csv_files):
            try:
                choice = int(input(f"\nEnter the number for the CSV file (1-{len(csv_files)}): "))
            except ValueError:
                print("Please enter a valid number.")
        
        matching_csv = csv_files[choice - 1]
    
    print(f"\nSelected CSV file: {matching_csv}")
    
    # Load service delivery data from CSV
    print("Loading service delivery data from CSV...")
    service_delivery_by_opportunity_df = data_loader.load_service_delivery_df_by_opportunity_from_csv(matching_csv)
    
    # Get the first opportunity's data (since we're only processing one at a time)
    opportunity_name = list(service_delivery_by_opportunity_df.keys())[0]
    service_df = service_delivery_by_opportunity_df[opportunity_name]
    
    # Load the data using the CoverageData model
    print("Loading data from Excel file...")
    coverage_data = data_loader.get_coverage_data_from_excel_and_csv(matching_excel, None)
    
    # Load service delivery data into coverage_data
    print("Loading service delivery data into coverage data...")
    coverage_data.load_service_delivery_from_datafame(service_df)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"flw_dashboard_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Change to output directory for generating files
    current_dir = os.getcwd()
    os.chdir(output_dir)
    
    # Generate FLW views
    print("\nGenerating FLW views...")
    flw_views_file = create_flw_views_report(coverage_data=coverage_data)
    
    # Generate statistics
    print("\nGenerating statistics...")
    stats_file = create_statistics_report(coverage_data=coverage_data)
    
    # Generate delivery map
    print("\nGenerating delivery map...")
    map_file = create_leaflet_map(coverage_data=coverage_data)
    
    # Generate opportunity comparison report
    print("\nGenerating opportunity comparison report...")
    comparison_report_file = create_opportunity_comparison_report({coverage_data.project_space: coverage_data})
    
    # Create index HTML
    print("\nCreating index HTML...")
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>FLW Dashboard</title>
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
        }}
        .btn {{
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
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
        <h1>FLW Dashboard</h1>
        
        <div class="card">
            <h2>FLW Views</h2>
            <p>Detailed analysis of Field-Level Worker performance and activities.</p>
            <a href="{os.path.basename(flw_views_file)}" class="btn">View FLW Analysis</a>
        </div>
        
        <div class="card">
            <h2>Coverage Statistics</h2>
            <p>Statistical analysis of delivery coverage and performance metrics.</p>
            <a href="{os.path.basename(stats_file)}" class="btn">View Statistics</a>
        </div>
        
        <div class="card">
            <h2>Delivery Map</h2>
            <p>Interactive map showing delivery units, service areas, and delivery points.</p>
            <a href="{os.path.basename(map_file)}" class="btn">View Map</a>
        </div>
        
        <div class="card">
            <h2>Opportunity Analysis</h2>
            <p>Detailed analysis including progress charts, statistics, and performance metrics.</p>
            <a href="{os.path.basename(comparison_report_file)}" class="btn">View Analysis</a>
        </div>
        
        <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>"""
    
    # Write the index HTML file
    index_path = "index.html"
    with open(index_path, "w") as f:
        f.write(index_html)
    
    # Change back to original directory
    os.chdir(current_dir)
    
    # Get absolute path for the index file
    abs_index_path = os.path.abspath(os.path.join(output_dir, "index.html"))
    
    print(f"\nAll done! Open the dashboard at: {abs_index_path}")
    
    # Open the dashboard in the default web browser
    print("Launching dashboard in your default browser...")
    try:
        webbrowser.open(f"file://{abs_index_path}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please open the file manually.")

if __name__ == "__main__":
    main()
