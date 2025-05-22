import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
import argparse
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from geopy.distance import geodesic
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .models import CoverageData
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData

# Function to calculate the shortest path distance within each Service Area using TSP
def calculate_shortest_path_distances(df):
    sa_distances = {}
    for sa_id, group in df.groupby('service_area_id'):
        # Improved parsing of centroid data
        centroids = []
        for c in group['centroid']:
            if isinstance(c, str):
                # Attempt to parse the string into a tuple of floats
                try:
                    lat, lon = map(float, c.split())
                    centroids.append((lat, lon))
                except ValueError:
                    raise ValueError(f"Centroid '{c}' for Service Area {sa_id} is not in a valid format.")
            elif isinstance(c, (list, tuple)) and len(c) == 2:
                centroids.append(tuple(c))
            else:
                raise ValueError(f"Centroid '{c}' for Service Area {sa_id} is not in a valid format.")

        # Calculate total distance using geodesic distance
        total_distance = 0
        for i in range(len(centroids) - 1):
            total_distance += geodesic(centroids[i], centroids[i + 1]).kilometers
        # Add distance from last to first to complete the cycle
        total_distance += geodesic(centroids[-1], centroids[0]).kilometers

        sa_distances[sa_id] = total_distance
    return sa_distances

def create_statistics_report(excel_file=None, service_delivery_csv=None, coverage_data=None):
    """
    Create statistics report from the DU Export Excel file and service delivery CSV
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
    
    # Create the HTML report
    html_content = create_html_report(coverage_data)
    
    # Write the HTML to a file
    output_filename = "coverage_statistics.html"
    with open(output_filename, "w") as f:
        f.write(html_content)
    
    print(f"Statistics report has been created: {output_filename}")
    return output_filename

def create_html_report(coverage_data):
    """
    Create HTML report with the statistics information
    
    Args:
        coverage_data: CoverageData object with precomputed values
    """

    # Use the DataFrame directly from CoverageData
    delivery_df = coverage_data.delivery_units_df
    
    # Use the convenience method to get service points DataFrame
    service_df = coverage_data.create_service_points_dataframe()

    # Calculate summary statistics
    total_units = coverage_data.total_delivery_units
    total_buildings = coverage_data.total_buildings
    completed_dus = coverage_data.total_completed_dus
    visited_dus = coverage_data.total_visited_dus
    unvisited_dus = coverage_data.total_unvisited_dus
    delivery_progress = coverage_data.completion_percentage
    unique_service_areas = coverage_data.total_service_areas
    unique_flws = coverage_data.total_flws
    
    # Service delivery stats
    service_stats = {}
    if service_df is not None:
        # Use properties from CoverageData instead of recalculating
        service_stats['total_points'] = coverage_data.total_service_points if hasattr(coverage_data, 'total_service_points') else len(service_df)
        
        # Reuse the FLW count we already have instead of recalculating
        service_stats['unique_flws'] = coverage_data.total_flws
      
    # Prepare delivery units data for the table (exclude status "---")
    du_table_data = delivery_df[delivery_df['du_status'] != '---'].copy()
    
    # Apply FLW name mapping
    if 'flw_commcare_id' in du_table_data.columns:
        du_table_data['flw_name'] = du_table_data['flw_commcare_id'].apply(
            lambda id: coverage_data.flw_commcare_id_to_name_map.get(id, id)  # Use the ID itself if no name mapping exists
        )
    
    # Add Delivery Count / Buildings column if both columns exist
    delivery_count_col = None
    buildings_col = None
    
    # Find the delivery count column (could be named differently)
    possible_delivery_cols = ['Delivery Count', 'delivery_count', 'DeliveryCount', 'delivery count']
    for col in possible_delivery_cols:
        if col in du_table_data.columns:
            delivery_count_col = col
            break
    
    # Find the buildings column (could be named differently)
    possible_buildings_cols = ['#Buildings', 'Buildings', 'buildings', 'num_buildings', 'building_count']
    for col in possible_buildings_cols:
        if col in du_table_data.columns:
            buildings_col = col
            break
    
    # Calculate the ratio if both columns exist
    if delivery_count_col and buildings_col:
        du_table_data['Delivery Count / Buildings'] = du_table_data.apply(
            lambda row: round(float(row[delivery_count_col]) / float(row[buildings_col])) 
            if pd.notnull(row[delivery_count_col]) and pd.notnull(row[buildings_col]) and float(row[buildings_col]) > 0 
            else None, 
            axis=1
        )
    
    # Calculate days in DU and flag
    def calculate_days_in_du(row):
        try:
            # Convert dates to datetime objects
            check_in = pd.to_datetime(row['checked_in_date'])
            today = pd.to_datetime('today').normalize()  # Get today's date at midnight
            
            if row['checked_out_date'] == '---':
                # Calculate days between today and check-in date
                days = (today - check_in).days
                return int(days) if days >= 0 else None
            else:
                check_out = pd.to_datetime(row['checked_out_date'])
                # Calculate days between check-out and check-in date
                days = (check_out - check_in).days
                return int(days) if days >= 0 else None
        except:
            return None
    
    # Add days_in_du column
    du_table_data['days_in_du'] = du_table_data.apply(calculate_days_in_du, axis=1)
    
    # Add flag_days_in_du column
    du_table_data['flag_days_in_du'] = du_table_data['days_in_du'].apply(
        lambda x: True if pd.notnull(x) and x >= 7 else None
    )
    
    # Add flag_delivery_per_building column
    du_table_data['flag_delivery_per_building'] = du_table_data['Delivery Count / Buildings'].apply(
        lambda x: True if pd.notnull(x) and x >= 10 else None
    )
    
    # Define columns to exclude from the table
    columns_to_exclude = [
        'caseid', 'centroid', 'BoundingBox', 'Wkt', 'Delivery Target', 'Closed', 'Closed by username',
        'Surface Area', 'Surface Area (sq. meters)', 'surface_area', 'Surface_Area',  # Surface Area variations
        'bounding box', 'Bounding Box', 'boundingbox', 'bounding_box',  # Bounding Box variations
        'wkt', 'WKT',  # WKT variations
        'Closed Date', 'closed_date', 'closed date',  # Closed Date variations
        'Last Modified By Username', 'last_modified_by_username', 'last modified by username',  # Last Modified By Username variations
        'Last Modified Date', 'last_modified_date', 'last modified date',  # Last Modified Date variations
        'Opened Date', 'opened_date', 'opened date',  # Opened Date variations
        'Closed By Username', 'closed_by_username', 'closed by username',  # Additional Closed By Username variations
        'Last Modified By User Username', 'last_modified_by_user_username', 'last modified by user username',  # Last Modified By User Username variations
        'Owner Name', 'owner_name', 'owner name',  # Owner Name variations
        'flw_commcare_id',  # Remove FLW ID from table
        'Service Area', # Remove Service Area from table
        'closed', 'Closed', 'CLOSED'  # Remove Closed column and its variations
    ]
    
    # Get the columns we want to include (filter out excluded columns)
    du_columns = [col for col in du_table_data.columns if col not in columns_to_exclude]
    
    # Define preferred column order (edit this array to change the order of columns)
    preferred_column_order = [
        'flw_name',              # Field worker name
        'delivery_unit_id',      # Delivery unit ID
        'du_status',             # Delivery unit status
        'service_area_id',       # Service area ID
        'Delivery Count / Buildings',  # Ratio of delivery count to buildings
        'flag_delivery_per_building',  # Flag for DUs with 10 or more deliveries per building
        'checked_in_date',       # Checked in date
        'checked_out_date',       # Checked out date
        'days_in_du',           # Number of days in delivery unit
        'flag_days_in_du'       # Flag for DUs with 7 or more days
    ]
    
    # Reorder columns based on preferred order
    ordered_columns = []
    for col in preferred_column_order:
        if col in du_columns:
            ordered_columns.append(col)
            du_columns.remove(col)
    
    # Final columns list with preferred columns first, then the rest
    du_columns = ordered_columns + du_columns
    

   
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Coverage Statistics Report</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <!-- Include jQuery for interactive features -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        
        <!-- Include DataTables for sortable and exportable tables -->
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.dataTables.min.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/colreorder/1.5.5/css/colReorder.dataTables.min.css">
        <script type="text/javascript" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/colreorder/1.5.5/js/dataTables.colReorder.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.print.min.js"></script>
        
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background-color: #fff;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .stat-card h3 {{
                margin-top: 0;
                color: #7f8c8d;
                font-size: 16px;
                font-weight: normal;
            }}
            .stat-card .value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
                margin: 10px 0;
            }}
            .figure-container {{
                margin: 30px 0;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                color: #95a5a6;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }}
            .progress-container {{
                width: 100%;
                background-color: #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .progress-bar {{
                height: 24px;
                border-radius: 5px;
                background-color: #27ae60;
                text-align: center;
                line-height: 24px;
                color: white;
                font-weight: bold;
            }}
            .delivery-units-table-container {{
                margin-top: 30px;
                overflow-x: auto;
            }}
            .table-info {{
                margin-bottom: 15px;
                color: #666;
                font-style: italic;
            }}
            table.dataTable {{
                width: 100% !important;
                margin: 15px 0 !important;
            }}
            .dataTables_wrapper .dataTables_filter input {{
                margin-left: 5px;
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .dataTables_wrapper .dataTables_length select {{
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .dt-buttons {{
                margin-bottom: 15px;
            }}
            .dt-button {{
                background-color: #4CAF50 !important;
                color: white !important;
                border: none !important;
                padding: 8px 15px !important;
                border-radius: 4px !important;
                cursor: pointer !important;
                margin-right: 5px !important;
            }}
            .dt-button:hover {{
                background-color: #45a049 !important;
            }}
        </style>
        
        <script>
            $(document).ready(function() {{
                // Initialize the Delivery Units table with DataTables
                $('#delivery-units-table').DataTable({{
                    paging: true,
                    searching: true,
                    ordering: true,
                    info: true,
                    lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                    dom: 'Bfrtip',
                    buttons: [
                        'copy', 'csv', 'excel'
                    ],
                    order: [[0, 'asc']], // Default sort by first column
                    scrollX: true,
                    colReorder: true, // Enable column reordering
                    pageLength: 25 // Default to showing 25 rows
                }});
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Coverage Statistics Report</h1>
            
            <section>
                <h2>Summary</h2>
                <div class="summary-grid">
                    <div class="stat-card">
                        <h3>Total Delivery Units</h3>
                        <div class="value">{total_units:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Completed DUs</h3>
                        <div class="value">{completed_dus:,}</div>
                        <div>{(completed_dus/total_units*100):.1f}% of total</div>
                    </div>
                    <div class="stat-card">
                        <h3>Visited DUs (not completed)</h3>
                        <div class="value">{visited_dus:,}</div>
                        <div>{(visited_dus/total_units*100):.1f}% of total</div>
                    </div>
                    <div class="stat-card">
                        <h3>Unvisited DUs</h3>
                        <div class="value">{unvisited_dus:,}</div>
                        <div>{(unvisited_dus/total_units*100):.1f}% of total</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Buildings (computed during Segmentation) </h3>
                        <div class="value">{total_buildings:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Service Areas</h3>
                        <div class="value">{unique_service_areas:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Field Workers</h3>
                        <div class="value">{unique_flws:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Overall Delivery Progress</h3>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {min(delivery_progress, 100)}%">
                                {delivery_progress:.1f}%
                            </div>
                        </div>
                        <div style="font-size: 14px; margin-top: 5px;">
                            ({completed_dus:,} of {total_units:,} DUs completed)
                        </div>
                    </div>
                </div>
            </section>
            
            <section>
                <h2>Delivery Unit Status</h2>
                <div class="summary-grid">
    """
    
    # Add status cards
    for status, count in coverage_data.delivery_status_counts.items():
        percentage = (count / total_units) * 100
        html_content += f"""
                    <div class="stat-card">
                        <h3>{status.title()}</h3>
                        <div class="value">{count:,}</div>
                        <div>{percentage:.1f}% of total</div>
                    </div>
        """
    
    html_content += """
                </div>
            </section>
    """
    
    # Add service delivery stats if available
    if service_df is not None and service_stats:
        html_content += f"""
            <section>
                <h2>Service Delivery</h2>
                <div class="summary-grid">
                    <div class="stat-card">
                        <h3>Total Service Points</h3>
                        <div class="value">{service_stats.get('total_points', 0):,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Field Workers</h3>
                        <div class="value">{unique_flws:,}</div>
                    </div>
                </div>
            </section>
        """
    
    # Add Delivery Units table section
    html_content += f"""
            <section>
                <h2>Delivery Units Data</h2>
                <div class="table-info">
                    Showing {len(du_table_data):,} delivery units with status other than "---". Use the search box to filter.
                </div>
                <div class="delivery-units-table-container">
                    <table id="delivery-units-table" class="display">
                        <thead>
                            <tr>
    """
    
    # Add table headers based on available columns
    for col in du_columns:
        # Format column header (replace underscores with spaces and capitalize)
        header = col.replace('_', ' ').title()
        html_content += f"<th>{header}</th>"
    
    html_content += """
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add table rows with data
    for _, row in du_table_data.iterrows():
        html_content += "<tr>"
        for col in du_columns:
            value = row[col]
            # Format the value - handle NaN/None values and format others as strings
            if pd.isna(value):
                formatted_value = ""
            else:
                formatted_value = str(value)
            
            html_content += f"<td>{formatted_value}</td>"
        html_content += "</tr>"
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </section>
    """
    
    # Close the HTML
    html_content += f"""
            <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create statistics report from Excel and CSV data")
    parser.add_argument("--excel", help="Excel file containing delivery unit data")
    parser.add_argument("--csv", help="CSV file containing service delivery data")
    args = parser.parse_args()
    
    excel_file = None
    delivery_csv = None
    
    # If arguments are provided, use them
    if args.excel and args.csv:
        excel_file = args.excel
        delivery_csv = args.csv
        
        print(f"\nCreating statistics report using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the statistics report using CoverageData model
        output_file = create_statistics_report(excel_file=excel_file, service_delivery_csv=delivery_csv)
        print(f"Statistics report created: {output_file}")
    else:
        # Interactive selection
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
        
        print(f"\nCreating statistics report using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the statistics report using CoverageData model
        output_file = create_statistics_report(excel_file=excel_file, service_delivery_csv=delivery_csv)
        print(f"Statistics report created: {output_file}") 