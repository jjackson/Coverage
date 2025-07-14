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

# Add this import
import constants

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .models import CoverageData
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData

def create_statistics_report(coverage_data=None):
    """
    Create statistics report from a CoverageData object.
    
    Args:
        coverage_data: Optional CoverageData object containing the already loaded data
    """
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
      
    # Debug: Check unique values in du_status column
    # print("Unique values in du_status column:")
    print(delivery_df['du_status'].value_counts(dropna=False))
    print(f"Total rows: {len(delivery_df)}")
    
    # Prepare delivery units data for the table (exclude status None, NaN)
    du_table_data = delivery_df[~delivery_df['du_status'].isna()].copy()
    print(f"Rows after filtering out None, NaN, and empty status: {len(du_table_data)}")
    
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
            
            if row['checked_out_date'] == None:
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
    
    # --- (1) Add Camping column with user input threshold and delivery_count > 20 ---
    # Default camping threshold
    camping_default = constants.DEFAULT_CAMPING_THRESHOLD
    # Add a Camping column (formerly flag_delivery_per_building)
    du_table_data['Camping'] = du_table_data.apply(
        lambda row: True if pd.notnull(row.get('Delivery Count / Buildings')) and pd.notnull(row.get(delivery_count_col))
            and float(row.get('Delivery Count / Buildings')) >= camping_default and float(row.get(delivery_count_col)) > constants.DEFAULT_DELIVERY_COUNT_THRESHOLD else False,
        axis=1
    )

    # --- (2) Add filter for Checked In Date (last 7 days) ---
    today = pd.to_datetime('today').normalize()
    du_table_data['checked_in_date_dt'] = pd.to_datetime(du_table_data['checked_in_date'], errors='coerce')
    du_table_data['checked_in_last_7_days'] = du_table_data['checked_in_date_dt'].apply(
        lambda d: (today - d).days <= 7 if pd.notnull(d) else False
    )

    # --- (3) Add filter for Camping=True ---
    du_table_data['camping_true'] = du_table_data['Camping'] == True

    # --- (4) Reduce table to only specified columns in order ---
    reduced_columns = [
        'flw_name',
        buildings_col,
        delivery_count_col,
        'flag_days_in_du',  # Will rename in header to 'Camping'
        'checked_in_date',
        'checked_out_date',
        'du_name',
        'du_status',
        'du_checkout_remark',
        'service_area_id',
    ]
    # Only keep columns that exist
    reduced_columns = [col for col in reduced_columns if col in du_table_data.columns]
    du_table_data = du_table_data[reduced_columns + ['Camping', 'checked_in_last_7_days', 'camping_true']]

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
        'Geometry', 'geometry', 'Geometry', 'geometry'  # Remove Geometry column and its variations
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
            # $(document).ready(function() {{
            #     // Initialize the Delivery Units table with DataTables
            #     $('#delivery-units-table').DataTable({{
            #         paging: true,
            #         searching: true,
            #         ordering: true,
            #         info: true,
            #         lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
            #         dom: 'Bfrtip',
            #         buttons: [
            #             'copy', 'csv', 'excel'
            #         ],
            #         order: [[0, 'asc']], // Default sort by first column
            #         scrollX: true,
            #         colReorder: true, // Enable column reordering
            #         pageLength: 25 // Default to showing 25 rows
            #     }});
            # }});
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
                <div id="delivery-units-table-filters" style="margin-bottom: 20px;">
                    <label>Camping threshold: <input type="number" id="camping-threshold" value="12" min="1" style="width:60px;"></label>
                    <label style="margin-left:20px;"><input type="checkbox" id="filter-last7"> Checked In Last 7 Days</label>
                    <label style="margin-left:20px;"><input type="checkbox" id="filter-camping"> Camping Only</label>
                </div>
                <div class="table-info">
                    Showing {len(du_table_data):,} delivery units with status other than None and empty strings. Use the search box to filter.
                </div>
                <div class="delivery-units-table-container">
                    <table id="delivery-units-table" class="display">
                        <thead>
                            <tr>
    """
    
    # Add table headers (renaming as needed)
    header_map = {
        'flw_name': 'Flw Name',
        buildings_col: 'Buildings',
        delivery_count_col: 'Delivery Count',
        'flag_days_in_du': 'Flag Days In Du',
        'Camping': 'Camping',
        'checked_in_date': 'Checked In Date',
        'checked_out_date': 'Checked Out Date',
        'du_name': 'Du Name',
        'du_status': 'Du Status',
        'du_checkout_remark': 'Du Checkout Remark',
        'service_area_id': 'Service Area Id',
    }
    for col in reduced_columns:
        html_content += f"<th>{header_map.get(col, col)}</th>"
    html_content += """
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add table rows with data
    for _, row in du_table_data.iterrows():
        # Add data attributes for camping and checked_in_last_7_days
        camping_attr = 'true' if row.get('Camping', False) else 'false'
        checked_in_last_7_days_attr = 'true' if row.get('checked_in_last_7_days', False) else 'false'
        delivery_count_per_buildings = row.get('Delivery Count / Buildings', '')
        delivery_count = row.get(delivery_count_col, '')
        html_content += f"<tr data-camping='{camping_attr}' data-checked-in-last-7-days='{checked_in_last_7_days_attr}' data-delivery-count-per-buildings='{delivery_count_per_buildings}' data-delivery-count='{delivery_count}'>"
        for col in reduced_columns:
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
    
    # Create service area summary data
    service_area_data = []
    for sa_id, service_area in coverage_data.service_areas.items():
        last_activity = service_area.last_activity_date
        last_activity_str = last_activity.strftime('%Y-%m-%d') if last_activity else ''
        
        # Get assigned FLW IDs, filtering out empty strings
        assigned_flw_ids = sorted(list(set(du.flw_commcare_id for du in service_area.delivery_units if du.flw_commcare_id)))
        
        # Map IDs to names, falling back to ID if no name is found
        flw_name_map = coverage_data.flw_commcare_id_to_name_map
        assigned_flw_names = [flw_name_map.get(flw_id, flw_id) for flw_id in assigned_flw_ids]
        
        service_area_data.append({
            'service_area_id': sa_id,
            'total_buildings': service_area.total_buildings,
            'total_delivery_units': service_area.total_units,
            'completed_units': service_area.completed_units,
            'is_started': service_area.is_started,
            'is_completed': service_area.is_completed,
            'assigned_flws': ", ".join(assigned_flw_names),
            'last_activity_date': last_activity_str,
            'delivery_units': service_area.delivery_units
        })
    
    # Add Service Areas table section
    html_content += f"""
            <section>
                <h2>Service Areas Summary</h2>
                <div class="table-info">
                    Showing {len(service_area_data):,} service areas. Click on a service area name to view its delivery units.
                </div>
                <div class="service-areas-table-container">
                    <table id="service-areas-table" class="display">
                        <thead>
                            <tr>
                                <th>Service Area Name</th>
                                <th>Total Buildings</th>
                                <th>Total Delivery Units</th>
                                <th>Completed Units</th>
                                <th>Is Started</th>
                                <th>Is Completed</th>
                                <th>Assigned FLWs</th>
                                <th>Last Activity Date</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add table rows with data
    for sa_data in service_area_data:
        sa_id = sa_data['service_area_id']
        html_content += f"""
                            <tr>
                                <td><a href="#" class="service-area-link" data-sa-id={json.dumps(sa_id)}>{sa_id}</a></td>
                                <td>{sa_data['total_buildings']:,}</td>
                                <td>{sa_data['total_delivery_units']:,}</td>
                                <td>{sa_data['completed_units']:,}</td>
                                <td>{'Yes' if sa_data['is_started'] else 'No'}</td>
                                <td>{'Yes' if sa_data['is_completed'] else 'No'}</td>
                                <td>{sa_data['assigned_flws']}</td>
                                <td>{sa_data['last_activity_date']}</td>
                            </tr>
        """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </section>
    """
    
    # Add modal for service area details
    html_content += """
            <!-- Modal for Service Area Details -->
            <div id="serviceAreaModal" class="modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3 id="modalTitle">Service Area Details</h3>
                        <span class="close">&times;</span>
                    </div>
                    <div class="modal-body">
                        <div id="modalContent">
                            <!-- Content will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
            </div>
    """
    
    # Add CSS for modal
    html_content += """
        <style>
            /* Modal styles */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                overflow: auto;
                background-color: rgba(0,0,0,0.4);
            }
            
            .modal-content {
                background-color: #fefefe;
                margin: 5% auto;
                padding: 0;
                border: 1px solid #888;
                width: 80%;
                max-width: 800px;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .modal-header {
                background-color: #2980b9;
                color: white;
                padding: 15px 20px;
                border-radius: 5px 5px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .modal-header h3 {
                margin: 0;
                color: white;
            }
            
            .close {
                color: white;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .close:hover,
            .close:focus {
                color: #ddd;
            }
            
            .modal-body {
                padding: 20px;
                max-height: 70vh;
                overflow-y: auto;
            }
            
            .service-area-link {
                color: #2980b9;
                text-decoration: none;
                font-weight: bold;
            }
            
            .service-area-link:hover {
                text-decoration: underline;
                cursor: pointer;
            }
            
            .du-list {
                margin-top: 15px;
            }
            
            .du-item {
                padding: 8px 12px;
                margin: 5px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            
            .du-item.completed {
                background-color: #d4edda;
                border-color: #c3e6cb;
            }
            
            .du-item.visited {
                background-color: #fff3cd;
                border-color: #ffeaa7;
            }
            
            .du-item.unvisited {
                background-color: #f8d7da;
                border-color: #f5c6cb;
            }
            
            .du-name {
                font-weight: bold;
                color: #2c3e50;
            }
            
            .du-status {
                float: right;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .du-status.completed {
                background-color: #28a745;
                color: white;
            }
            
            .du-status.visited {
                background-color: #ffc107;
                color: #212529;
            }
            
            .du-status.unvisited {
                background-color: #dc3545;
                color: white;
            }
        </style>
    """
    
    # Add JavaScript for modal functionality
    html_content += """
        <script>
            // Service area data for modal
            var serviceAreaData = {
    """
    
    # Add service area data to JavaScript
    for sa_data in service_area_data:
        sa_id = sa_data['service_area_id']
        html_content += f"""
                {json.dumps(sa_id)}: {{
                    "total_buildings": {sa_data['total_buildings']},
                    "total_delivery_units": {sa_data['total_delivery_units']},
                    "completed_units": {sa_data['completed_units']},
                    "is_started": {str(sa_data['is_started']).lower()},
                    "is_completed": {str(sa_data['is_completed']).lower()},
                    "assigned_flws": {json.dumps(sa_data['assigned_flws'])},
                    "last_activity_date": {json.dumps(sa_data['last_activity_date'])},
                    "delivery_units": [
        """
        
        for du in sa_data['delivery_units']:
            status_class = du.status if du.status else 'unvisited'
            html_content += f"""
                        {{
                            "name": {json.dumps(du.du_name)},
                            "status": {json.dumps(du.status if du.status else 'unvisited')},
                            "status_class": {json.dumps(status_class)},
                            "buildings": {du.buildings},
                            "delivery_count": {du.delivery_count}
                        }},
            """
        
        html_content += """
                    ]
                },
        """
    
    html_content += """
            };
            
            // Modal functionality
            var modal = document.getElementById("serviceAreaModal");
            var span = document.getElementsByClassName("close")[0];
            
            // Close modal when clicking X
            span.onclick = function() {
                modal.style.display = "none";
            }
            
            // Close modal when clicking outside
            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }
            
            // Handle service area link clicks
            $(document).on('click', '.service-area-link', function(e) {
                e.preventDefault();
                var saId = $(this).data('sa-id');
                showServiceAreaDetails(saId);
            });
            
            function showServiceAreaDetails(saId) {
                var data = serviceAreaData[saId];
                if (!data) return;
                
                // Update modal title
                document.getElementById('modalTitle').textContent = 'Service Area: ' + saId;
                
                // Create content
                var content = '<div class="service-area-summary">';
                content += '<h4>Summary</h4>';
                content += '<p><strong>Total Buildings:</strong> ' + data.total_buildings.toLocaleString() + '</p>';
                content += '<p><strong>Total Delivery Units:</strong> ' + data.total_delivery_units.toLocaleString() + '</p>';
                content += '<p><strong>Completed Units:</strong> ' + data.completed_units.toLocaleString() + '</p>';
                content += '<p><strong>Assigned FLWs:</strong> ' + data.assigned_flws + '</p>';
                content += '<p><strong>Is Started:</strong> ' + (data.is_started ? 'Yes' : 'No') + '</p>';
                content += '<p><strong>Is Completed:</strong> ' + (data.is_completed ? 'Yes' : 'No') + '</p>';
                content += '<p><strong>Last Activity Date:</strong> ' + data.last_activity_date + '</p>';
                content += '</div>';
                
                content += '<div class="du-list">';
                content += '<h4>Delivery Units</h4>';
                
                data.delivery_units.forEach(function(du) {
                    content += '<div class="du-item ' + du.status_class + '">';
                    content += '<span class="du-name">' + du.name + '</span>';
                    content += '<span class="du-status ' + du.status_class + '">' + du.status + '</span>';
                    content += '<br><small>Buildings: ' + du.buildings + ' | Deliveries: ' + du.delivery_count + '</small>';
                    content += '</div>';
                });
                
                content += '</div>';
                
                document.getElementById('modalContent').innerHTML = content;
                modal.style.display = "block";
            }
        </script>
    """
    
    # Close the HTML
    html_content += f"""
            <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <!-- DataTables initialization script -->
        <script>
            $(document).ready(function() {{
                // Initialize the Delivery Units table
                if ($.fn.DataTable.isDataTable('#delivery-units-table')) {{
                    $('#delivery-units-table').DataTable().destroy();
                }}
                var duTable = $('#delivery-units-table').DataTable({{
                    paging: true,
                    searching: true,
                    ordering: true,
                    info: true,
                    lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                    dom: 'Bfrtip',
                    buttons: [
                        'copy', 'csv', 'excel'
                    ],
                    order: [[0, 'asc']],
                    scrollX: true,
                    colReorder: true,
                    pageLength: 25
                }});

                // Custom filter for Camping threshold, Checked In Last 7 Days, Camping Only
                $.fn.dataTable.ext.search.push(
                    function(settings, data, dataIndex) {{
                        if (settings.nTable.id !== 'delivery-units-table') return true;
                        var campingThreshold = parseFloat($('#camping-threshold').val()) || 12;
                        var filterLast7 = $('#filter-last7').prop('checked');
                        var filterCamping = $('#filter-camping').prop('checked');

                        // Get row node and data attributes
                        var rowNode = duTable.row(dataIndex).node();
                        var campingAttr = $(rowNode).attr('data-camping');
                        var checkedInLast7Attr = $(rowNode).attr('data-checked-in-last-7-days');
                        var deliveryCountPerBuildings = parseFloat($(rowNode).attr('data-delivery-count-per-buildings'));
                        var deliveryCount = parseFloat($(rowNode).attr('data-delivery-count'));

                        // Filter by Camping Only
                        if (filterCamping) {{
                            var isCamping = false;
                            if (!isNaN(deliveryCountPerBuildings) && !isNaN(deliveryCount)) {{
                                isCamping = (deliveryCountPerBuildings >= campingThreshold && deliveryCount > 20);
                            }}
                            if (!isCamping) return false;
                        }}
                        // Filter by Checked In Last 7 Days
                        if (filterLast7) {{
                            if (checkedInLast7Attr !== 'true') return false;
                        }}
                        return true;
                    }}
                );

                // Redraw table on filter change
                $('#camping-threshold, #filter-last7, #filter-camping').on('input change', function() {{
                    duTable.draw();
                }});
                
                // Initialize the Service Areas table
                $('#service-areas-table').DataTable({{
                    paging: true,
                    searching: true,
                    ordering: true,
                    info: true,
                    lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                    dom: 'Bfrtip',
                    buttons: [
                        'copy', 'csv', 'excel'
                    ],
                    order: [[0, 'asc']],
                    scrollX: true,
                    colReorder: true,
                    pageLength: 25
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content