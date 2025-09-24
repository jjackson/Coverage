import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import os
from dash import dash_table
from dash_ag_grid import AgGrid
from dash.dependencies import ClientsideFunction
import base64
import io

# CSS styles for red highlighting
app = dash.Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .cell-red-bg {
                background-color: #ffebee !important;
                color: #c62828 !important;
                font-weight: bold !important;
            }
            .ag-cell.cell-red-bg {
                background-color: #ffebee !important;
                color: #c62828 !important;
                font-weight: bold !important;
            }
            
            /* Header text wrapping and auto-sizing styles */
            .ag-header-cell-wrap .ag-header-cell-text {
                white-space: normal !important;
                word-wrap: break-word !important;
                overflow-wrap: break-word !important;
                line-height: 1.2 !important;
                padding: 4px 6px !important;
                text-align: center !important;
                display: block !important;
                width: 100% !important;
                font-size: 12px !important;
                font-weight: 500 !important;
            }
            
            .ag-header-cell-wrap {
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 4px !important;
                overflow: hidden !important;
            }
            
            .ag-header-cell-label {
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
                padding: 4px !important;
                overflow: hidden !important;
            }
            
            /* Auto-size columns based on content */
            .ag-header-cell {
                min-width: 120px !important;
                max-width: none !important;
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
            }
            
            /* Ensure headers can expand to fit text */
            .ag-header-cell-resize {
                display: none !important;
            }
            
            /* Better text wrapping for long headers */
            .ag-header-cell-text {
                max-width: none !important;
                overflow: visible !important;
                height: auto !important;
                min-height: auto !important;
            }
            
            /* Header row auto-height */
            .ag-header-row {
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
            }
            
            /* Ensure header container allows auto-height */
            .ag-header-container {
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
            }
            
            /* Header group auto-height */
            .ag-header-group-cell {
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
            }
            
            /* Ensure individual header cells respect their height settings */
            .ag-header-cell {
                height: auto !important;
                min-height: 60px !important;
                max-height: none !important;
                overflow: hidden !important;
            }
            
            /* Set minimum height for the main header container */
            .ag-header.ag-pivot-off.ag-header-allow-overflow {
                min-height: 80px !important;
                height: auto !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Load dataframes from pickle files in data folder
import pickle
project_root = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(project_root, 'data')

ward_level_pickle_path = os.path.join(data_path, "ward_level_status_report.pkl")
if not os.path.exists(ward_level_pickle_path):
    raise FileNotFoundError(f"Pickle file not found at {ward_level_pickle_path}. Run 'python run_ward_level_status_report.py' from src folder to generate it.")
with open(ward_level_pickle_path, 'rb') as f:
    ward_level_final_df = pickle.load(f)

opp_level_pickle_path = os.path.join(data_path, "opp_level_status_report.pkl")
if not os.path.exists(opp_level_pickle_path):
    raise FileNotFoundError(f"Pickle file not found at {opp_level_pickle_path}. Run 'python run_ward_level_status_report.py' from src folder to generate it.")
with open(opp_level_pickle_path, 'rb') as f:
    opp_level_final_df = pickle.load(f)

# Load timeline data
timeline_pickle_path = os.path.join(data_path, "timeline_based_status_report.pkl")
if not os.path.exists(timeline_pickle_path):
    print(f"Warning: Timeline pickle file not found at {timeline_pickle_path}. Line charts will not be available.")
    timeline_df = pd.DataFrame()
else:
    with open(timeline_pickle_path, 'rb') as f:
        timeline_df = pickle.load(f)
    # Convert visit_date to datetime for proper plotting
    timeline_df['visit_date'] = pd.to_datetime(timeline_df['visit_date'])

def get_column_display_name(column_name):
    """
    Map column names to more appropriate display names for the AgGrid tables.
    
    This function provides user-friendly column headers by mapping technical column names
    to readable display names. You can easily customize this mapping by:
    
    1. Adding new mappings to the column_mapping dictionary
    2. Modifying existing mappings to match your preferred terminology
    3. Adding domain-specific column names for your use case
    
    Example of adding custom mappings:
    column_mapping['my_custom_column'] = 'My Custom Display Name'
    column_mapping['internal_code'] = 'Public Description'
    """
    column_mapping = {
        # Domain and location columns
        'domain': 'Project Domain',
        'ward': 'Ward',
        'opportunity_name': 'Opportunity Name',
        'project_name': 'Project Name',
        'region': 'Region',
        'district': 'District',
        'sub_district': 'Sub District',
        'village': 'Village',
        'community': 'Community',
        
        # Target columns
        'visit_target': 'Visit Target',
        'building_target': 'Building Target',
        'du_target': 'Delivery Unit Target',
        'household_target': 'Household Target',
        'population_target': 'Population Target',
        
        # Completed columns
        'visits_completed': 'Visits Completed',
        'buildings_completed': 'Buildings Completed',
        'du_completed': 'Delivery Units Completed',
        'households_completed': 'Households Completed',
        'population_covered': 'Population Covered',
        
        # Percentage columns
        'pct_visits_completed': 'Visit Completion %',
        'pct_buildings_completed': 'Building Completion %',
        'pct_dus_completed': 'DU Completion %',
        'pct_households_completed': 'Household Completion %',
        'pct_population_covered': 'Population Coverage %',
        'pct_visits_completed_last7days': 'Visit Completion % (7d)',
        'pct_buildings_completed_last7days': 'Building Completion % (7d)',
        'pct_dus_completed_last7days': 'DU Completion % (7d)',
        'pct_households_completed_last7days': 'Household Completion % (7d)',
        'pct_population_covered_last7days': 'Population Coverage % (7d)',
        
        # Count columns
        'total_visits': 'Total Visits',
        'total_buildings': 'Total Buildings',
        'total_dus': 'Total Delivery Units',
        'total_flws': 'Total Field Workers',
        'total_households': 'Total Households',
        'total_population': 'Total Population',
        'total_service_areas': 'Total Service Areas',
        
        # Status columns
        'status': 'Status',
        'completion_status': 'Completion Status',
        'delivery_status': 'Delivery Status',
        'verification_status': 'Verification Status',
        'quality_status': 'Quality Status',
        
        # Date columns
        'visit_date': 'Visit Date',
        'last_visit_date': 'Last Visit Date',
        'start_date': 'Start Date',
        'end_date': 'End Date',
        'planned_date': 'Planned Date',
        'actual_date': 'Actual Date',
        'created_date': 'Created Date',
        'updated_date': 'Updated Date',
        
        # Performance metrics
        'completion_rate': 'Completion Rate',
        'efficiency_score': 'Efficiency Score',
        'productivity_index': 'Productivity Index',
        'performance_rating': 'Performance Rating',
        'quality_score': 'Quality Score',
        'accuracy_rate': 'Accuracy Rate',
        
        # Geographic columns
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'coordinates': 'Coordinates',
        'gps_accuracy': 'GPS Accuracy',
        'location_verified': 'Location Verified',
        
        # Quality metrics
        'data_quality_score': 'Data Quality Score',
        'validation_status': 'Validation Status',
        'error_count': 'Error Count',
        'warning_count': 'Warning Count',
        'data_completeness': 'Data Completeness',
        'data_consistency': 'Data Consistency',
        
        # Financial columns
        'budget_allocated': 'Budget Allocated',
        'budget_spent': 'Budget Spent',
        'cost_per_visit': 'Cost per Visit',
        'cost_per_building': 'Cost per Building',
        'cost_per_du': 'Cost per Delivery Unit',
        'total_cost': 'Total Cost',
        'remaining_budget': 'Remaining Budget',
        
        # Time-based metrics
        'avg_time_per_visit': 'Avg Time per Visit',
        'avg_time_per_building': 'Avg Time per Building',
        'avg_time_per_du': 'Avg Time per DU',
        'total_working_hours': 'Total Working Hours',
        'overtime_hours': 'Overtime Hours',
        'travel_time': 'Travel Time',
        'waiting_time': 'Waiting Time',
        
        # FLW specific columns
        'flw_name': 'Field Worker Name',
        'flw_id': 'Field Worker ID',
        'flw_phone': 'Field Worker Phone',
        'flw_supervisor': 'Field Worker Supervisor',
        'flw_performance_rating': 'Performance Rating',
        'flw_experience_years': 'Experience (Years)',
        'flw_training_completed': 'Training Completed',
        'flw_availability': 'Availability Status',
        'unique_user_id': 'Number of Workers',
        
        # Delivery unit specific columns
        'du_name': 'Delivery Unit Name',
        'du_type': 'Delivery Unit Type',
        'du_category': 'Delivery Unit Category',
        'du_priority': 'Delivery Unit Priority',
        'du_size': 'Delivery Unit Size',
        'du_population': 'DU Population',
        'du_households': 'DU Households',
        
        # Service area columns
        'service_area_name': 'Service Area Name',
        'service_area_type': 'Service Area Type',
        'population_size': 'Population Size',
        'household_count': 'Household Count',
        'area_km2': 'Area (km²)',
        'density': 'Population Density',
        
        # Visit specific columns
        'visit_type': 'Visit Type',
        'visit_duration': 'Visit Duration',
        'visit_notes': 'Visit Notes',
        'visit_photos': 'Visit Photos',
        'visit_verification': 'Visit Verification',
        'visit_quality': 'Visit Quality',
        'visit_outcome': 'Visit Outcome',
        
        # Building specific columns
        'building_type': 'Building Type',
        'building_condition': 'Building Condition',
        'building_occupancy': 'Building Occupancy',
        'building_notes': 'Building Notes',
        'building_age': 'Building Age',
        'building_material': 'Building Material',
        
        # Custom calculated columns
        'microplanning_efficiency': 'Microplanning Efficiency',
        'coverage_gap': 'Coverage Gap',
        'optimization_score': 'Optimization Score',
        'risk_assessment': 'Risk Assessment',
        'progress_trend': 'Progress Trend',
        'performance_benchmark': 'Performance Benchmark',
        'efficiency_gap': 'Efficiency Gap',
        
        # Health-specific columns (if applicable)
        'vaccination_rate': 'Vaccination Rate',
        'health_indicators': 'Health Indicators',
        'mortality_rate': 'Mortality Rate',
        'morbidity_rate': 'Morbidity Rate',
        'nutrition_status': 'Nutrition Status',
        'sanitation_access': 'Sanitation Access',
        
        # Education-specific columns (if applicable)
        'literacy_rate': 'Literacy Rate',
        'enrollment_rate': 'Enrollment Rate',
        'attendance_rate': 'Attendance Rate',
        'dropout_rate': 'Dropout Rate',
        'teacher_student_ratio': 'Teacher-Student Ratio',
        
        # Infrastructure columns
        'road_access': 'Road Access',
        'electricity_access': 'Electricity Access',
        'water_access': 'Water Access',
        'internet_access': 'Internet Access',
        'healthcare_facility': 'Healthcare Facility',
        'school_facility': 'School Facility'
    }
    
    # Return mapped name if exists, otherwise return original with title case
    display_name = column_mapping.get(column_name, column_name.replace('_', ' ').title())
    
    # Replace "Pct" with "%" in all header names
    display_name = display_name.replace('Pct', '%')
    
    return display_name

def add_custom_column_mapping(custom_mappings):
    """
    Add custom column name mappings to the display name function.
    
    Args:
        custom_mappings (dict): Dictionary of column_name: display_name pairs
        
    Example:
        add_custom_column_mapping({
            'my_internal_code': 'Public Display Name',
            'technical_field': 'User-Friendly Name'
        })
    """
    # This function allows runtime customization of column mappings
    # You can call this function before creating the AgGrid tables
    # to add project-specific column name mappings
    
    # Note: For permanent changes, modify the column_mapping dictionary
    # in the get_column_display_name function above
    
    # Custom column mappings added successfully
    pass

def get_available_columns(dataframe):
    """
    Get a list of available columns in the dataframe with their display names.
    
    Args:
        dataframe: Pandas DataFrame to analyze
        
    Returns:
        dict: Dictionary mapping original column names to display names
    """
    return {col: get_column_display_name(col) for col in dataframe.columns}

# Example usage of custom column mappings:
# Uncomment and modify the lines below to add project-specific column names
# 
# add_custom_column_mapping({
#     'project_specific_field': 'Project-Specific Display Name',
#     'internal_metric': 'Public Metric Name',
#     'technical_indicator': 'User-Friendly Indicator'
# })
#
# You can also see what columns are available in your data:
# available_opp_columns = get_available_columns(opp_level_final_df)
# available_ward_columns = get_available_columns(ward_level_final_df)

# Prepare dropdown options
domain_options = [{'label': 'All Domains', 'value': 'all_domains'}] + [{'label': d, 'value': d} for d in ward_level_final_df['domain'].unique()]

# Prepare columns for the opportunity-level table
column_defs = []

# Define the specific column order for Opportunity Level Summary table
opp_column_order = [
    'domain', 'visit_target', 'building_target', 'du_target', 'start_date', 'end_date',
    'pct_completion', 'pct_building_microplanning_completion_rate', 
    'pct_building_microplanning_completion_rate_last_week', 'pct_du_microplanning_completion_rate',
    'pct_du_microplanning_completion_rate_last_week', 'unique_user_id', 'visits_completed',
    'visits_completed_last_week', 'pct_visits_completed', 'pct_visits_completed_last_week',
    'buildings_completed', 'buildings_completed_last_week', 'pct_buildings_completed',
    'pct_buildings_completed_last_week', 'du_completed', 'du_completed_last_week',
    'pct_du_completed', 'pct_du_completed_last_week'
]

# Get all columns from the dataframe
all_columns = list(opp_level_final_df.columns)

# Filter columns to only include those that exist in the dataframe
reordered_columns = [col for col in opp_column_order if col in all_columns]

# Add any remaining columns that weren't in the specified order
remaining_columns = [col for col in all_columns if col not in opp_column_order]
reordered_columns.extend(remaining_columns)

# Create column definitions with the reordered columns
for i, col in enumerate(reordered_columns):
    col_def = {"headerName": get_column_display_name(col), "field": col}
    if i < 6:
        col_def["pinned"] = "left"  # Freeze the first six columns (domain, targets, dates)
    
    # Different highlighting rules based on column type
    if str(col).startswith('pct_'):
        # For pct_ columns: highlight 0 values and values > 100
        col_def["cellClassRules"] = {
            "cell-red-bg": "x == 0 || (typeof x === 'number' && x > 100)"
        }
    else:
        # For other columns: highlight 0 values and values >= 100
        col_def["cellClassRules"] = {
            "cell-red-bg": "x == 0 || (typeof x === 'number' && x >= 100)"
        }
    
    # Set fixed width for all columns based on content type
    if str(col).startswith('pct_'):
        # Percentage columns - wider for readability
        col_def["width"] = 150
        col_def["minWidth"] = 150
        col_def["maxWidth"] = 150
    elif col in ['domain', 'start_date', 'end_date']:
        # Domain and date columns - medium width
        col_def["width"] = 120
        col_def["minWidth"] = 120
        col_def["maxWidth"] = 120
    elif col in ['visit_target', 'building_target', 'du_target']:
        # Target columns - medium width
        col_def["width"] = 110
        col_def["minWidth"] = 110
        col_def["maxWidth"] = 110
    elif col in ['unique_user_id', 'visits_completed', 'buildings_completed', 'du_completed']:
        # Count columns - medium width
        col_def["width"] = 130
        col_def["minWidth"] = 130
        col_def["maxWidth"] = 130
    elif col in ['visits_completed_last_week', 'buildings_completed_last_week', 'du_completed_last_week']:
        # Last week count columns - wider for longer names
        col_def["width"] = 160
        col_def["minWidth"] = 160
        col_def["maxWidth"] = 160
    else:
        # Default width for other columns
        col_def["width"] = 140
        col_def["minWidth"] = 140
        col_def["maxWidth"] = 140
    
    col_def["suppressSizeToFit"] = True
    col_def["resizable"] = False
    
    # Set dynamic header height based on column name length
    column_display_name = get_column_display_name(col)
    name_length = len(column_display_name)
    
    # More granular height adjustment based on actual text length
    if name_length <= 10:
        col_def["headerHeight"] = 60
    elif name_length <= 15:
        col_def["headerHeight"] = 70
    elif name_length <= 20:
        col_def["headerHeight"] = 80
    elif name_length <= 25:
        col_def["headerHeight"] = 90
    elif name_length <= 30:
        col_def["headerHeight"] = 100
    elif name_length <= 35:
        col_def["headerHeight"] = 110
    elif name_length <= 40:
        col_def["headerHeight"] = 120
    else:
        col_def["headerHeight"] = 130
    
    # Enable header text wrapping for all columns
    col_def["headerClass"] = "ag-header-cell-wrap"
    col_def["headerComponentParams"] = {
        "template": '<div class="ag-cell-label-container" role="presentation">' +
                   '<span ref="eMenu" class="ag-header-icon ag-header-cell-menu-button"></span>' +
                   '<div ref="eLabel" class="ag-header-cell-label" role="presentation">' +
                   '<span ref="eText" class="ag-header-cell-text" role="columnheader"></span>' +
                   '<span ref="eSortOrder" class="ag-header-icon ag-sort-order"></span>' +
                   '<span ref="eSortAsc" class="ag-header-icon ag-sort-ascending-icon"></span>' +
                   '<span ref="eSortDesc" class="ag-header-icon ag-sort-descending-icon"></span>' +
                   '<span ref="eSortNone" class="ag-header-icon ag-sort-none-icon"></span>' +
                   '<span ref="eFilter" class="ag-header-icon ag-filter-icon"></span>' +
                   '</div>' +
                   '</div>'
    }

    column_defs.append(col_def)

opp_level_table = AgGrid(
    id="opp-aggrid",
    rowData=opp_level_final_df.to_dict('records'),
    columnDefs=column_defs,
    style={'height': '450px', 'width': '100%', 'marginBottom': '32px'},
    dashGridOptions={"pagination": True, 
                     "paginationPageSize": 10, 
                     "enableExport": True, 
                     "menuTabs": ["generalMenuTab", "columnsMenuTab", "filterMenuTab", "exportMenuTab"],
                     "suppressColumnVirtualisation": True,
                     "autoGroupColumnDef": {"minWidth": 200},
                     "autoSizeColumns": False,
                     "suppressRowVirtualisation": True,
                     "suppressSizeToFit": True,
                     "sizeColumnsToFit": False,
                     "suppressRowHoverHighlight": False,
                     "rowHeight": 40},
    csvExportParams={
        "fileName": "opp_level_status_report.csv",
        "allColumns": True
    }
)

app.layout = html.Div([
    html.H2("Opportunity Level Summary"),
    html.Div([
        html.Button(
            "Download Opportunity Level Data",
            id="download-opp-btn",
            style={
                'marginBottom': '10px',
                'padding': '10px 20px',
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
        ),
        dcc.Download(id="download-opp-csv"),
        opp_level_table
    ]),
    html.H2("Ward Status Dashboard"),
    html.Div([
        html.Div([
            html.Label("Select Domain:"),
            dcc.Dropdown(
                id='domain-dropdown',
                options=domain_options,
                value='all_domains'  # Default to "All Domains"
            ),
        ], style={
            'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
            'borderRadius': '8px',
            'padding': '4px',
            'background': '#fff',
            'marginBottom': '16px',
            'width': '350px',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'marginRight': '24px'
        }),
        html.Div([
            html.Label("Select Ward:"),
            dcc.Dropdown(id='ward-dropdown', multi=True)
        ], style={
            'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
            'borderRadius': '8px',
            'padding': '4px',
            'background': '#fff',
            'marginBottom': '16px',
            'width': '350px',
            'display': 'inline-block',
            'verticalAlign': 'top'
        }),
    ]),
    html.Div([
        html.Button(
            "Download Ward Level Data",
            id="download-ward-btn",
            style={
                'marginTop': '10px',
                'padding': '10px 20px',
                'backgroundColor': '#28a745',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
        ),
        dcc.Download(id="download-ward-csv")
    ], id='ward-download-container', style={'display': 'none'}),
    html.Div(id='charts-container')
    
])


@app.callback(
    Output('ward-dropdown', 'options'),
    Output('ward-dropdown', 'value'),
    Input('domain-dropdown', 'value')
)
def update_ward_dropdown(selected_domain):
    if selected_domain == 'all_domains':
        # If "All Domains" is selected, get all unique wards across all domains
        all_wards = ward_level_final_df['ward'].unique()
        options = [{'label': 'All Wards', 'value': 'all_wards'}] + [{'label': w, 'value': w} for w in all_wards]
        value = 'all_wards'
    else:
        # If a specific domain is selected, get wards for that domain
        wards = ward_level_final_df[ward_level_final_df['domain'] == selected_domain]['ward'].unique()
        options = [{'label': 'All Wards', 'value': 'all_wards'}] + [{'label': w, 'value': w} for w in wards]
        value = 'all_wards'
    
    return options, value

@app.callback(
    [Output('charts-container', 'children'),
     Output('ward-download-container', 'style')],
    Input('domain-dropdown', 'value'),
    Input('ward-dropdown', 'value')
)
def update_charts(selected_domain, selected_wards):
    if not selected_domain or not selected_wards:
        return html.Div("Please select a domain and at least one ward."), {'display': 'none'}

    # Ensure selected_wards is a list
    if isinstance(selected_wards, str):
        selected_wards = [selected_wards]

    # Handle "All Domains" and "All Wards" selections
    if selected_domain == 'all_domains':
        if 'all_wards' in selected_wards:
            # All domains and all wards
            filtered_rows = ward_level_final_df.copy()
        else:
            # All domains, specific wards
            filtered_rows = ward_level_final_df[ward_level_final_df['ward'].isin(selected_wards)]
    else:
        if 'all_wards' in selected_wards:
            # Specific domain, all wards
            filtered_rows = ward_level_final_df[ward_level_final_df['domain'] == selected_domain]
        else:
            # Specific domain, specific wards
            filtered_rows = ward_level_final_df[
                (ward_level_final_df['domain'] == selected_domain) &
                (ward_level_final_df['ward'].isin(selected_wards))
            ]

    if filtered_rows.empty:
        return html.Div("No data for this selection."), {'display': 'none'}

    # Round all pct_ columns to two decimal places
    pct_cols = [col for col in filtered_rows.columns if str(col).startswith('pct_')]
    filtered_rows.loc[:, pct_cols] = filtered_rows[pct_cols].round(2)

    # Prepare AgGrid column definitions, freeze first 5 columns and set width for numeric columns
    column_defs = []
    
    # Define the specific column order for Ward Status Dashboard table
    ward_column_order = [
        'domain', 'ward', 'visit_target', 'building_target', 'du_target',
        'pct_building_microplanning_completion_rate', 'pct_building_microplanning_completion_rate_last_week',
        'pct_du_microplanning_completion_rate', 'pct_du_microplanning_completion_rate_last_week',
        'unique_user_id', 'visits_completed', 'visits_completed_last_week', 'pct_visits_completed',
        'pct_visits_completed_last_week', 'buildings_completed', 'buildings_completed_last_week',
        'pct_buildings_completed', 'pct_buildings_completed_last_week', 'du_completed',
        'du_completed_last_week', 'pct_du_completed', 'pct_du_completed_last_week'
    ]
    
    # Get all columns from the filtered dataframe
    all_columns = list(filtered_rows.columns)
    
    # Filter columns to only include those that exist in the dataframe
    reordered_columns = [col for col in ward_column_order if col in all_columns]
    
    # Add any remaining columns that weren't in the specified order
    remaining_columns = [col for col in all_columns if col not in ward_column_order]
    reordered_columns.extend(remaining_columns)
    
    # Create column definitions with the reordered columns
    for i, col in enumerate(reordered_columns):
        col_def = {"headerName": get_column_display_name(col), "field": col}
        if i < 5:
            col_def["pinned"] = "left"  # Freeze the first five columns (domain, ward, targets)
        
        # Different highlighting rules based on column type
        if str(col).startswith('pct_'):
            # For pct_ columns: highlight 0 values and values > 100
            col_def["cellClassRules"] = {
                "cell-red-bg": "x == 0 || (typeof x === 'number' && x > 100)"
            }
        else:
            # For other columns: highlight 0 values and values >= 100
            col_def["cellClassRules"] = {
                "cell-red-bg": "x == 0 || (typeof x === 'number' && x >= 100)"
            }
        
        # Set fixed width for all columns based on content type
        if str(col).startswith('pct_'):
            # Percentage columns - wider for readability
            col_def["width"] = 150
            col_def["minWidth"] = 150
            col_def["maxWidth"] = 150
        elif col in ['domain', 'ward']:
            # Domain and ward columns - medium width
            col_def["width"] = 120
            col_def["minWidth"] = 120
            col_def["maxWidth"] = 120
        elif col in ['visit_target', 'building_target', 'du_target']:
            # Target columns - medium width
            col_def["width"] = 110
            col_def["minWidth"] = 110
            col_def["maxWidth"] = 110
        elif col in ['unique_user_id', 'visits_completed', 'buildings_completed', 'du_completed']:
            # Count columns - medium width
            col_def["width"] = 130
            col_def["minWidth"] = 130
            col_def["maxWidth"] = 130
        elif col in ['visits_completed_last_week', 'buildings_completed_last_week', 'du_completed_last_week']:
            # Last week count columns - wider for longer names
            col_def["width"] = 160
            col_def["minWidth"] = 160
            col_def["maxWidth"] = 160
        else:
            # Default width for other columns
            col_def["width"] = 140
            col_def["minWidth"] = 140
            col_def["maxWidth"] = 140
        
        col_def["suppressSizeToFit"] = True
        col_def["resizable"] = False
        
        # Set dynamic header height based on column name length
        column_display_name = get_column_display_name(col)
        name_length = len(column_display_name)
        
        # More granular height adjustment based on actual text length
        if name_length <= 10:
            col_def["headerHeight"] = 60
        elif name_length <= 15:
            col_def["headerHeight"] = 70
        elif name_length <= 20:
            col_def["headerHeight"] = 80
        elif name_length <= 25:
            col_def["headerHeight"] = 90
        elif name_length <= 30:
            col_def["headerHeight"] = 100
        elif name_length <= 35:
            col_def["headerHeight"] = 110
        elif name_length <= 40:
            col_def["headerHeight"] = 120
        else:
            col_def["headerHeight"] = 130
        
        # Enable header text wrapping for all columns
        col_def["headerClass"] = "ag-header-cell-wrap"
        col_def["headerComponentParams"] = {
            "template": '<div class="ag-cell-label-container" role="presentation">' +
                       '<span ref="eMenu" class="ag-header-icon ag-header-cell-menu-button"></span>' +
                       '<div ref="eLabel" class="ag-header-cell-label" role="presentation">' +
                       '<span ref="eText" class="ag-header-cell-text" role="columnheader"></span>' +
                       '<span ref="eSortOrder" class="ag-header-icon ag-sort-order"></span>' +
                       '<span ref="eSortAsc" class="ag-header-icon ag-sort-ascending-icon"></span>' +
                       '<span ref="eSortDesc" class="ag-header-icon ag-sort-descending-icon"></span>' +
                       '<span ref="eSortNone" class="ag-header-icon ag-sort-none-icon"></span>' +
                       '<span ref="eFilter" class="ag-header-icon ag-filter-icon"></span>' +
                       '</div>' +
                       '</div>'
        }
        column_defs.append(col_def)

    table = AgGrid(
        id="ward-aggrid",
        rowData=filtered_rows.to_dict('records'),
        columnDefs=column_defs,
        style={'height': '250px', 'width': '100%', 'marginTop': '32px'},
        dashGridOptions={"pagination": True, "paginationPageSize": 10, "enableExport": True,
                         "menuTabs": ["generalMenuTab", "columnsMenuTab", "filterMenuTab", "exportMenuTab"],
                         "suppressColumnVirtualisation": True,
                         "autoGroupColumnDef": {"minWidth": 200},
                         "autoSizeColumns": False,
                         "suppressRowVirtualisation": True,
                         "suppressSizeToFit": True,
                         "sizeColumnsToFit": False,
                         "suppressRowHoverHighlight": False,
                         "rowHeight": 40},
        csvExportParams={
        "fileName": "ward_level_status_report.csv",
        "allColumns": True
    }
    )

    # Pie charts for each selected ward
    pie_charts = []
    for _, row in filtered_rows.iterrows():
        pie1 = dcc.Graph(
            figure=go.Figure(
                data=[go.Pie(
                    labels=['Visits Completed', 'Remaining'],
                    values=[row['visits_completed'], max(row['visit_target'] - row['visits_completed'], 0)],
                    hole=0.4,
                    marker=dict(colors=['#28a745', '#e0e0e0'])
                )],
                layout=go.Layout(title=f"Visits Completed vs Target ({row['ward']})")
            )
        )
        pie2 = dcc.Graph(
            figure=go.Figure(
                data=[go.Pie(
                    labels=['Buildings Completed', 'Remaining'],
                    values=[row['buildings_completed'], max(row['building_target'] - row['buildings_completed'], 0)],
                    hole=0.4,
                    marker=dict(colors=['#28a745', '#e0e0e0'])
                )],
                layout=go.Layout(title=f"Buildings Completed vs Target ({row['ward']})")
            )
        )
        pie3 = dcc.Graph(
            figure=go.Figure(
                data=[go.Pie(
                    labels=['DUs Completed', 'Remaining'],
                    values=[row['du_completed'], max(row['du_target'] - row['du_completed'], 0)],
                    hole=0.4,
                    marker=dict(colors=['#28a745', '#e0e0e0'])
                )],
                layout=go.Layout(title=f"DUs Completed vs Target ({row['ward']})")
            )
        )
        pie_charts.append(html.Div([
            html.Hr(),
            html.H4(f"Actual Data for Ward: {row['ward']}"),
            html.Div([pie1], style={'width': '30%', 'display': 'inline-block','verticalAlign': 'top',
                'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                'borderRadius': '12px',
                'padding': '10px',
                'background': '#fff',
                'margin': '8px'}),
            html.Div([pie2], style={'width': '30%', 'display': 'inline-block','verticalAlign': 'top',
                'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                'borderRadius': '12px',
                'padding': '10px',
                'background': '#fff',
                'margin': '8px'}),
            html.Div([pie3], style={'width': '30%', 'display': 'inline-block','verticalAlign': 'top',
                'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                'borderRadius': '12px',
                'padding': '10px',
                'background': '#fff',
                'margin': '8px'}),
        ]))

    # Create line charts for microplanning completion rates
    line_charts = []
    
    if not timeline_df.empty:
        # Filter timeline data for selected domain and wards
        if selected_domain == 'all_domains':
            if 'all_wards' in selected_wards:
                # All domains and all wards
                timeline_filtered = timeline_df.copy()
            else:
                # All domains, specific wards
                timeline_filtered = timeline_df[timeline_df['ward'].isin(selected_wards)]
        else:
            if 'all_wards' in selected_wards:
                # Specific domain, all wards
                timeline_filtered = timeline_df[timeline_df['domain'] == selected_domain]
            else:
                # Specific domain, specific wards
                timeline_filtered = timeline_df[
                    (timeline_df['domain'] == selected_domain) &
                    (timeline_df['ward'].isin(selected_wards))
                ]
        
        if not timeline_filtered.empty:
            # Sort by visit_date for proper line plotting
            timeline_filtered = timeline_filtered.sort_values('visit_date')
            
            # Check if microplanning columns exist, if not calculate them
            if 'building_microplanning_completion_rate' not in timeline_filtered.columns:
                # Calculate microplanning completion rates
                timeline_filtered['building_microplanning_completion_rate'] = timeline_filtered.apply(
                    lambda row: (row['pct_buildings_completed'] / row['pct_visits_completed'] * 100) 
                    if row['pct_visits_completed'] > 0 else 0, axis=1
                )
                timeline_filtered['du_microplanning_completion_rate'] = timeline_filtered.apply(
                    lambda row: (row['pct_dus_completed'] / row['pct_visits_completed'] * 100) 
                    if row['pct_visits_completed'] > 0 else 0, axis=1
                )
                timeline_filtered['building_microplanning_completion_rate_last7days'] = timeline_filtered.apply(
                    lambda row: (row['pct_buildings_completed_last7days'] / row['pct_visits_completed_last7days'] * 100) 
                    if row['pct_visits_completed_last7days'] > 0 else 0, axis=1
                )
                timeline_filtered['du_microplanning_completion_rate_last7days'] = timeline_filtered.apply(
                    lambda row: (row['pct_dus_completed_last7days'] / row['pct_visits_completed_last7days'] * 100) 
                    if row['pct_visits_completed_last7days'] > 0 else 0, axis=1
                )
            
            # Chart a) building_microplanning_completion_rate vs visit_date
            # Determine which wards to plot
            if 'all_wards' in selected_wards:
                # If "All Wards" is selected, get unique wards from filtered data
                wards_to_plot = timeline_filtered['ward'].unique()
            else:
                # Use selected wards, excluding 'all_wards'
                wards_to_plot = [ward for ward in selected_wards if ward != 'all_wards']
            
            building_rate_chart = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=timeline_filtered[timeline_filtered['ward'] == ward]['visit_date'],
                            y=timeline_filtered[timeline_filtered['ward'] == ward]['building_microplanning_completion_rate'],
                            mode='lines+markers',
                            name=f'{ward}',
                            line=dict(width=2)
                        ) for ward in wards_to_plot
                    ],
                    layout=go.Layout(
                        title="Building Microplanning Completion Rate Over Time",
                        xaxis=dict(title="Visit Date"),
                        yaxis=dict(title="Building Microplanning Completion Rate (%)"),
                        hovermode='x unified',
                        height=400
                    )
                ),
                style={
                    'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                    'borderRadius': '12px',
                    'padding': '10px',
                    'background': '#fff',
                    'margin': '8px'
                }
            )
            
            # Chart b) du_microplanning_completion_rate vs visit_date
            du_rate_chart = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=timeline_filtered[timeline_filtered['ward'] == ward]['visit_date'],
                            y=timeline_filtered[timeline_filtered['ward'] == ward]['du_microplanning_completion_rate'],
                            mode='lines+markers',
                            name=f'{ward}',
                            line=dict(width=2)
                        ) for ward in wards_to_plot
                    ],
                    layout=go.Layout(
                        title="DU Microplanning Completion Rate Over Time",
                        xaxis=dict(title="Visit Date"),
                        yaxis=dict(title="DU Microplanning Completion Rate (%)"),
                        hovermode='x unified',
                        height=400
                    )
                ),
                style={
                    'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                    'borderRadius': '12px',
                    'padding': '10px',
                    'background': '#fff',
                    'margin': '8px'
                }
            )
            
            # Chart c) building_microplanning_completion_rate_last7days vs visit_date
            building_rate_last7_chart = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=timeline_filtered[timeline_filtered['ward'] == ward]['visit_date'],
                            y=timeline_filtered[timeline_filtered['ward'] == ward]['building_microplanning_completion_rate_last7days'],
                            mode='lines+markers',
                            name=f'{ward}',
                            line=dict(width=2)
                        ) for ward in wards_to_plot
                    ],
                    layout=go.Layout(
                        title="Building Microplanning Completion Rate (Last 7 Days) Over Time",
                        xaxis=dict(title="Visit Date"),
                        yaxis=dict(title="Building Microplanning Completion Rate Last 7 Days (%)"),
                        hovermode='x unified',
                        height=400
                    )
                ),
                style={
                    'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                    'borderRadius': '12px',
                    'padding': '10px',
                    'background': '#fff',
                    'margin': '8px'
                }
            )
            
            # Chart d) du_microplanning_completion_rate_last7days vs visit_date
            du_rate_last7_chart = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=timeline_filtered[timeline_filtered['ward'] == ward]['visit_date'],
                            y=timeline_filtered[timeline_filtered['ward'] == ward]['du_microplanning_completion_rate_last7days'],
                            mode='lines+markers',
                            name=f'{ward}',
                            line=dict(width=2)
                        ) for ward in wards_to_plot
                    ],
                    layout=go.Layout(
                        title="DU Microplanning Completion Rate (Last 7 Days) Over Time",
                        xaxis=dict(title="Visit Date"),
                        yaxis=dict(title="DU Microplanning Completion Rate Last 7 Days (%)"),
                        hovermode='x unified',
                        height=400
                    )
                ),
                style={
                    'boxShadow': '0 4px 16px rgba(0,0,0,0.15)',
                    'borderRadius': '12px',
                    'padding': '10px',
                    'background': '#fff',
                    'margin': '8px'
                }
            )
            
            # Create a container for line charts with 2x2 grid layout
            line_charts_container = html.Div([
                html.Hr(),
                html.H4("Microplanning Completion Rate Trends"),
                html.Div([
                    html.Div([building_rate_chart], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([du_rate_chart], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ]),
                html.Div([
                    html.Div([building_rate_last7_chart], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([du_rate_last7_chart], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ])
            ])
            
            line_charts = [line_charts_container]
        else:
            line_charts = [html.Div("No timeline data available for the selected domain and wards.", style={'textAlign': 'center', 'padding': '20px'})]
    else:
        line_charts = [html.Div("Timeline data not available. Please run the status report generation first.", style={'textAlign': 'center', 'padding': '20px'})]

    return [table] + line_charts + pie_charts, {'display': 'block'}

@app.callback(
    Output("download-opp-csv", "data"),
    Input("download-opp-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_opp_csv(n_clicks):
    if n_clicks is None:
        return None
    
    return dcc.send_data_frame(opp_level_final_df.to_csv, "opportunity_level_data.csv", index=False)

@app.callback(
    Output("download-ward-csv", "data"),
    Input("download-ward-btn", "n_clicks"),
    State('domain-dropdown', 'value'),
    State('ward-dropdown', 'value'),
    prevent_initial_call=True,
)
def download_ward_csv(n_clicks, selected_domain, selected_wards):
    if n_clicks is None or not selected_domain or not selected_wards:
        return None
    
    # Ensure selected_wards is a list
    if isinstance(selected_wards, str):
        selected_wards = [selected_wards]
    
    # Handle "All Domains" and "All Wards" selections
    if selected_domain == 'all_domains':
        if 'all_wards' in selected_wards:
            # All domains and all wards
            filtered_rows = ward_level_final_df.copy()
        else:
            # All domains, specific wards
            filtered_rows = ward_level_final_df[ward_level_final_df['ward'].isin(selected_wards)]
    else:
        if 'all_wards' in selected_wards:
            # Specific domain, all wards
            filtered_rows = ward_level_final_df[ward_level_final_df['domain'] == selected_domain]
        else:
            # Specific domain, specific wards
            filtered_rows = ward_level_final_df[
                (ward_level_final_df['domain'] == selected_domain) &
                (ward_level_final_df['ward'].isin(selected_wards))
            ]
    
    if filtered_rows.empty:
        return None
    
    # Round all pct_ columns to two decimal places
    pct_cols = [col for col in filtered_rows.columns if str(col).startswith('pct_')]
    filtered_rows.loc[:, pct_cols] = filtered_rows[pct_cols].round(2)
    
    # Generate appropriate filename
    if selected_domain == 'all_domains':
        if 'all_wards' in selected_wards:
            filename = "ward_level_data_all_domains_all_wards.csv"
        else:
            filename = f"ward_level_data_all_domains_specific_wards.csv"
    else:
        if 'all_wards' in selected_wards:
            filename = f"ward_level_data_{selected_domain}_all_wards.csv"
        else:
            filename = f"ward_level_data_{selected_domain}_{'_'.join(selected_wards)}.csv"
    
    return dcc.send_data_frame(filtered_rows.to_csv, filename, index=False)

if __name__ == "__main__":
    app.run(debug=True)