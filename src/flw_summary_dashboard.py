import logging
import os
import dash, json
from dash import html, dcc, Input, Output, State, no_update
import pandas as pd
from dash import dash_table
from src.org_summary import generate_summary
from datetime import timedelta
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from dash_ag_grid import AgGrid
from src.flw_pathways import integrate_with_flw_dashboard
from geopy.distance import geodesic
import folium

find_dotenv()
load_dotenv(override=True,verbose=True)
org_values = list(json.loads(os.getenv("OPPORTUNITY_DOMAIN_MAPPING")).values())

log_dir = '../../'
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
log_file_path = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    filename= log_file_path,           # Log file name
    filemode='a',                 # Append mode ('w' to overwrite)
    level=logging.INFO,           # Minimum log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_overall_opp_median_metrics_per_opportunity(data_median_metrics,selected_orgs):
    median_metrics_final = pd.DataFrame(columns=[ 'opportunity','visit_day', 'median_averge_visits', 'median_dus_per_day'])
    median_metrics_final = (
    data_median_metrics
    .groupby(['opportunity', 'visit_day'])[['avrg_forms_per_day_mavrg', 'dus_per_day_mavrg']]
    .median()
    .reset_index()
    .rename(columns={
        'avrg_forms_per_day_mavrg': 'median_averge_visits',
        'dus_per_day_mavrg': 'median_dus_per_day'
    })
)
    return median_metrics_final      


def get_overall_opp_median_metrics(data_median_metrics):
    median_metrics_final = pd.DataFrame(columns=[ 'visit_day', 'median_averge_visits', 'median_dus_per_day'])
    median_metrics_final = (
    data_median_metrics
    .groupby([ 'visit_day'])[['avrg_forms_per_day_mavrg', 'dus_per_day_mavrg']]
    .median()
    .reset_index()
    .rename(columns={
        'avrg_forms_per_day_mavrg': 'median_averge_visits',
        'dus_per_day_mavrg': 'median_dus_per_day'
    })
)
    return median_metrics_final      

def _generate_lightweight_pathway_data(coverage_data_objects):
    """
    Generate lightweight pathway statistics without creating HTML maps.
    This is much faster and uses less memory.
    """
    # Combine all service data
    all_service_data = []
    for org, coverage_data in coverage_data_objects.items():
        service_df = coverage_data.create_service_points_dataframe()
        if service_df is not None and not service_df.empty:
            service_df['opportunity'] = org
            all_service_data.append(service_df)
    
    if not all_service_data:
        return {'summary': {}, 'flw_statistics': []}
    
    combined_service_df = pd.concat(all_service_data, ignore_index=True)
    
    # Clean and prepare data
    df = combined_service_df.copy()
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df = df.dropna(subset=['lattitude', 'longitude', 'visit_date'])
    df = df.sort_values(['flw_id', 'visit_date'])
    df['date'] = df['visit_date'].dt.date
    
    # Calculate pathway segments
    segments = []
    flw_stats = {}
    
    for flw_id in df['flw_id'].unique():
        flw_data = df[df['flw_id'] == flw_id].copy()
        flw_segments = []
        
        # Group by date
        for date in flw_data['date'].unique():
            daily_data = flw_data[flw_data['date'] == date].copy()
            
            if len(daily_data) < 2:
                continue
            
            daily_data = daily_data.sort_values('visit_date')
            
            # Create segments between consecutive visits
            for i in range(len(daily_data) - 1):
                current = daily_data.iloc[i]
                next_visit = daily_data.iloc[i + 1]
                
                try:
                    distance = geodesic(
                        (current['lattitude'], current['longitude']),
                        (next_visit['lattitude'], next_visit['longitude'])
                    ).kilometers
                except:
                    distance = 0
                
                time_diff = (next_visit['visit_date'] - current['visit_date']).total_seconds() / 3600
                is_unusual = distance > 5 or time_diff > 2
                
                segment = {
                    'flw_id': flw_id,
                    'distance': distance,
                    'is_unusual': is_unusual
                }
                
                flw_segments.append(segment)
        
        if flw_segments:
            flw_distance = sum(seg['distance'] for seg in flw_segments)
            flw_unusual = sum(1 for seg in flw_segments if seg['is_unusual'])
            flw_avg_distance = flw_distance / len(flw_segments) if flw_segments else 0
            
            flw_stats[flw_id] = {
                'flw_id': flw_id,
                'flw_name': flw_data['flw_name'].iloc[0] if not flw_data.empty else '',
                'total_segments': len(flw_segments),
                'total_distance_km': round(flw_distance, 2),
                'unusual_segments': flw_unusual,
                'avg_segment_distance_km': round(flw_avg_distance, 2)
            }
            
            segments.extend(flw_segments)
    
    # Calculate summary statistics
    if segments:
        total_distance = sum(seg['distance'] for seg in segments)
        unusual_segments = sum(1 for seg in segments if seg['is_unusual'])
        avg_distance = total_distance / len(segments) if segments else 0
        
        summary = {
            'total_flws': len(flw_stats),
            'total_segments': len(segments),
            'total_distance_km': round(total_distance, 2),
            'unusual_segments': unusual_segments,
            'unusual_percentage': round((unusual_segments / len(segments)) * 100, 1),
            'avg_segment_distance_km': round(avg_distance, 2)
        }
    else:
        summary = {}
    
    return {
        'summary': summary,
        'flw_statistics': list(flw_stats.values())
    }

def _generate_lightweight_pathway_data_for_flws(coverage_data_objects, target_flw_ids):
    """
    Generate lightweight pathway statistics for specific FLW IDs only.
    """
    # Combine all service data
    all_service_data = []
    for org, coverage_data in coverage_data_objects.items():
        service_df = coverage_data.create_service_points_dataframe()
        if service_df is not None and not service_df.empty:
            service_df['opportunity'] = org
            all_service_data.append(service_df)
    
    if not all_service_data:
        return {'summary': {}, 'flw_statistics': []}
    
    combined_service_df = pd.concat(all_service_data, ignore_index=True)
    
    # Filter for specific FLW IDs
    filtered_df = combined_service_df[combined_service_df['flw_id'].isin(target_flw_ids)].copy()
    
    if filtered_df.empty:
        return {'summary': {}, 'flw_statistics': []}
    
    # Clean and prepare data
    df = filtered_df.copy()
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df = df.dropna(subset=['lattitude', 'longitude', 'visit_date'])
    df = df.sort_values(['flw_id', 'visit_date'])
    df['date'] = df['visit_date'].dt.date
    
    # Calculate pathway segments
    segments = []
    flw_stats = {}
    
    for flw_id in df['flw_id'].unique():
        flw_data = df[df['flw_id'] == flw_id].copy()
        flw_segments = []
        
        # Group by date
        for date in flw_data['date'].unique():
            daily_data = flw_data[flw_data['date'] == date].copy()
            
            if len(daily_data) < 2:
                continue
            
            daily_data = daily_data.sort_values('visit_date')
            
            # Create segments between consecutive visits
            for i in range(len(daily_data) - 1):
                current = daily_data.iloc[i]
                next_visit = daily_data.iloc[i + 1]
                
                try:
                    distance = geodesic(
                        (current['lattitude'], current['longitude']),
                        (next_visit['lattitude'], next_visit['longitude'])
                    ).kilometers
                except:
                    distance = 0
                
                time_diff = (next_visit['visit_date'] - current['visit_date']).total_seconds() / 3600
                is_unusual = distance > 5 or time_diff > 2
                
                segment = {
                    'flw_id': flw_id,
                    'distance': distance,
                    'is_unusual': is_unusual
                }
                
                flw_segments.append(segment)
        
        if flw_segments:
            flw_distance = sum(seg['distance'] for seg in flw_segments)
            flw_unusual = sum(1 for seg in flw_segments if seg['is_unusual'])
            flw_avg_distance = flw_distance / len(flw_segments) if flw_segments else 0
            
            flw_stats[flw_id] = {
                'flw_id': flw_id,
                'flw_name': flw_data['flw_name'].iloc[0] if not flw_data.empty else '',
                'total_segments': len(flw_segments),
                'total_distance_km': round(flw_distance, 2),
                'unusual_segments': flw_unusual,
                'avg_segment_distance_km': round(flw_avg_distance, 2)
            }
            
            segments.extend(flw_segments)
    
    # Calculate summary statistics
    if segments:
        total_distance = sum(seg['distance'] for seg in segments)
        unusual_segments = sum(1 for seg in segments if seg['is_unusual'])
        avg_distance = total_distance / len(segments) if segments else 0
        
        summary = {
            'total_flws': len(flw_stats),
            'total_segments': len(segments),
            'total_distance_km': round(total_distance, 2),
            'unusual_segments': unusual_segments,
            'unusual_percentage': round((unusual_segments / len(segments)) * 100, 1),
            'avg_segment_distance_km': round(avg_distance, 2)
        }
    else:
        summary = {}
    
    return {
        'summary': summary,
        'flw_statistics': list(flw_stats.values())
    }

def _generate_pathway_map(coverage_data_objects, target_flw_ids):
    """
    Generate an interactive map showing pathway segments for specific FLW IDs.
    """
    try:
        import folium
        from folium import Icon, Marker
        
        # Combine all service data
        all_service_data = []
        for org, coverage_data in coverage_data_objects.items():
            service_df = coverage_data.create_service_points_dataframe()
            if service_df is not None and not service_df.empty:
                service_df['opportunity'] = org
                all_service_data.append(service_df)
        
        if not all_service_data:
            return html.P("No data available for map generation")
        
        combined_service_df = pd.concat(all_service_data, ignore_index=True)
        
        # Filter for specific FLW IDs
        filtered_df = combined_service_df[combined_service_df['flw_id'].isin(target_flw_ids)].copy()
        
        if filtered_df.empty:
            return html.P("No data found for the specified FLW IDs")
        
        # Clean and prepare data
        df = filtered_df.copy()
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        df = df.dropna(subset=['lattitude', 'longitude', 'visit_date'])
        df = df.sort_values(['flw_id', 'visit_date'])
        df['date'] = df['visit_date'].dt.date
        
        # Create pathway segments for mapping
        segments = []
        for flw_id in df['flw_id'].unique():
            flw_data = df[df['flw_id'] == flw_id].copy()
            
            # Group by date
            for date in flw_data['date'].unique():
                daily_data = flw_data[flw_data['date'] == date].copy()
                
                if len(daily_data) < 2:
                    continue
                
                daily_data = daily_data.sort_values('visit_date')
                
                # Create segments between consecutive visits
                for i in range(len(daily_data) - 1):
                    current = daily_data.iloc[i]
                    next_visit = daily_data.iloc[i + 1]
                    
                    try:
                        distance = geodesic(
                            (current['lattitude'], current['longitude']),
                            (next_visit['lattitude'], next_visit['longitude'])
                        ).kilometers
                    except:
                        distance = 0
                    
                    time_diff = (next_visit['visit_date'] - current['visit_date']).total_seconds() / 3600
                    is_unusual = distance > 5 or time_diff > 2
                    
                    segment = {
                        'flw_id': flw_id,
                        'flw_name': current['flw_name'],
                        'date': date,
                        'latitude': current['lattitude'],
                        'longitude': current['longitude'],
                        'lat_next': next_visit['lattitude'],
                        'lon_next': next_visit['longitude'],
                        'distance': distance,
                        'time_diff_hours': time_diff,
                        'is_unusual': is_unusual,
                        'du_name': current['du_name'],
                        'next_du_name': next_visit['du_name']
                    }
                    
                    segments.append(segment)
        


        if not segments:
            return html.P("No pathway segments found for mapping")
        
        # Center the map around the average coordinates
        avg_lat = sum(seg['latitude'] for seg in segments) / len(segments)
        avg_lon = sum(seg['longitude'] for seg in segments) / len(segments)
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
        
        # Add segments to map
        for segment in segments:
            if pd.notnull(segment['lat_next']) and pd.notnull(segment['lon_next']):
                coords = [(segment['latitude'], segment['longitude']), 
                         (segment['lat_next'], segment['lon_next'])]
                
                color = 'red' if segment['is_unusual'] else 'blue'
                
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=3,
                    tooltip=f"{segment['flw_name']} | {segment['date']} | {segment['distance']:.2f} km"
                ).add_to(m)
        
        # Add start and end markers for each day
        daily_groups = {}
        for segment in segments:
            date = segment['date']
            if date not in daily_groups:
                daily_groups[date] = []
            daily_groups[date].append(segment)
        
        for date, day_segments in daily_groups.items():
            if day_segments:
                # Start marker
                start_seg = day_segments[0]
                Marker(
                    location=(start_seg['latitude'], start_seg['longitude']),
                    popup=f"START: {start_seg['flw_name']} on {date}",
                    icon=Icon(color='green', icon='play')
                ).add_to(m)
                
                # End marker
                end_seg = day_segments[-1]
                Marker(
                    location=(end_seg['lat_next'], end_seg['lon_next']),
                    popup=f"END: {end_seg['flw_name']} on {date}",
                    icon=Icon(color='red', icon='stop')
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    top: 50px; right: 50px; width: 200px;  
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Pathway Legend</b></p>
        <p><i class="fa fa-circle" style="color:blue"></i> Normal Segment</p>
        <p><i class="fa fa-circle" style="color:red"></i> Unusual Segment</p>
        <p><i class="fa fa-play" style="color:green"></i> Start Point</p>
        <p><i class="fa fa-stop" style="color:red"></i> End Point</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Convert map to HTML
        map_html = m._repr_html_()
        
        return html.Iframe(
            srcDoc=map_html,
            style={'width': '100%', 'border': 'none', 'min-height': '800px', 'overflow': 'hidden'},
            title="FLW Pathway Map"
        )
        
    except Exception as e:
        return html.P(f"Error generating map: {str(e)}", style={'color': 'red'})

def _generate_recent_unusual_segments(coverage_data_objects):
    """
    Generate a list of unusual segments from the past 7 days.
    """
    try:
        from datetime import datetime, timedelta
        
        # Combine all service data
        all_service_data = []
        for org, coverage_data in coverage_data_objects.items():
            service_df = coverage_data.create_service_points_dataframe()
            if service_df is not None and not service_df.empty:
                service_df['opportunity'] = org
                all_service_data.append(service_df)
        
        if not all_service_data:
            return []
        
        combined_service_df = pd.concat(all_service_data, ignore_index=True)
        
        # Clean and prepare data
        df = combined_service_df.copy()
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        df = df.dropna(subset=['lattitude', 'longitude', 'visit_date'])
        df = df.sort_values(['flw_id', 'visit_date'])
        df['date'] = df['visit_date'].dt.date
        
        # Filter for past 7 days
        seven_days_ago = datetime.now().date() - timedelta(days=7)
        df = df[df['date'] >= seven_days_ago].copy()
        
        if df.empty:
            return []
        
        # Create pathway segments
        unusual_segments = []
        
        for flw_id in df['flw_id'].unique():
            flw_data = df[df['flw_id'] == flw_id].copy()
            
            # Group by date
            for date in flw_data['date'].unique():
                daily_data = flw_data[flw_data['date'] == date].copy()
                
                if len(daily_data) < 2:
                    continue
                
                daily_data = daily_data.sort_values('visit_date')
                
                # Create segments between consecutive visits
                for i in range(len(daily_data) - 1):
                    current = daily_data.iloc[i]
                    next_visit = daily_data.iloc[i + 1]
                    
                    try:
                        distance = geodesic(
                            (current['lattitude'], current['longitude']),
                            (next_visit['lattitude'], next_visit['longitude'])
                        ).kilometers
                    except:
                        distance = 0
                    
                    time_diff = (next_visit['visit_date'] - current['visit_date']).total_seconds() / 3600
                    is_unusual = distance > 5 or time_diff > 2
                    
                    if is_unusual:
                        segment = {
                            'flw_id': flw_id,
                            'flw_name': current['flw_name'],
                            'date': date,
                            'distance_km': round(distance, 2),
                            'time_diff_hours': round(time_diff, 1),
                            'du_name': current['du_name'],
                            'next_du_name': next_visit['du_name'],
                            'opportunity': current.get('opportunity', 'Unknown'),
                            'reason': 'Distance' if distance > 5 else 'Time' if time_diff > 2 else 'Both'
                        }
                        unusual_segments.append(segment)
        
        # Sort by date (most recent first)
        unusual_segments.sort(key=lambda x: x['date'], reverse=True)
        
        return unusual_segments
        
    except Exception as e:
        logging.error(f"Error generating recent unusual segments: {str(e)}")
        return []

def create_flw_dashboard(coverage_data_objects):
    app = dash.Dash(__name__)
    summary_df, _ = generate_summary(coverage_data_objects, group_by='flw')
    summary_df['days_since_active'] = pd.to_numeric(summary_df['days_since_active'], errors='coerce')
    summary_df['avrg_forms_per_day_mavrg'] = pd.to_numeric(summary_df['avrg_forms_per_day_mavrg'], errors='coerce')
    summary_df['dus_per_day_mavrg'] = pd.to_numeric(summary_df['dus_per_day_mavrg'], errors='coerce')
    
    # Initialize empty pathway data - will be populated on demand
    pathway_data = {'summary': {}, 'flw_statistics': []}
    
    app.layout = html.Div([
        html.Div([
            html.H1("FLW Summary Dashboard", style={
                'color': '#333',
                'borderBottom': '1px solid #ddd',
                'paddingBottom': '10px',
                'marginBottom': '20px'
            }),
            dcc.Dropdown(
                id='org-selector',
                options=[{'label': k, 'value': k} for k in coverage_data_objects],
                multi=True,
                placeholder="Filter by opportunity",
                style={
                    'marginBottom': '20px',
                    'fontFamily': 'Arial, sans-serif'
                }
            ),
           html.Button("Export as CSV", id="export-csv-btn", n_clicks=0, style={"marginBottom": "10px"}), 
            AgGrid(
                id='flw-summary-table',
                columnDefs=[
    {
        "headerName": summary_df.columns[0],
        "field": summary_df.columns[0],
        "headerClass": "wrap-header",
        "pinned": "left",
        "width": 150,
        "minWidth": 140,
        "maxWidth": 160,
        "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}
    },
    {
        "headerName": summary_df.columns[1],
        "field": summary_df.columns[1],
        "headerClass": "wrap-header",
        "pinned": "left",
        "width": 150,
        "minWidth": 140,
        "maxWidth": 160,
        "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}
    },
    {
        "headerName": summary_df.columns[2],
        "field": summary_df.columns[2],
        "headerClass": "wrap-header",
        "width": 240,
        "minWidth": 220,
        "maxWidth": 250,
        "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}
    },
    # Conditional formatting for days_since_active
    {
        "headerName": "days_since_active",
    "field": "days_since_active",
    "width": 140,
    "minWidth": 130,
    "maxWidth": 150,
    "cellClassRules": {
        "cell-abnormal": "params.value > 7"
        }
    },
    # Conditional formatting for avrg_forms_per_day_mavrg
    {
        "headerName": "avrg_forms_per_day_mavrg",
        "field": "avrg_forms_per_day_mavrg",
        "width": 140,
        "minWidth": 130,
        "maxWidth": 150,
       "cellClassRules": {
        "cell-abnormal": "params.value < 10"
        }
    },
    # Conditional formatting for dus_per_day_mavrg
    {
        "headerName": "dus_per_day_mavrg",
        "field": "dus_per_day_mavrg",
        "width": 140,
        "minWidth": 130,
        "maxWidth": 150,
        "cellClassRules": {
        "cell-abnormal": "params.value < 1"
        }
    },
    # ...add other columns as needed...
] + [
    {"headerName": i, "field": i, "headerClass": "wrap-header", "width": 140, "minWidth": 130, "maxWidth": 150, "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}}
    for i in summary_df.columns[3:] if i not in ["days_since_active", "avrg_forms_per_day_mavrg", "dus_per_day_mavrg"]
],
                rowData=summary_df.to_dict("records"),
            defaultColDef={
                
                "sortable": True,
                "filter": True,
                "wrapHeaderText": False,  # Enable header wrapping
                "autoHeaderHeight": True, # Adjust header height automatically,
                "flex":1, # This allows columns to grow/shrink to fill the grid width
                
            },
            dashGridOptions = {
                "domLayout": "normal",  # or "normal" for fixed height
                "maxRowsToShow": 5,  # Adjust this to your desired maximum
                "pagination": True,  # Enable pagination
                "paginationPageSize" : 10, # Number of rows per page
                "paginationPageSizeSelector": [10, 20, 50, 100],  # User can pick page size
                "rowSelection": "multiple",
                "suppressHorizontalScroll": False,  # Explicitly allow horizontal scrolling 
                "enableBrowserTooltips": True,        # <-- Optional: tooltips for overflow
                "enableExport": True,
                "menuTabs": ["generalMenuTab", "columnsMenuTab", "filterMenuTab"],  # Show all menu tabs
                # "getMainMenuItems": {
                #     "function": "defaultItems => [...defaultItems, 'export']" 
                # }
            },
            #enableEnterpriseModules=True, 
            csvExportParams={                         # <-- Optional: customize CSV export
                "fileName": "flw_summary_export.csv",
                "allColumns": True
            },
            style={'height': '400px', 'width': '100%', 'overflowX' : 'auto'},
            className="ag-theme-alpine",
            
            ),
            html.Div([
                html.H2("Rolling 7-Day Averages", style={
                    'color': '#333',
                    'borderBottom': '1px solid #ddd',
                    'paddingBottom': '10px',
                    'marginTop': '40px',
                    'marginBottom': '20px'
                }),
                html.Div([
                    html.Div([
                        dcc.Graph(id='forms-rolling-chart'),
                    ], className='chart-item'),
                    html.Div([
                        dcc.Graph(id='dus-rolling-chart'),
                    ], className='chart-item')
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '30px',
                    'margin': '20px 0'
                }),
                html.Div([
                    html.Div([
                        dcc.Graph(id='forms-median-chart'),
                    ], className='chart-item'),
                    html.Div([
                        dcc.Graph(id='dus-median-chart'),
                    ],className='chart-item')
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '30px',
                    'margin': '20px 0'
                })
            ], style={
                'backgroundColor': '#fafafa',
                'padding': '20px',
                'borderRadius': '5px',
                'marginTop': '30px'
            }),
            
            # FLW Pathway Analysis Section
            html.Div([
                html.H2("FLW Pathway Analysis", style={
                    'color': '#333',
                    'borderBottom': '1px solid #ddd',
                    'paddingBottom': '10px',
                    'marginTop': '40px',
                    'marginBottom': '20px'
                }),
                
                # Recent Unusual Segments Table
                html.Div([
                    html.H3("Recent Unusual Segments (Past 7 Days)", style={
                        'color': '#333',
                        'marginBottom': '15px'
                    }),
                    html.Div(id='recent-unusual-segments-container', style={
                        'backgroundColor': 'white',
                        'padding': '15px',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    })
                ], style={
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'marginBottom': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # FLW ID Input Section
                html.Div([
                    html.H3("Analyze Specific FLW Pathways", style={
                        'color': '#333',
                        'marginBottom': '15px'
                    }),
                    html.P("Enter specific FLW IDs to analyze their pathways (comma-separated):", style={
                        'color': '#666',
                        'marginBottom': '10px'
                    }),
                    dcc.Input(
                        id='flw-ids-input',
                        type='text',
                        placeholder='e.g., 1298, 1208, 1500',
                        style={
                            'width': '100%',
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'borderRadius': '5px',
                            'marginBottom': '10px'
                        }
                    ),
                    html.Button(
                        'Analyze Pathways',
                        id='analyze-pathways-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '10px 20px',
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'marginBottom': '20px'
                        }
                    ),
                    html.Div(id='pathway-analysis-status', style={
                        'marginBottom': '20px',
                        'padding': '10px',
                        'borderRadius': '5px'
                    })
                ], style={
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'marginBottom': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Pathway Summary Cards
                html.Div([
                    html.Div([
                        html.H3("Total FLWs Analyzed", style={'margin': '0', 'color': '#666'}),
                        html.H2(id='total-flws-analyzed', style={'margin': '10px 0', 'color': '#4CAF50'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }),
                    html.Div([
                        html.H3("Total Distance", style={'margin': '0', 'color': '#666'}),
                        html.H2(id='total-distance', style={'margin': '10px 0', 'color': '#2196F3'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }),
                    html.Div([
                        html.H3("Unusual Segments", style={'margin': '0', 'color': '#666'}),
                        html.H2(id='unusual-segments', style={'margin': '10px 0', 'color': '#FF9800'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }),
                    html.Div([
                        html.H3("Avg Segment Distance", style={'margin': '0', 'color': '#666'}),
                        html.H2(id='avg-segment-distance', style={'margin': '10px 0', 'color': '#9C27B0'})
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(4, 1fr)',
                    'gap': '20px',
                    'marginBottom': '30px'
                }),
                
                # FLW Pathway Table
                html.Div([
                    html.H3("Individual FLW Pathway Statistics", style={
                        'color': '#333',
                        'marginBottom': '15px'
                    }),
                    html.Div(id='pathway-table-container', style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ]),
                
                # Pathway Map Section
                html.Div([
                    html.H3("FLW Pathway Map", style={
                        'color': '#333',
                        'marginBottom': '15px'
                    }),
                    html.Div(id='pathway-map-container', style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'minHeight': '500px'
                    })
                ])
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '5px',
                'marginTop': '30px'
            })
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 0 10px rgba(0,0,0,0.1)'
        })
    ], style={
        'fontFamily': 'Arial, sans-serif',
        'margin': '0',
        'padding': '20px',
        'backgroundColor': '#f5f5f5'
    })
    
    # Add custom CSS for pathway section
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>FLW Summary Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                .pathway-card {
                    transition: transform 0.2s ease-in-out;
                }
                .pathway-card:hover {
                    transform: translateY(-2px);
                }
                .pathway-table {
                    margin-top: 20px;
                }
                .pathway-table .dash-table-container {
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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

    @app.callback(
        Output('flw-summary-table', 'rowData'),
        Output('forms-rolling-chart', 'figure'),
        Output('dus-rolling-chart', 'figure'),
        Output('forms-median-chart', 'figure'),
        Output('dus-median-chart', 'figure'),
        Input('org-selector', 'value')
    )

    def update_dashboard(selected_orgs):
        sorted_by_flw = []
        if not selected_orgs:
            # No selection - show opportunity level summary
            summary_df, _ = generate_summary(coverage_data_objects, group_by='opportunity')

            # Create opportunity-level time series
            all_service_dfs = []
            for org, cov in coverage_data_objects.items():
                df = cov.create_service_points_dataframe()
                if df is not None and not df.empty:
                    df = df[df['visit_date'].notna()]
                    df['visit_day'] = pd.to_datetime(df['visit_date'], format='ISO8601', utc=True).dt.date
                    df['opportunity'] = org
                    all_service_dfs.append(df[['visit_day', 'opportunity', 'visit_id', 'du_name']])

            service_timeline_df = pd.concat(all_service_dfs)

            # Create historical metrics DataFrame
            historical_metrics = []
            median_metrics = []
            # Get unique dates
            unique_dates = sorted(service_timeline_df['visit_day'].unique())
            
        
            for current_date in unique_dates:
                window_metrics_temp = pd.DataFrame(columns=['opportunity', 'visit_day', 'avrg_forms_per_day_mavrg', 'dus_per_day_mavrg'])
                # Calculate the 7-day window
                window_start = current_date - timedelta(days=6)

                # Filter data for this window
                window_data = service_timeline_df[
                    (service_timeline_df['visit_day'] >= window_start) &
                    (service_timeline_df['visit_day'] <= current_date)
                    ]

                # Calculate metrics for this window
                window_metrics = window_data.groupby(['opportunity', 'visit_day']).agg(
                    visits_last7=('visit_id', 'count'),
                    dus_last7=('du_name', pd.Series.nunique)
                ).reset_index()

                recent_median_df = window_data.groupby(['opportunity', 'visit_day']).agg(
                    visit_count=('visit_id', 'count')
                ).reset_index()
                #looping through all unique opportunities to calculate all average parameters of each opportunity for current_data
                unique_opportunities = sorted(recent_median_df['opportunity'].unique())
                init_row = 0
                for opp in unique_opportunities:
                    #visit_median = round(recent_median_df[recent_median_df['opportunity'] == opp]['visit_count'].median(),2)
                    window_visit_average = round(window_metrics[window_metrics['opportunity'] == opp]['visits_last7'].sum()/7, 2)
                    window_du_visit_average = round(window_metrics[window_metrics['opportunity'] == opp]['dus_last7'].sum()/7, 2)
                    window_metrics_temp.loc[init_row] = {'opportunity': opp, 'visit_day': current_date,
                                                         'avrg_forms_per_day_mavrg': window_visit_average,
                                                      'dus_per_day_mavrg': window_du_visit_average}

                    init_row = init_row + 1

                median_metrics.append(window_metrics_temp)
                historical_metrics.append(window_metrics_temp)

            

            # Combine all historical metrics
            chart_data = pd.concat(historical_metrics, ignore_index=True)
            median_data = pd.concat(median_metrics, ignore_index=True)
            data_median_metrics = get_overall_opp_median_metrics(median_data)

        else:
            # Selection made - show FLW level summary
            summary_df, _ = generate_summary(coverage_data_objects, group_by='flw')
            if 'opportunity' not in summary_df.columns and 'opportunity_name' in summary_df.columns:
                summary_df['opportunity'] = summary_df['opportunity_name']
            if 'opportunity_name' not in summary_df.columns and 'opportunity' in summary_df.columns:
                summary_df['opportunity_name'] = summary_df['opportunity']
            summary_df = summary_df[summary_df['opportunity'].isin(selected_orgs)]

            # Create FLW-level time series
            all_service_dfs = []
            for org, cov in coverage_data_objects.items():
                if org in selected_orgs:  # Only process selected opportunities
                    df = cov.create_service_points_dataframe()
                    if df is not None and not df.empty:
                        df = df[df['visit_date'].notna()]
                        df['visit_day'] = pd.to_datetime(df['visit_date'], format='ISO8601', utc=True).dt.date
                        df['opportunity'] = org
                        all_service_dfs.append(
                            df[['flw_name', 'visit_day', 'flw_id', 'opportunity', 'visit_id', 'du_name']])

            service_timeline_df = pd.concat(all_service_dfs)

            # Create historical metrics DataFrame
            historical_metrics = []

            # Get unique dates
            unique_dates = sorted(service_timeline_df['visit_day'].unique())

            for current_date in unique_dates:
                window_metrics_temp = pd.DataFrame(
                    columns=['flw_id', 'visit_day', 'avrg_forms_per_day_mavrg', 'dus_per_day_mavrg'])

                # Calculate the 7-day window
                window_start = current_date - timedelta(days=6)

                # Filter data for this window
                window_data = service_timeline_df[
                    (service_timeline_df['visit_day'] >= window_start) &
                    (service_timeline_df['visit_day'] <= current_date)
                    ]
            
                # Calculate metrics for this window
                window_metrics = window_data.groupby(['flw_id', 'opportunity']).agg(
                    visits_last7=('visit_id', 'count'),
                    dus_last7=('du_name', pd.Series.nunique)
                ).reset_index()

                #window_metrics contains flw_id and oppurtunity mapping
                

                # looping through all unique opportunities to calculate all average parameters of each opportunity for current_data
                unique_flw_ids = sorted(window_metrics['flw_id'].unique())
                init_row = 0
                for flw_id in unique_flw_ids:
                    window_visit_average = round(window_metrics[window_metrics['flw_id'] == flw_id]['visits_last7'].sum()/7, 2)
                    window_du_visit_average = round(window_metrics[window_metrics['flw_id'] == flw_id]['dus_last7'].sum()/7, 2)
                    window_metrics_temp.loc[init_row] = {'flw_id': flw_id, 'visit_day': current_date,
                                                         'avrg_forms_per_day_mavrg': window_visit_average,
                                                         'dus_per_day_mavrg': window_du_visit_average}

                    
                    init_row = init_row + 1

                # adding back flw_name
                window_metrics_temp = window_metrics_temp.merge(
                    window_data[['flw_id', 'flw_name']].drop_duplicates(),
                    on='flw_id',
                    how='left'
                )
                historical_metrics.append(window_metrics_temp)
                
                # #adding back flw_name
                # window_metrics = window_metrics.merge(
                #     window_data[['flw_id', 'flw_name']].drop_duplicates(),
                #     on='flw_id',
                #     how='left'
                # )

                # # Calculate 7-day averages
                # window_metrics['avrg_forms_per_day_mavrg'] = window_metrics['visits_last7'] / 7
                # window_metrics['dus_per_day_mavrg'] = window_metrics['dus_last7'] / 7

                # Add date
                # window_metrics['visit_day'] = current_date

                # historical_metrics.append(window_metrics)

            # Combine all historical metrics
            

            chart_data = pd.concat(historical_metrics, ignore_index=True)
            
            #Including back the opputunity
            chart_data = chart_data.merge(
                window_data[['flw_id', 'opportunity']].drop_duplicates(),
                on='flw_id',how='left')
            chart_data.fillna({'avrg_forms_per_day_mavrg': 0, 'dus_per_day_mavrg': 0, 'visit_count_median': 0}, inplace=True)
            chart_data["flw"] = "(" + chart_data["flw_id"] + ")" +chart_data["flw_name"]
            sorted_by_flw = sorted(chart_data['flw'].unique())
            data_median_metrics = get_overall_opp_median_metrics_per_opportunity(chart_data,selected_orgs)

            
            

        # Create the charts
        if not selected_orgs:
            # Opportunity level charts
            forms_fig = px.line(
                chart_data,
                x='visit_day',
                y='avrg_forms_per_day_mavrg',
                color='opportunity',
                title='7-Day Rolling Average of Forms Submitted',
                labels={"visit_day": "Visit Day", "avrg_forms_per_day_mavrg": "Average Form Submissions"}
            )

            dus_fig = px.line(
                chart_data,
                x='visit_day',
                y='dus_per_day_mavrg',
                color='opportunity',
                title='7-Day Rolling Average of DUs Visited',
                labels={"visit_day": "Visit Day", "dus_per_day_mavrg": "Average DUs Visited "}
            )
            forms_fig2 = px.line(
                data_median_metrics,
                x='visit_day',
                y='median_averge_visits',
                title='7-Day Median of Forms Submitted',
                labels={"visit_day": "Visit Day", "median_averge_visits": "Median Form Submissions"}
            )

            dus_fig2 = px.line(
                data_median_metrics,
                x='visit_day',
                y='median_dus_per_day',
                title='7-Day Median Average of DUs Visited',
                labels={"visit_day": "Visit Day", "median_dus_per_day": " Median DUs Visited "}
            )

            
        else:
            # FLW level charts
            forms_fig = px.line(
                chart_data,
                x='visit_day',
                y='avrg_forms_per_day_mavrg',
                color='flw',
                title='7-Day Rolling Average of Forms Submitted',
                labels={"visit_day": "Visit Day", "avrg_forms_per_day_mavrg": "Average Form Submissions"},
                category_orders = {'flw': sorted_by_flw}

            )

            dus_fig = px.line(
                chart_data,
                x='visit_day',
                y='dus_per_day_mavrg',
                color='flw',
                title='7-Day Rolling Average of DUs Visited',
                labels={"visit_day": "Visit Day", "dus_per_day_mavrg": "Average DUs Visited "},
                category_orders={'flw': sorted_by_flw}
            )
            forms_fig2 = px.line(
                data_median_metrics,
                x='visit_day',
                y='median_averge_visits',
                color='opportunity',
                title='7-Day Median of Forms Submitted',
                labels={"visit_day": "Visit Day", "median_averge_visits": "Median Form Submissions"}
            )

            dus_fig2 = px.line(
                data_median_metrics,
                x='visit_day',
                y='median_dus_per_day',
                color='opportunity',
                title='7-Day Median Average of DUs Visited',
                labels={"visit_day": "Visit Day", "median_dus_per_day": "Median DUs Visited "}
            )
            


        
        return summary_df.to_dict('records'), forms_fig, dus_fig, forms_fig2, dus_fig2 
    @app.callback(
        Output('flw-summary-table', 'exportDataAsCsv'),
        Input('export-csv-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def export_csv(n_clicks):
        if n_clicks:
            return True  # Triggers export with default params
        return no_update
    
    @app.callback(
        Output('total-flws-analyzed', 'children'),
        Output('total-distance', 'children'),
        Output('unusual-segments', 'children'),
        Output('avg-segment-distance', 'children'),
        Output('pathway-table-container', 'children'),
        Output('pathway-analysis-status', 'children'),
        Output('pathway-map-container', 'children'),
        Input('analyze-pathways-btn', 'n_clicks'),
        State('flw-ids-input', 'value'),
        prevent_initial_call=True
    )
    def analyze_pathways(n_clicks, flw_ids_input):
        if not n_clicks or not flw_ids_input:
            return "0", "0 km", "0", "0 km", html.P("No pathway data available"), "", html.P("No map data available")
        
        try:
            # Parse FLW IDs from input
            flw_ids = [flw_id.strip() for flw_id in flw_ids_input.split(',') if flw_id.strip()]
            
            if not flw_ids:
                return "0", "0 km", "0", "0 km", html.P("No valid FLW IDs provided"), html.P("Please enter valid FLW IDs", style={'color': 'red'}), html.P("No map data available")
            
            # Generate pathway data for specific FLWs
            pathway_data = _generate_lightweight_pathway_data_for_flws(coverage_data_objects, flw_ids)
            
            summary_info = pathway_data.get('summary', {})
            flw_stats = pathway_data.get('flw_statistics', [])
            
            # Generate pathway map
            pathway_map = _generate_pathway_map(coverage_data_objects, flw_ids)
            
            # Create pathway table
            if flw_stats:
                pathway_df = pd.DataFrame(flw_stats)
                pathway_table = dash_table.DataTable(
                    id='pathway-table',
                    columns=[
                        {"name": "FLW ID", "id": "flw_id"},
                        {"name": "FLW Name", "id": "flw_name"},
                        {"name": "Total Segments", "id": "total_segments"},
                        {"name": "Total Distance (km)", "id": "total_distance_km"},
                        {"name": "Unusual Segments", "id": "unusual_segments"},
                        {"name": "Avg Segment Distance (km)", "id": "avg_segment_distance_km"}
                    ],
                    data=pathway_df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'unusual_segments', 'filter_query': '{unusual_segments} > 0'},
                            'backgroundColor': '#fff3cd',
                            'color': '#856404'
                        }
                    ]
                )
            else:
                pathway_table = html.P("No pathway data found for the specified FLW IDs")
            
            status_message = html.P(f"Successfully analyzed {len(flw_stats)} FLW(s)", style={'color': 'green'})
            
            return (
                summary_info.get('total_flws', 0),
                f"{summary_info.get('total_distance_km', 0)} km",
                summary_info.get('unusual_segments', 0),
                f"{summary_info.get('avg_segment_distance_km', 0)} km",
                pathway_table,
                status_message,
                pathway_map
            )
            
        except Exception as e:
            error_message = html.P(f"Error analyzing pathways: {str(e)}", style={'color': 'red'})
            return "0", "0 km", "0", "0 km", html.P("Error occurred during analysis"), error_message, html.P("Error occurred during map generation")
    
    @app.callback(
        Output('recent-unusual-segments-container', 'children'),
        Input('org-selector', 'value')
    )
    def update_recent_unusual_segments(selected_orgs):
        """Update the recent unusual segments table when organization selection changes."""
        try:
            print(selected_orgs)
            print(coverage_data_objects.items())
            # Generate recent unusual segments data
            if(selected_orgs == None):
                unusual_segments = _generate_recent_unusual_segments(coverage_data_objects)
            else:
                filtered_coverage_data_objects = {k: v for k, v in coverage_data_objects.items() if k in selected_orgs}
                unusual_segments = _generate_recent_unusual_segments(filtered_coverage_data_objects)
            
            if not unusual_segments:
                return html.P("No unusual segments found in the past 7 days.", style={
                    'textAlign': 'center',
                    'color': '#666',
                    'fontStyle': 'italic'
                })
            
            # Create table data
            table_data = []
            for segment in unusual_segments:
                table_data.append({
                    'FLW ID': segment['flw_id'],
                    'FLW Name': segment['flw_name'],
                    'Date': segment['date'].strftime('%Y-%m-%d'),
                    'Distance (km)': segment['distance_km'],
                    'Time Gap (hrs)': segment['time_diff_hours'],
                    'From DU': segment['du_name'],
                    'To DU': segment['next_du_name'],
                    'Opportunity': segment['opportunity'],
                    'Reason': segment['reason']
                })
            
            # Create the table
            return dash_table.DataTable(
                id='recent-unusual-segments-table',
                columns=[
                    {'name': 'FLW ID', 'id': 'FLW ID'},
                    {'name': 'FLW Name', 'id': 'FLW Name'},
                    {'name': 'Date', 'id': 'Date'},
                    {'name': 'Distance (km)', 'id': 'Distance (km)'},
                    {'name': 'Time Gap (hrs)', 'id': 'Time Gap (hrs)'},
                    {'name': 'From DU', 'id': 'From DU'},
                    {'name': 'To DU', 'id': 'To DU'},
                    {'name': 'Opportunity', 'id': 'Opportunity'},
                    {'name': 'Reason', 'id': 'Reason'}
                ],
                data=table_data,
                style_table={
                    'overflowX': 'auto',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'border': '1px solid #dee2e6'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'border': '1px solid #dee2e6',
                    'fontSize': '14px'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Distance (km)', 'filter_query': '{Distance (km)} > 5'},
                        'backgroundColor': '#fff3cd',
                        'color': '#856404'
                    },
                    {
                        'if': {'column_id': 'Time Gap (hrs)', 'filter_query': '{Time Gap (hrs)} > 2'},
                        'backgroundColor': '#f8d7da',
                        'color': '#721c24'
                    }
                ],
                page_size=10,
                sort_action='native',
                filter_action='native',
                sort_mode='multi'
            )
            
        except Exception as e:
            return html.P(f"Error loading recent unusual segments: {str(e)}", style={
                'color': 'red',
                'textAlign': 'center'
            })
    
    app.run(debug=True, port=8080)


def create_static_flw_report(coverage_data_objects, output_dir):
    """Create a static HTML report for FLW summary data."""

    # Generate the same data as the dashboard
    summaries = {}
    toplines = {}
    for key, cov in coverage_data_objects.items():
        summary_df, topline = generate_summary(cov)
        summary_df['opportunity'] = key
        summaries[key] = summary_df
        toplines[key] = topline

    combined_df = pd.concat(summaries.values(), ignore_index=True)

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FLW Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .card {{ 
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            table {{ 
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{ 
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{ background-color: #f8f9fa; }}
            .good {{ background-color: #d4edda; color: #155724; }}
            .warning {{ background-color: #fff3cd; color: #856404; }}
            .bad {{ background-color: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <h1>Field Level Worker Summary Report</h1>
    """

    # Add topline metrics
    html_content += "<div class='card'><h2>Topline Metrics</h2>"
    for org, topline in toplines.items():
        html_content += f"<h3>{org}</h3>"
        html_content += "<table>"
        for key, value in topline.items():
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html_content += "</table>"
    html_content += "</div>"

    # Add FLW summary table
    html_content += "<div class='card'><h2>FLW Summary</h2>"
    html_content += summary_df.to_html(classes='table', index=False)
    html_content += "</div>"

    # Close HTML
    html_content += """
    </body>
    </html>
    """

    # Write to file
    output_path = os.path.join(output_dir, "flw_summary.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


if __name__ == "__main__":
    from run_flw_dashboard import load_coverage_data_objects

    coverage_data_objects = load_coverage_data_objects()
    create_flw_dashboard(coverage_data_objects)
