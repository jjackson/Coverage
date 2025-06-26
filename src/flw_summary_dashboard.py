import logging
import os
import dash, json
from dash import html, dcc, Input, Output,no_update
import pandas as pd
from dash import dash_table
from src.org_summary import generate_summary
from datetime import timedelta
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from dash_ag_grid import AgGrid

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

def create_flw_dashboard(coverage_data_objects):
    app = dash.Dash(__name__)
    summary_df, _ = generate_summary(coverage_data_objects, group_by='flw')

    # Define color coding conditions for the table
    style_data_conditional = [
        # {
        #     'if': {
        #         'filter_query': '{days_since_active} < 7',
        #         'column_id': 'days_since_active'
        #     },
        #     'backgroundColor': '#d4edda',
        #     'color': '#155724',
        # },
        {
            'if': {
                'filter_query': '{days_since_active} >= 7',
                'column_id': 'days_since_active'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },
        # Color code average forms per day
        # {
        #     'if': {
        #         'filter_query': '{avrg_forms_per_day_mavrg} >= 10',
        #         'column_id': 'avrg_forms_per_day_mavrg'
        #     },
        #     'backgroundColor': '#d4edda',
        #     'color': '#155724',
        # },
        {
            'if': {
                'filter_query': '{avrg_forms_per_day_mavrg} < 10',
                'column_id': 'avrg_forms_per_day_mavrg'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },
        # Color code dus per day
        # {
        #     'if': {
        #         'filter_query': '{dus_per_day_mavrg} >= 1',
        #         'column_id': 'dus_per_day_mavrg'
        #     },
        #     'backgroundColor': '#d4edda',
        #     'color': '#155724',
        # },
        {
            'if': {
                'filter_query': '{dus_per_day_mavrg} < 1',
                'column_id': 'dus_per_day_mavrg'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        }
        # ,
        # # Add alternating row colors
        # {
        #     'if': {'row_index': 'odd'},
        #     'backgroundColor': '#f9f9f9'
        # }
    ]

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
                columnDefs=[{"headerName": summary_df.columns[0], "field": summary_df.columns[0], "headerClass": "wrap-header", "pinned": "left", "width": 90, "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}},
    {"headerName": summary_df.columns[1], "field": summary_df.columns[1], "headerClass": "wrap-header", "pinned": "left", "width": 90, "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}},
] + [
    {"headerName": i, "field": i, "headerClass": "wrap-header", "width": 100, "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}}
    for i in summary_df.columns[2:]],
                rowData=summary_df.to_dict("records"),
            defaultColDef={
                "resizable": True,
                "sortable": True,
                "filter": True,
                "wrapHeaderText": False,  # Enable header wrapping
                "autoHeaderHeight": True, # Adjust header height automatically,
                "flex":1, # This allows columns to grow/shrink to fill the grid width
                "cellStyle": {"whiteSpace": "pre-line", "overflowWrap": "anywhere"}  # Prevent trimming  # Wrap row data
            },
            dashGridOptions = {
                "domLayout": "normal",  # or "normal" for fixed height
                "maxRowsToShow": 5,  # Adjust this to your desired maximum
                "pagination": True,  # Enable pagination
                "paginationPageSize" : 10, # Number of rows per page
                "rowSelection": "multiple",
                "suppressHorizontalScroll": False,  # Explicitly allow horizontal scrolling 
                "enableBrowserTooltips": True,        # <-- Optional: tooltips for overflow
                "enableExport": True,
                "menuTabs": ["generalMenuTab", "columnsMenuTab", "filterMenuTab"],  # Show all menu tabs
                "getMainMenuItems": {
                    "function": "defaultItems => [...defaultItems, 'export']"
                }
            },
            enableEnterpriseModules=True, 
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
