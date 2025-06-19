import os

import dash
from dash import html, dcc, Input, Output
import pandas as pd
from dash import dash_table
from src.org_summary import generate_summary
from datetime import date, timedelta
import plotly.express as px


def create_flw_dashboard(coverage_data_objects):
    app = dash.Dash(__name__)
    #print("I came here in Create !!")
    # Initial summary for table columns
    summary_df, _ = generate_summary(coverage_data_objects, group_by='flw')

    # with pd.option_context('display.max_columns', None):
    #     print("-----summary_df in Create-----")
    #     print(summary_df)

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
            dash_table.DataTable(
                id='flw-summary-table',
                columns=[{"name": i, "id": i} for i in summary_df.columns],
                style_table={
                    'overflowX': 'auto',
                    'boxShadow': '0 0 10px rgba(0,0,0,0.1)',
                    'borderRadius': '5px',
                    'marginBottom': '30px',
                    'maxHeight': '800px',
                    'overflowY': 'scroll',
                    'margin': '0px',
                    'width': '100%',
                    'max-width': 'none'
                },
                fill_width=False,
                fixed_columns={'headers': True, 'data': 2},
                filter_action='native',
                sort_action='native',
                export_format='csv',
                export_headers='display',
                style_data_conditional=style_data_conditional,
                style_cell={
                    'textAlign': 'center',
                    'padding': '2px',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '14px',
                    'border': '1px solid #ddd',
                    'whiteSpace': 'normal',
                    'width': '140px',
                    'minWidth': '120px',
                    'maxWidth': '150px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                },
                style_header={
                    'backgroundColor': '#f2f2f2',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'border': '1px solid #ddd',
                    'padding': '6px',
                    'whiteSpace': 'normal',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_data={
                    'border': '1px solid #ddd',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                page_size=10
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
        Output('flw-summary-table', 'data'),
        Output('forms-rolling-chart', 'figure'),
        Output('dus-rolling-chart', 'figure'),
        Input('org-selector', 'value')
    )
    def update_dashboard(selected_orgs):
        #("I came herein update !!")
        if not selected_orgs:
            # No selection - show opportunity level summary
            summary_df, _ = generate_summary(coverage_data_objects, group_by='opportunity')
            with pd.option_context('display.max_columns', None):
                print("-----summary_df in Update's if not selected_orgs-----")
                print(summary_df)

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

            # Get unique dates
            unique_dates = sorted(service_timeline_df['visit_day'].unique())

            for current_date in unique_dates:
                # Calculate the 7-day window
                window_start = current_date - timedelta(days=6)

                # Filter data for this window
                window_data = service_timeline_df[
                    (service_timeline_df['visit_day'] >= window_start) &
                    (service_timeline_df['visit_day'] <= current_date)
                ]

                # Calculate metrics for this window
                window_metrics = window_data.groupby('opportunity').agg(
                    visits_last7=('visit_id', 'count'),
                    dus_last7=('du_name', pd.Series.nunique)
                ).reset_index()

                # Calculate 7-day averages
                window_metrics['avrg_forms_per_day_mavrg'] = window_metrics['visits_last7'] / 7
                window_metrics['dus_per_day_mavrg'] = window_metrics['dus_last7'] / 7

                # Add date
                window_metrics['visit_day'] = current_date

                historical_metrics.append(window_metrics)

            # Combine all historical metrics
            chart_data = pd.concat(historical_metrics, ignore_index=True)

        else:
            # Selection made - show FLW level summary
            summary_df, _ = generate_summary(coverage_data_objects, group_by='flw')
            # with pd.option_context('display.max_columns', None):
            #     print("-----summary_df in Update's if  selected_orgs-----")
            #     print(summary_df)
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
                        all_service_dfs.append(df[['flw_name', 'visit_day', 'flw_id', 'opportunity', 'visit_id', 'du_name']])

            service_timeline_df = pd.concat(all_service_dfs)

            # Create historical metrics DataFrame
            historical_metrics = []

            # Get unique dates
            unique_dates = sorted(service_timeline_df['visit_day'].unique())

            for current_date in unique_dates:
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

                #adding back flw_name
                window_metrics = window_metrics.merge(
                    window_data[['flw_id', 'flw_name']].drop_duplicates(),
                    on='flw_id',
                    how='left'
                )

                # Calculate 7-day averages
                window_metrics['avrg_forms_per_day_mavrg'] = window_metrics['visits_last7'] / 7
                window_metrics['dus_per_day_mavrg'] = window_metrics['dus_last7'] / 7

                # Add date
                window_metrics['visit_day'] = current_date

                historical_metrics.append(window_metrics)

            # Combine all historical metrics
            chart_data = pd.concat(historical_metrics, ignore_index=True)
            #saving flw_id joined with flw_name as a new column to show on graph
            chart_data["flw"] = chart_data["flw_name"] + "(" +chart_data["flw_id"] + ")"

        # Create the charts
        if not selected_orgs:
            # Opportunity level charts
            forms_fig = px.line(
                chart_data,
                x='visit_day',
                y='avrg_forms_per_day_mavrg',
                color='opportunity',
                title='7-Day Rolling Average of Forms Submitted'
            )

            dus_fig = px.line(
                chart_data,
                x='visit_day',
                y='dus_per_day_mavrg',
                color='opportunity',
                title='7-Day Rolling Average of DUs Visited'
            )
        else:
            # FLW level charts
            forms_fig = px.line(
                chart_data,
                x='visit_day',
                y='avrg_forms_per_day_mavrg',
                color='flw',
                title='7-Day Rolling Average of Forms Submitted'
            )

            dus_fig = px.line(
                chart_data,
                x='visit_day',
                y='dus_per_day_mavrg',
                color='flw',
                title='7-Day Rolling Average of DUs Visited'
            )

        return summary_df.to_dict('records'), forms_fig, dus_fig

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