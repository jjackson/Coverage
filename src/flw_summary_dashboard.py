import dash
from dash import html, dcc, Input, Output
import pandas as pd
from dash import dash_table
from src.org_summary import generate_flw_summary
import os


def create_flw_dashboard(coverage_data_objects):
    app = dash.Dash(__name__)

    summaries = {}
    toplines = {}
    for key, cov in coverage_data_objects.items():
        summary_df, topline = generate_flw_summary(cov)
        summary_df['opportunity'] = key
        summaries[key] = summary_df
        toplines[key] = topline

    combined_df = pd.concat(summaries.values(), ignore_index=True)

    # Define color coding conditions for the table
    style_data_conditional = [
        # Color code completion rates (total_dus_completed)
        {
            'if': {
                'filter_query': '{total_dus_completed} >= 80',
                'column_id': 'total_dus_completed'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },
        {
            'if': {
                'filter_query': '{total_dus_completed} >= 50 && {total_dus_completed} < 80',
                'column_id': 'total_dus_completed'
            },
            'backgroundColor': '#fff3cd',
            'color': '#856404',
        },
        {
            'if': {
                'filter_query': '{total_dus_completed} < 50',
                'column_id': 'total_dus_completed'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code visited rates (total_dus_visited)
        {
            'if': {
                'filter_query': '{total_dus_visited} >= 20',
                'column_id': 'total_dus_visited'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },
        {
            'if': {
                'filter_query': '{total_dus_visited} >= 10 && {total_dus_visited} < 20',
                'column_id': 'total_dus_visited'
            },
            'backgroundColor': '#fff3cd',
            'color': '#856404',
        },
        {
            'if': {
                'filter_query': '{total_dus_visited} < 10',
                'column_id': 'total_dus_visited'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code percentage days working
        {
            'if': {
                'filter_query': '{pct_days_working} >= 50',
                'column_id': 'pct_days_working'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },

        {
            'if': {
                'filter_query': '{pct_days_working} < 50',
                'column_id': 'pct_days_working'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code days since active (reverse logic - fewer days is better)
        {
            'if': {
                'filter_query': '{days_since_active} < 7',
                'column_id': 'days_since_active'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },

        {
            'if': {
                'filter_query': '{days_since_active} >= 7',
                'column_id': 'days_since_active'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code average forms per day
        {
            'if': {
                'filter_query': '{avrg_forms_per_day} >= 10',
                'column_id': 'avrg_forms_per_day'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },

        {
            'if': {
                'filter_query': '{avrg_forms_per_day} < 10',
                'column_id': 'avrg_forms_per_day'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code dus per day
        {
            'if': {
                'filter_query': '{dus_per_day} >= 1',
                'column_id': 'dus_per_day'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },

        {
            'if': {
                'filter_query': '{dus_per_day} < 1',
                'column_id': 'dus_per_day'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code forms per day last 7 days
        {
            'if': {
                'filter_query': '{forms_per_day_last_7d} >= 10',
                'column_id': 'forms_per_day_last_7d'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },

        {
            'if': {
                'filter_query': '{forms_per_day_last_7d} < 10',
                'column_id': 'forms_per_day_last_7d'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

        # Color code dus per day last 7 days
        {
            'if': {
                'filter_query': '{dus_per_day_last_7d} >= 1',
                'column_id': 'dus_per_day_last_7d'
            },
            'backgroundColor': '#d4edda',
            'color': '#155724',
        },

        {
            'if': {
                'filter_query': '{dus_per_day_last_7d} < 1',
                'column_id': 'dus_per_day_last_7d'
            },
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
        },

    ]

    import plotly.express as px

    # Combine all service data
    all_service_dfs = []
    for org, cov in coverage_data_objects.items():
        df = cov.create_service_points_dataframe()
        if df is not None and not df.empty:
            df = df[df['visit_date'].notna()]
            df['visit_day'] = pd.to_datetime(df['visit_date'], format='ISO8601', utc=True).dt.date
            df['opportunity'] = org
            all_service_dfs.append(df[['visit_day', 'flw_id', 'du_name', 'opportunity']])

    service_timeline_df = pd.concat(all_service_dfs)

    # Group by visit_day and opportunity
    daily_stats = service_timeline_df.groupby(['visit_day', 'opportunity']).agg(
        total_forms=('flw_id', 'count'),
        total_dus=('du_name', pd.Series.nunique)
    ).reset_index()

    # Sort for rolling window
    daily_stats = daily_stats.sort_values(by=['opportunity', 'visit_day'])

    # Rolling 7-day average
    daily_stats['forms_7d_avg'] = (
        daily_stats.groupby('opportunity')['total_forms'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    daily_stats['dus_7d_avg'] = (
        daily_stats.groupby('opportunity')['total_dus'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Plotly line chart (Forms)
    forms_fig = px.line(
        daily_stats,
        x='visit_day',
        y='forms_7d_avg',
        color='opportunity',
        title='7-Day Rolling Average of Forms Submitted'
    )

    # Plotly line chart (DUs)
    dus_fig = px.line(
        daily_stats,
        x='visit_day',
        y='dus_7d_avg',
        color='opportunity',
        title='7-Day Rolling Average of DUs Visited'
    )

    app.layout = html.Div([
        html.H1("FLW Summary Dashboard"),
        dcc.Dropdown(
            id='org-selector',
            options=[{'label': k, 'value': k} for k in coverage_data_objects],
            multi=True,
            placeholder="Filter by opportunity"
        ),
        html.Div(id='topline-metrics', style={'marginTop': 20}),


        dash_table.DataTable(
            id='flw-summary-table',
            columns=[{"name": i, "id": i} for i in combined_df.columns],
            page_size=20,
            style_table={'overflowX': 'auto'},
            filter_action='native',
            sort_action='native',
            style_data_conditional=style_data_conditional,
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '14px'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': '1px solid #dee2e6'
            },
            style_data={
                'border': '1px solid #dee2e6'
            }
        ),
        html.Div([
            html.H3("Rolling 7-Day Averages", style={'marginTop': '40px'}),
            dcc.Graph(id='forms-rolling-chart'),
            dcc.Graph(id='dus-rolling-chart')
        ])
    ])



    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>FLW Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                .card {
                    padding: 20px;
                    border-radius: 12px;
                    background-color: #ffffff;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                    flex: 1;
                    text-align: center;
                    font-family: 'Arial', sans-serif;
                }
                body {
                    background-color: #f4f6f9;
                    margin: 0;
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

Output('topline-metrics', 'children'),
        Output('flw-summary-table', 'data'),
        Output('forms-rolling-chart', 'figure'),
        Output('dus-rolling-chart', 'figure'),
        Input('org-selector', 'value')
    )
    def update_dashboard(selected_orgs):
        if selected_orgs:
            df = combined_df[combined_df['opportunity'].isin(selected_orgs)]
            filtered_toplines = [toplines[k] for k in selected_orgs if k in toplines]
        else:
            df = combined_df.copy()
            filtered_toplines = toplines.values()

        def safe_sum(key):
            return sum(t.get(key, 0) for t in filtered_toplines)

        total_dus = safe_sum('total_delivery_units')
        total_completed_dus = safe_sum('total_dus_completed')
        completion_pct = round((total_completed_dus / total_dus * 100) if total_dus else 0, 1)

        topline = html.Div([
            html.Div([
                html.Div([
                    html.H4("Total DUs", style={'marginBottom': '5px'}),
                    html.H2(f"{total_dus:,}")
                ], className="card"),
                html.Div([
                    html.H4("Completed DUs", style={'marginBottom': '5px'}),
                    html.H2(f"{total_completed_dus:,}")
                ], className="card"),
                html.Div([
                    html.H4("Service Points", style={'marginBottom': '5px'}),
                    html.H2(f"{safe_sum('total_visits'):,}")
                ], className="card"),
                html.Div([
                    html.H4("Total FLWs", style={'marginBottom': '5px'}),
                    html.H2(f"{safe_sum('total_flws'):,}")
                ], className="card"),
                html.Div([
                    html.H4("Avg. Coverage", style={'marginBottom': '5px'}),
                    html.H2(f"{completion_pct}%")
                ], className="card")
            ], style={
                'display': 'flex',
                'gap': '25px',
                'marginBottom': '30px',
                'justifyContent': 'space-between'
            })
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '40px'})

        df_records = df.to_dict('records') if not df.empty else []
        return topline, df_records, forms_fig, dus_fig

    app.run(debug=True, port=8080)


def create_static_flw_report(coverage_data_objects, output_dir):
    """Create a static HTML report for FLW summary data."""
    
    # Generate the same data as the dashboard
    summaries = {}
    toplines = {}
    for key, cov in coverage_data_objects.items():
        summary_df, topline = generate_flw_summary(cov)
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
    html_content += combined_df.to_html(classes='table', index=False)
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