import dash
from dash import html, dcc, Input, Output
import pandas as pd
from dash import dash_table
from src.org_summary import generate_flw_summary


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
        )
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
        Output('flw-summary-table', 'data'),
        Output('topline-metrics', 'children'),
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

        return df.to_dict('records'), topline

    app.run(debug=True, port=8080)


if __name__ == "__main__":
    from run_flw_dashboard import load_coverage_data_objects

    coverage_data_objects = load_coverage_data_objects()
    create_flw_dashboard(coverage_data_objects)