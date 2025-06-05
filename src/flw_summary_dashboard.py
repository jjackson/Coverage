
import pandas as pd
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
from src.org_summary import generate_flw_summary

def create_flw_dashboard(coverage_data_objects):
    # Precompute all FLW summaries across opportunities
    records = []
    for opp_key, cov_data in coverage_data_objects.items():
        summary_df = generate_flw_summary(cov_data)
        if summary_df is not None and not summary_df.empty:
            summary_df['opportunity'] = opp_key
            records.append(summary_df)

    if not records:
        print("No FLW data found.")
        return

    all_flw_df = pd.concat(records, ignore_index=True)

    # Generate flag columns
    all_flw_df['flag_days_since_active'] = all_flw_df['days_since_active'] >= 6
    all_flw_df['flag_pct_days_working'] = (all_flw_df['pct_days_working'] < 50) & (all_flw_df['active_period_days'] >= 7)
    all_flw_df['flag_avrg_forms_per_day'] = all_flw_df['avrg_forms_per_day'] <= 10
    all_flw_df['flag_dus_per_day'] = all_flw_df['dus_per_day'] < 1

    # Create app
    app = dash.Dash(__name__)
    app.title = "FLW Summary Dashboard"

    app.layout = html.Div([
        html.H1("FLW Performance Summary", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='opportunity-filter',
            options=[{'label': opp, 'value': opp} for opp in sorted(all_flw_df['opportunity'].unique())],
            multi=True,
            placeholder="Filter by opportunity"
        ),
        dash_table.DataTable(
            id='flw-table',
            columns=[{"name": col, "id": col} for col in all_flw_df.columns],
            data=all_flw_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            page_size=20,
            filter_action="native",
            sort_action="native",
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{flag_days_since_active} eq True',
                        'column_id': 'flag_days_since_active'
                    },
                    'backgroundColor': '#ffcccc'
                },
                {
                    'if': {
                        'filter_query': '{flag_pct_days_working} eq True',
                        'column_id': 'flag_pct_days_working'
                    },
                    'backgroundColor': '#ffebcc'
                },
                {
                    'if': {
                        'filter_query': '{flag_avrg_forms_per_day} eq True',
                        'column_id': 'flag_avrg_forms_per_day'
                    },
                    'backgroundColor': '#ffffcc'
                },
                {
                    'if': {
                        'filter_query': '{flag_dus_per_day} eq True',
                        'column_id': 'flag_dus_per_day'
                    },
                    'backgroundColor': '#e0ccff'
                },
            ]
        )
    ])

    @app.callback(
        Output('flw-table', 'data'),
        Input('opportunity-filter', 'value')
    )
    def update_table(selected_opps):
        if selected_opps:
            filtered_df = all_flw_df[all_flw_df['opportunity'].isin(selected_opps)]
        else:
            filtered_df = all_flw_df
        return filtered_df.to_dict('records')

    app.run(debug=True, port=8080)
