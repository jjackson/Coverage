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
        )
    ])

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
            html.H3("Topline Metrics"),
            html.Ul([
                html.Li(f"Total Delivery Units: {total_dus:,}"),
                html.Li(f"Completed DUs: {total_completed_dus:,}"),
                html.Li(f"Service Points: {safe_sum('total_visits'):,}"),
                html.Li(f"Total FLWs: {safe_sum('total_flws'):,}"),
                html.Li(f"Average Coverage: {completion_pct}%")
            ])
        ])

        return df.to_dict('records'), topline

    app.run(debug=True, port=8080)

if __name__ == "__main__":
    from run_flw_dashboard import load_coverage_data_objects
    coverage_data_objects = load_coverage_data_objects()
    create_flw_dashboard(coverage_data_objects)
