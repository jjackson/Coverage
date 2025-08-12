import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import os
from dash import dash_table
from dash_ag_grid import AgGrid


# Load dataframes from Excel in Downloads
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
ward_level_excel_path = os.path.join(downloads_dir, "ward_level_status_report.xlsx")  # Change filename if needed
if not os.path.exists(ward_level_excel_path):
    raise FileNotFoundError(f"Excel file not found at {ward_level_excel_path}. Run 'python run_ward_level_status_report.py' from src folder to generate it.")
ward_level_final_df = pd.read_excel(ward_level_excel_path)

opp_level_excel_path = os.path.join(downloads_dir, "opp_level_status_report.xlsx")  # Change filename if needed
if not os.path.exists(opp_level_excel_path):
    raise FileNotFoundError(f"Excel file not found at {opp_level_excel_path}. Run 'python run_ward_level_status_report.py' from src folder to generate it.")
opp_level_final_df = pd.read_excel(opp_level_excel_path)

app = dash.Dash(__name__)

# Prepare dropdown options
domain_options = [{'label': d, 'value': d} for d in ward_level_final_df['domain'].unique()]

# Prepare columns for the opportunity-level table
column_defs = []
for i, col in enumerate(opp_level_final_df.columns):
    col_def = {"headerName": col, "field": col}
    if i < 4:
        col_def["pinned"] = "left"  # Freeze the first four columns
    column_defs.append(col_def)

opp_level_table = AgGrid(
    rowData=opp_level_final_df.to_dict('records'),
    columnDefs=column_defs,
    style={'height': '450px', 'width': '100%', 'marginBottom': '32px'},
    dashGridOptions={"pagination": True, "paginationPageSize": 10}
)

app.layout = html.Div([
    html.H2("Opportunity Level Summary"),
    opp_level_table,
    html.H2("Ward Status Dashboard"),
    html.Div([
        html.Div([
            html.Label("Select Domain:"),
            dcc.Dropdown(
                id='domain-dropdown',
                options=domain_options,
                value=domain_options[0]['value'] if domain_options else None
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
    html.Br(),
    html.Div(id='charts-container')
])


@app.callback(
    Output('ward-dropdown', 'options'),
    Output('ward-dropdown', 'value'),
    Input('domain-dropdown', 'value')
)
def update_ward_dropdown(selected_domain):
    wards = ward_level_final_df[ward_level_final_df['domain'] == selected_domain]['ward'].unique()
    options = [{'label': w, 'value': w} for w in wards]
    value = options[0]['value'] if options else None
    return options, value

@app.callback(
    Output('charts-container', 'children'),
    Input('domain-dropdown', 'value'),
    Input('ward-dropdown', 'value')
)
def update_charts(selected_domain, selected_wards):
    if not selected_domain or not selected_wards:
        return html.Div("Please select a domain and at least one ward.")

    # Ensure selected_wards is a list
    if isinstance(selected_wards, str):
        selected_wards = [selected_wards]

    filtered_rows = ward_level_final_df[
        (ward_level_final_df['domain'] == selected_domain) &
        (ward_level_final_df['ward'].isin(selected_wards))
    ]

    if filtered_rows.empty:
        return html.Div("No data for this selection.")

    # For each selected ward, show charts and table
    children = []
    for _, row in filtered_rows.iterrows():
        pct_cols = [col for col in row.index if str(col).startswith('pct_')]
        # (Optional) Round pct_ columns
        for col in pct_cols:
            if col in row.index:
                row[col] = round(float(row[col]), 2)

        table = AgGrid(
            rowData=[row.to_dict()],
            columnDefs=[{"headerName": col, "field": col} for col in row.index],
            style={'height': '100px', 'width': '100%', 'marginTop': '32px'},
            dashGridOptions={"pagination": False}
        )

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

        children.append(html.Div([
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
            table
        ]))

    return children

if __name__ == "__main__":
    app.run(debug=True)