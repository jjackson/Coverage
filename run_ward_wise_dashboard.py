import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import os
from dash import dash_table


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
opp_level_table = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in opp_level_final_df.columns],
    data=opp_level_final_df.to_dict('records'),
    style_table={'overflowX': 'auto', 'marginBottom': '32px'},
    style_cell={'textAlign': 'left', 'padding': '6px'},
    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
    page_size=10  # Show 10 rows per page, adjust as needed
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
            dcc.Dropdown(id='ward-dropdown'),
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

# ...existing code...

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
def update_charts(selected_domain, selected_ward):
    if not selected_domain or not selected_ward:
        return html.Div("Please select a domain and ward.")

    row = ward_level_final_df[(ward_level_final_df['domain'] == selected_domain) & (ward_level_final_df['ward'] == selected_ward)]
    if row.empty:
        return html.Div("No data for this selection.")

    row = row.iloc[0]
    # Prepare a DataTable for the selected row
    pct_cols = [col for col in row.index if str(col).startswith('pct_')]

# Build conditional formatting for those columns
    style_data_conditional = [
    {
        'if': {
            'column_id': col,
            'filter_query': f'{{{col}}} = 0 || {{{col}}} > 100'
        },
        'backgroundColor': '#ffcccc',
        'color': 'black'
    }
    for col in pct_cols
    ]
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in row.index],
        data=[row.to_dict()],
        style_table={'overflowX': 'auto', 'marginTop': '32px'},
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
        style_data_conditional=style_data_conditional
    )
    # Pie 1: visits_completed vs visit_target
    pie1 = dcc.Graph(
        figure=go.Figure(
            data=[go.Pie(
                labels=['Visits Completed', 'Remaining'],
                values=[row['visits_completed'], max(row['visit_target'] - row['visits_completed'], 0)],
                hole=0.4,
                marker=dict(colors=['#28a745', '#e0e0e0'])  # Green for completed, gray for remaining
            )],
            layout=go.Layout(title="Visits Completed vs Target")
        )
    )

    # Pie 2: buildings_completed vs building_target
    pie2 = dcc.Graph(
        figure=go.Figure(
            data=[go.Pie(
                labels=['Buildings Completed', 'Remaining'],
                values=[row['buildings_completed'], max(row['building_target'] - row['buildings_completed'], 0)],
                hole=0.4,
                marker=dict(colors=['#28a745', '#e0e0e0'])  # Green for completed, gray for remaining
            )],
            layout=go.Layout(title="Buildings Completed vs Target")
        )
    )

    # Pie 3: du_completed vs du_target
    pie3 = dcc.Graph(
        figure=go.Figure(
            data=[go.Pie(
                labels=['DUs Completed', 'Remaining'],
                values=[row['du_completed'], max(row['du_target'] - row['du_completed'], 0)],
                hole=0.4,
                marker=dict(colors=['#28a745', '#e0e0e0'])  # Green for completed, gray for remaining
            )],
            layout=go.Layout(title="DUs Completed vs Target")
        )
    )

    return html.Div([
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
        html.Hr(),
        html.H4("Actual Data for Selected Ward"),
        table
    ])

if __name__ == "__main__":
    app.run(debug=True)