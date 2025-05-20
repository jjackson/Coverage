import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
import argparse
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from geopy.distance import geodesic
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .models import CoverageData
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData

# Function to calculate the shortest path distance within each Service Area using TSP
def calculate_shortest_path_distances(df):
    sa_distances = {}
    for sa_id, group in df.groupby('service_area_id'):
        # Improved parsing of centroid data
        centroids = []
        for c in group['centroid']:
            if isinstance(c, str):
                # Attempt to parse the string into a tuple of floats
                try:
                    lat, lon = map(float, c.split())
                    centroids.append((lat, lon))
                except ValueError:
                    raise ValueError(f"Centroid '{c}' for Service Area {sa_id} is not in a valid format.")
            elif isinstance(c, (list, tuple)) and len(c) == 2:
                centroids.append(tuple(c))
            else:
                raise ValueError(f"Centroid '{c}' for Service Area {sa_id} is not in a valid format.")

        # Calculate total distance using geodesic distance
        total_distance = 0
        for i in range(len(centroids) - 1):
            total_distance += geodesic(centroids[i], centroids[i + 1]).kilometers
        # Add distance from last to first to complete the cycle
        total_distance += geodesic(centroids[-1], centroids[0]).kilometers

        sa_distances[sa_id] = total_distance
    return sa_distances

def create_statistics_report(excel_file=None, service_delivery_csv=None, coverage_data=None):
    """
    Create statistics report from the DU Export Excel file and service delivery CSV
    or directly from a CoverageData object.
    
    Args:
        excel_file: Path to the DU Export Excel file (not needed if coverage_data is provided)
        service_delivery_csv: Optional path to service delivery GPS coordinates CSV (not needed if coverage_data is provided)
        coverage_data: Optional CoverageData object containing the already loaded data
    """
    # Either use provided coverage_data or load from files
    if coverage_data is None:
        if excel_file is None:
            raise ValueError("Either coverage_data or excel_file must be provided")
        # Load data using the CoverageData model
        coverage_data = CoverageData.from_excel_and_csv(excel_file, service_delivery_csv)
    
    # Use the DataFrame directly from CoverageData
    delivery_units_df = coverage_data.delivery_units_df
    
    # Use the convenience method to get service points DataFrame
    service_df = coverage_data.create_service_points_dataframe()
    
    # Generate figures for the report
    figures = []
    
    # Use precomputed values from CoverageData for summary statistics
    total_units = coverage_data.total_delivery_units
    total_buildings = coverage_data.total_buildings
    completed_dus = coverage_data.total_completed_dus
    visited_dus = coverage_data.total_visited_dus
    unvisited_dus = coverage_data.total_unvisited_dus
    delivery_progress = coverage_data.completion_percentage
    
    unique_service_areas = coverage_data.total_service_areas
    unique_flws = coverage_data.total_flws
    
    # Get status counts from precomputed values
    status_counts = coverage_data.delivery_status_counts
    
    # 1. Status Distribution by FLW
    # Use pre-computed status distribution for all FLWs
    status_counts_by_flw = coverage_data.get_status_counts_by_flw()
    
    # Create a Plotly figure
    fig1 = go.Figure()
    
    # Add traces for each status
    for status in coverage_data.unique_status_values:
        status_data = status_counts_by_flw[status_counts_by_flw['du_status'] == status]
        
        fig1.add_trace(go.Bar(
            x=status_data['flw_name'],
            y=status_data['count'],
            name=status.title(),
            hovertemplate='<b>%{x}</b><br>' +
                          'Status: ' + status.title() + '<br>' +
                          'Count: %{y}<extra></extra>'
        ))
    
    # Update layout
    fig1.update_layout(
        title='Delivery Unit Status by Field Worker (Filterable)',
        xaxis_title='Field Worker',
        yaxis_title='Count',
        barmode='stack',
        height=500,
        legend=dict(
            title='Status',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        )
    )
    
    # Total status counts for summary view (use precomputed values)
    status_counts_df = pd.DataFrame({
        'Status': coverage_data.unique_status_values,
        'Count': [coverage_data.delivery_status_counts[status] for status in coverage_data.unique_status_values]
    })
    
    # Create a second figure for summary view (not filterable)
    fig1_summary = px.bar(
        status_counts_df, 
        x='Status', 
        y='Count',
        title='Overall Distribution of Delivery Unit Status',
        text='Count',
        color='Status',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig1_summary.update_traces(texttemplate='%{text}', textposition='outside')
    
    # Add both charts to figures
    figures.append(create_plotly_div(fig1) + '<div class="summary-view">' + create_plotly_div(fig1_summary) + '</div>')
    
    # 2. Delivery Progress by Service Area and FLW
    # Get service area progress
    service_area_progress = coverage_data.get_service_area_progress()
    
    # Sort and get top service areas for summary view
    top_service_areas = service_area_progress.sort_values('percentage', ascending=False).head(10)
    
    # Create a summary chart (not filterable)
    fig2_summary = px.bar(
        top_service_areas,
        x='service_area_id',
        y='percentage',
        title='Top 10 Service Areas by Delivery Progress (%)',
        text=top_service_areas['percentage'].round(1).astype(str) + '%',
        color='percentage',
        color_continuous_scale='YlGnBu'
    )
    fig2_summary.update_traces(textposition='outside')
    fig2_summary.add_shape(
        type="line",
        line=dict(dash='dash', color='red'),
        y0=100, y1=100, x0=-0.5, x1=len(top_service_areas)-0.5
    )
    
    # Now use pre-computed service area progress by FLW
    flw_service_area_progress = coverage_data.get_flw_service_area_progress()
    
    # Create filterable chart
    fig2 = go.Figure()
    
    # Add a trace for each FLW's service area progress
    for flw in sorted(flw_service_area_progress['flw_name'].unique()):
        flw_data = flw_service_area_progress[flw_service_area_progress['flw_name'] == flw]
        
        fig2.add_trace(go.Bar(
            x=flw_data['service_area_id'],
            y=flw_data['percentage'],
            name=flw,
            text=flw_data['percentage'].round(1).astype(str) + '%',
            textposition='outside',
            hovertext=flw_data.apply(lambda row: 
                f"<b>{row['flw_name']}</b><br>" +
                f"Service Area: {row['service_area_id']}<br>" +
                f"Progress: {row['percentage']:.1f}%<br>" +
                f"Completed: {row['completed_dus']}/{row['total_dus']}", 
                axis=1
            ),
            hoverinfo='text'
        ))
    
    # Add a horizontal line at 100%
    fig2.add_shape(
        type='line',
        x0=-0.5,
        x1=len(flw_service_area_progress['service_area_id'].unique())-0.5,
        y0=100,
        y1=100,
        line=dict(
            color='red',
            width=2,
            dash='dash',
        )
    )
    
    # Update layout
    fig2.update_layout(
        title='Service Area Progress by Field Worker (Filterable)',
        xaxis_title='Service Area ID',
        yaxis_title='Completion Percentage (%)',
        barmode='group',
        height=600,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        )
    )
    
    # Add both charts to figures
    figures.append(create_plotly_div(fig2) + '<div class="summary-view">' + create_plotly_div(fig2_summary) + '</div>')
    
    # 3. FLW Performance with interactive filtering
    # Get all FLWs with their completion rates using pre-computed data
    flw_data = coverage_data.get_flw_completion_data()
    
    # Make sure we sort by completion rate descending for initial view
    sorted_flw_data = flw_data.sort_values('completion_rate', ascending=False)
    
    # Create hover text with detailed information
    hover_text = [
        f"<b>{row['flw_name']}</b><br>" +
        f"Completion Rate: {row['completion_rate']:.1f}%<br>" +
        f"Completed: {row['completed_units']}/{row['assigned_units']}<br>" +
        f"Service Areas: {row['service_areas']}"
        for _, row in sorted_flw_data.iterrows()
    ]
    
    fig3 = go.Figure()
    
    # Add the bar trace
    fig3.add_trace(go.Bar(
        x=sorted_flw_data['flw_name'],
        y=sorted_flw_data['completion_rate'],
        text=sorted_flw_data['completion_rate'].round(1).astype(str) + '%',
        textposition='outside',
        marker=dict(
            color=sorted_flw_data['completion_rate'],
            colorscale='RdYlGn',
            colorbar=dict(title='Completion Rate %')
        ),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Add a horizontal line at 100%
    fig3.add_shape(
        type='line',
        x0=-0.5,
        x1=len(sorted_flw_data)-0.5,
        y0=100,
        y1=100,
        line=dict(
            color='red',
            width=2,
            dash='dash',
        )
    )
    
    # Update layout for better appearance
    fig3.update_layout(
        title='FLW Completion Rates (Interactive & Filterable)',
        xaxis=dict(
            title='Field Worker',
            tickangle=45,
            tickfont=dict(size=10),
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(
            title='Completion Rate (%)',
            range=[0, max(sorted_flw_data['completion_rate'].max() * 1.1, 100)]
        ),
        height=600,
        margin=dict(b=100)  # Add more bottom margin for the rangeslider
    )
    
    figures.append(create_plotly_div(fig3))
    
    # 4. Add a searchable FLW table with completion rates
    # Create a more interactive and filterable table
    flw_table = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>FLW Name</b>', '<b>Completion Rate (%)</b>', '<b>Completed Units</b>', '<b>Total Units</b>', '<b>Service Areas</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left',
            height=30
        ),
        cells=dict(
            values=[
                sorted_flw_data['flw_name'],
                sorted_flw_data['completion_rate'].round(1),
                sorted_flw_data['completed_units'],
                sorted_flw_data['assigned_units'],
                sorted_flw_data['service_areas']
            ],
            fill_color=[['#f8f9fa', '#edf2f7'] * len(sorted_flw_data)],
            align='left',
            font=dict(size=11),
            height=25,
        ),
        columnwidth=[3, 2, 2, 2, 5]
    )])
    
    flw_table.update_layout(
        title='FLW Performance Table (Searchable & Filterable)',
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # Add custom attributes to help with filtering
    flw_table_html = create_plotly_div(flw_table)
    flw_table_html = flw_table_html.replace('<div ', '<div data-flw-table="true" ')
    figures.append(flw_table_html)
    
    # 5. Building density per Service Area and FLW (if building data available)
    if '#Buildings' in delivery_units_df.columns:
        # Use pre-computed building density data
        building_density = coverage_data.get_building_density()
        
        # Sort and get top density areas for summary
        top_density = building_density.sort_values('density', ascending=False).head(10)
        
        # Create summary chart
        fig5_summary = px.bar(
            top_density,
            x='service_area_id',
            y='density',
            title='Top 10 Service Areas by Building Density',
            text=top_density['density'].round(1),
            color='density',
            color_continuous_scale='YlOrRd'
        )
        fig5_summary.update_traces(textposition='outside')
        
        # Now use pre-computed FLW building density
        flw_building_density = coverage_data.get_flw_building_density()
        
        # Create filterable chart
        fig5 = go.Figure()
        
        # Add a trace for each FLW
        for flw in sorted(flw_building_density['flw_name'].unique()):
            flw_data = flw_building_density[flw_building_density['flw_name'] == flw]
            
            fig5.add_trace(go.Bar(
                x=flw_data['service_area_id'],
                y=flw_data['density'],
                name=flw,
                text=flw_data['density'].round(1).astype(str),
                textposition='outside',
                hovertext=flw_data.apply(lambda row: 
                    f"<b>{row['flw_name']}</b><br>" +
                    f"Service Area: {row['service_area_id']}<br>" +
                    f"Density: {row['density']:.1f} buildings/km²<br>" +
                    f"Buildings: {row['#Buildings']}<br>" +
                    f"Area: {row['Surface Area (sq. meters)'] / 1000000:.2f} km²", 
                    axis=1
                ),
                hoverinfo='text'
            ))
        
        # Update layout
        fig5.update_layout(
            title='Building Density by Field Worker and Service Area (Filterable)',
            xaxis_title='Service Area ID',
            yaxis_title='Building Density (buildings/km²)',
            barmode='group',
            height=600,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=10)
            )
        )
        
        # Add both charts to figures
        figures.append(create_plotly_div(fig5) + '<div class="summary-view">' + create_plotly_div(fig5_summary) + '</div>')
    
    # 6. Service Delivery by Date (if service data available)
    if service_df is not None:
        # Service Delivery by Date (if date column exists)
        if 'service_date' in service_df.columns or 'date' in service_df.columns:
            date_col = 'service_date' if 'service_date' in service_df.columns else 'date'
            
            # Convert to datetime if not already
            service_df[date_col] = pd.to_datetime(service_df[date_col], errors='coerce')
            
            # Group by date and count for summary
            date_counts = service_df.groupby(service_df[date_col].dt.date).size().reset_index(name='count')
            date_counts = date_counts.sort_values(date_col)
            
            # Create summary chart (not filterable)
            fig6_summary = px.line(
                date_counts,
                x=date_col,
                y='count',
                title='Overall Service Delivery by Date',
                markers=True
            )
            
            # Check if we have FLW data in the service data
            if 'flw_name' in service_df.columns or 'flw' in service_df.columns:
                flw_col = 'flw_name' if 'flw_name' in service_df.columns else 'flw'
                
                # Group by date and FLW
                flw_date_counts = service_df.groupby([service_df[date_col].dt.date, flw_col]).size().reset_index(name='count')
                flw_date_counts = flw_date_counts.sort_values(date_col)
                
                # Create filterable chart
                fig6 = go.Figure()
                
                # Add a trace for each FLW
                for flw in sorted(flw_date_counts[flw_col].unique()):
                    flw_data = flw_date_counts[flw_date_counts[flw_col] == flw]
                    
                    fig6.add_trace(go.Scatter(
                        x=flw_data[date_col],
                        y=flw_data['count'],
                        mode='lines+markers',
                        name=flw,
                        hovertemplate=f"<b>{flw}</b><br>Date: %{{x|%Y-%m-%d}}<br>Count: %{{y}}<extra></extra>"
                    ))
                
                # Update layout
                fig6.update_layout(
                    title='Service Delivery by Field Worker Over Time (Filterable)',
                    xaxis_title='Date',
                    yaxis_title='Number of Deliveries',
                    height=500
                )
                
                # Add both charts
                figures.append(create_plotly_div(fig6) + '<div class="summary-view">' + create_plotly_div(fig6_summary) + '</div>')
            else:
                # If no FLW data, just add the summary view
                figures.append(create_plotly_div(fig6_summary))
    
    # Calculate assignment statistics
    # Use the objects to get the statistics
    sa_per_user = {flw_name: len(flw.service_areas) for flw_name, flw in coverage_data.flws.items()}
    min_sa = min(sa_per_user.values()) if sa_per_user else 0
    max_sa = max(sa_per_user.values()) if sa_per_user else 0
    stddev_sa = np.std(list(sa_per_user.values())) if sa_per_user else 0
    median_sa = np.median(list(sa_per_user.values())) if sa_per_user else 0

    # Calculate the number of Delivery Units each user is assigned
    du_per_user = {flw_name: flw.assigned_units for flw_name, flw in coverage_data.flws.items()}
    min_du = min(du_per_user.values()) if du_per_user else 0
    max_du = max(du_per_user.values()) if du_per_user else 0
    stddev_du = np.std(list(du_per_user.values())) if du_per_user else 0
    median_du = np.median(list(du_per_user.values())) if du_per_user else 0

    # Get travel distances (if computed in the model)
    # Calculate travel distances using the CoverageData model
    coverage_data.calculate_travel_distances()
    distance_per_user_df = coverage_data.get_travel_distances_by_flw()

    # Check if we have any distance data
    if not distance_per_user_df.empty:
        # Calculate statistics for user distances
        min_user_distance = distance_per_user_df['total_distance'].min()
        max_user_distance = distance_per_user_df['total_distance'].max() 
        stddev_user_distance = distance_per_user_df['total_distance'].std()
        median_user_distance = distance_per_user_df['total_distance'].median()
    else:
        # Fallback values if no centroids available
        min_user_distance = 0
        max_user_distance = 0
        stddev_user_distance = 0
        median_user_distance = 0

    # Calculate the number of SAs with 50 or more DUs
    sas_50_plus_ids = coverage_data.delivery_units_df.groupby('service_area_id').filter(lambda x: len(x) >= 50)['service_area_id'].unique()
    sas_50_plus = len(sas_50_plus_ids)

    # Identify unique users with at least one SA having more than 50 DUs
    flws_with_50_plus_dus = coverage_data.delivery_units_df[coverage_data.delivery_units_df['service_area_id'].isin(sas_50_plus_ids)]['flw_commcare_id'].unique()
    dus_50_plus_users = len(flws_with_50_plus_dus)  # Count of unique FLWs with SAs having 50+ DUs

    # Calculate the number of SAs with between 35 and 50 DUs
    sas_35_to_50_ids = coverage_data.delivery_units_df.groupby('service_area_id').filter(lambda x: 35 <= len(x) < 50)['service_area_id'].unique()
    sas_35_to_50 = len(sas_35_to_50_ids)

    # Identify unique users with SAs having between 35 and 50 DUs
    flws_with_35_to_50_dus = coverage_data.delivery_units_df[coverage_data.delivery_units_df['service_area_id'].isin(sas_35_to_50_ids)]['flw_commcare_id'].unique()
    users_sas_35_to_50 = len(flws_with_35_to_50_dus)

    # Create the HTML report
    html_content = create_html_report(delivery_units_df, service_df, figures, coverage_data)
    
    # Add statistics for DUs
    html_content += f"""
    <section>
        <h2>User Assignment Statistics</h2>
        <table style='width:100%; border-collapse: collapse;'>
            <tr>
                <th style='border: 1px solid #ddd; padding: 8px;'>Statistic</th>
                <th style='border: 1px solid #ddd; padding: 8px;'>Min</th>
                <th style='border: 1px solid #ddd; padding: 8px;'>Max</th>
                <th style='border: 1px solid #ddd; padding: 8px;'>StdDev</th>
                <th style='border: 1px solid #ddd; padding: 8px;'>Median</th>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>Service Areas per User</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{min_sa}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{max_sa}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{stddev_sa:.2f}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{median_sa}</td>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>Delivery Units per User</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{min_du}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{max_du}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{stddev_du:.2f}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{median_du}</td>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>User Travel Distance</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{min_user_distance}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{max_user_distance}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{stddev_user_distance:.2f}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{median_user_distance}</td>
            </tr>
        </table>
    </section>
    """
    
    # Add key metrics section
    html_content += f"""
    <section>
        <h2>Key Metrics</h2>
        <table style='width:100%; border-collapse: collapse;'>
            <tr>
                <th style='border: 1px solid #ddd; padding: 8px;'>Metric</th>
                <th style='border: 1px solid #ddd; padding: 8px;'>Count</th>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>SAs with >= 50 DUs</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{sas_50_plus}</td>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>Users with >= 50 DUs</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{dus_50_plus_users}</td>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>SAs with 35-50 DUs</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{sas_35_to_50}</td>
            </tr>
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>Users with SAs 35-50 DUs</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{users_sas_35_to_50}</td>
            </tr>
        </table>
    </section>
    """
    
    # Write the HTML to a file
    output_filename = "coverage_statistics.html"
    with open(output_filename, "w") as f:
        f.write(html_content)
    
    print(f"Statistics report has been created: {output_filename}")
    return output_filename

def create_plotly_div(fig):
    """Create a self-contained HTML div with the Plotly figure"""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_html_report(delivery_df, service_df, figures, coverage_data):
    """
    Create HTML report with the statistics information
    
    Args:
        delivery_df: DataFrame with delivery unit data
        service_df: DataFrame with service delivery data
        figures: List of base64-encoded figures
        coverage_data: CoverageData object with precomputed values
    """
    # Calculate summary statistics
    total_units = coverage_data.total_delivery_units
    total_buildings = coverage_data.total_buildings
    completed_dus = coverage_data.total_completed_dus
    visited_dus = coverage_data.total_visited_dus
    unvisited_dus = coverage_data.total_unvisited_dus
    
    delivery_progress = coverage_data.completion_percentage
    
    unique_service_areas = coverage_data.total_service_areas
    unique_flws = coverage_data.total_flws
    
    status_counts = coverage_data.delivery_status_counts
    
    # Service delivery stats
    service_stats = {}
    if service_df is not None:
        total_service_points = len(service_df)
        service_stats['total_points'] = total_service_points
        
        if 'flw_name' in service_df.columns:
            service_stats['unique_flws'] = service_df['flw_name'].nunique()
    
    # Create FLW data for filtering
    # First check which column to use for counting delivery units
    count_column = 'du_id'
    if count_column not in delivery_df.columns:
        # Try alternative column names
        if 'id' in delivery_df.columns:
            count_column = 'id'
        else:
            # Use the first column as a last resort
            count_column = delivery_df.columns[0]
    
    # Create FLW data for filtering
    flw_data = delivery_df.groupby('flw_commcare_id').agg({
        'du_status': lambda x: (x == 'completed').sum(),
        count_column: 'count',
        'service_area_id': lambda x: ', '.join(str(i) for i in sorted(set(x)))  # Convert all values to strings before joining
    }).reset_index()
    
    # Create a mapping from flw_commcare_id to display name
    flw_name_map = {}
    # First try to use the name mapping from coverage_data
    if hasattr(coverage_data, 'flw_commcare_id_to_name_map') and coverage_data.flw_commcare_id_to_name_map:
        flw_name_map = coverage_data.flw_commcare_id_to_name_map
    # If that's not available, fallback to the FLW objects in coverage_data
    elif hasattr(coverage_data, 'flws') and coverage_data.flws:
        for flw_id, flw_obj in coverage_data.flws.items():
            flw_name_map[flw_id] = flw_obj.name
    
    # Add a display name column using the mapping
    flw_data['flw_name'] = flw_data['flw_commcare_id'].apply(
        lambda id: flw_name_map.get(id, id)  # Use the ID itself if no name mapping exists
    )
    
    flw_data.rename(columns={
        count_column: 'assigned_units',
        'du_status': 'completed_units',
        'service_area_id': 'service_areas'
    }, inplace=True)
    
    flw_data['completion_rate'] = flw_data['completed_units'] / flw_data['assigned_units'] * 100
    flw_data['completion_rate'] = flw_data['completion_rate'].fillna(0)
    
    # Convert to JSON for JavaScript
    flw_json = json.dumps([{
        'flw_name': row['flw_name'],  # Display name for UI
        'flw_id': row['flw_commcare_id'],  # Internal ID for filtering
        'completed_units': int(row['completed_units']),
        'assigned_units': int(row['assigned_units']),
        'completion_rate': float(row['completion_rate']),
        'service_areas': row['service_areas']
    } for _, row in flw_data.iterrows()])
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Coverage Statistics Report</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <!-- Include jQuery for interactive features -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        
        <!-- Include Select2 for better dropdowns -->
        <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
        <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>
        
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background-color: #fff;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .stat-card h3 {{
                margin-top: 0;
                color: #7f8c8d;
                font-size: 16px;
                font-weight: normal;
            }}
            .stat-card .value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
                margin: 10px 0;
            }}
            .figure-container {{
                margin: 30px 0;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                color: #95a5a6;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }}
            .progress-container {{
                width: 100%;
                background-color: #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .progress-bar {{
                height: 24px;
                border-radius: 5px;
                background-color: #27ae60;
                text-align: center;
                line-height: 24px;
                color: white;
                font-weight: bold;
            }}
            .filter-controls {{
                margin: 15px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e9ecef;
            }}
            .filter-controls label {{
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }}
            .filter-controls select {{
                width: 100%;
                max-width: 600px;
                margin-bottom: 10px;
            }}
            .filter-controls button {{
                padding: 8px 15px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 10px;
            }}
            .filter-controls button#reset-flw-filter {{
                background-color: #6c757d;
            }}
            .filter-controls button:hover {{
                background-color: #0069d9;
            }}
            .filter-controls button#reset-flw-filter:hover {{
                background-color: #5a6268;
            }}
            .tab-container {{
                margin-top: 20px;
            }}
            .tab-buttons {{
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }}
            .tab-buttons button {{
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 16px;
            }}
            .tab-buttons button:hover {{
                background-color: #ddd;
            }}
            .tab-buttons button.active {{
                background-color: #ccc;
            }}
            .tab-content {{
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }}
            .tab-content.active {{
                display: block;
            }}
            #search-flw {{
                width: 100%;
                max-width: 300px;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }}
            .summary-view {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px dashed #ccc;
            }}
            .summary-view:before {{
                content: "Summary View (Not Filtered)";
                display: block;
                font-weight: bold;
                margin-bottom: 10px;
                color: #6c757d;
            }}
            /* View toggle controls */
            .view-toggle {{
                margin: 10px 0;
                text-align: right;
            }}
            .view-toggle button {{
                padding: 5px 10px;
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 4px;
                cursor: pointer;
                margin-left: 5px;
            }}
            .view-toggle button.active {{
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }}
            /* Hide summary view when filtered */
            .filtering-active .summary-view {{
                display: none;
            }}
        </style>
        
        <!-- Add custom search functionality for FLW table and filtering -->
        <script>
            $(document).ready(function() {{
                // Search functionality for FLW table
                $("#search-flw").on("keyup", function() {{
                    var value = $(this).val().toLowerCase();
                    $(".flw-row").filter(function() {{
                        $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
                    }});
                }});
                
                // Tab functionality
                $(".tab-button").click(function() {{
                    var tabId = $(this).attr("data-tab");
                    
                    $(".tab-button").removeClass("active");
                    $(".tab-content").removeClass("active");
                    
                    $(this).addClass("active");
                    $("#" + tabId).addClass("active");
                }});
                
                // Show the first tab by default
                $(".tab-button:first").click();
                
                // FLW Filter functionality
                // Populate FLW dropdown with all unique FLWs
                const flwData = {flw_json};
                const uniqueFlws = [...new Set(flwData.map(item => item.flw_name))].sort();
                
                uniqueFlws.forEach(flw => {{
                    $("#flw-selector").append(`<option value="${{flw}}">${{flw}}</option>`);
                }});
                
                // Make the select element more user-friendly with select2 (if available)
                if (typeof $.fn.select2 !== 'undefined') {{
                    $("#flw-selector").select2({{
                        placeholder: "Select Field Workers",
                        allowClear: true,
                        width: '100%',
                        maximumSelectionLength: 10
                    }});
                }}
                
                // Apply filter button click handler
                $("#apply-flw-filter").click(function() {{
                    const selectedFlws = $("#flw-selector").val();
                    console.log("Selected FLWs:", selectedFlws);
                    
                    // If "All Field Workers" is selected or nothing selected, don't filter
                    if (selectedFlws.includes("all") || selectedFlws.length === 0) {{
                        resetFilters();
                        return;
                    }}
                    
                    // Create a mapping of names to IDs for lookup
                    const flwNameToIdMap = {{}};
                    flwData.forEach(item => {{
                        flwNameToIdMap[item.flw_name] = item.flw_id;
                    }});
                    
                    // Get the corresponding IDs for the selected names
                    const selectedFlwIds = selectedFlws.map(name => flwNameToIdMap[name]);
                    console.log("Selected FLW IDs:", selectedFlwIds);
                    
                    // Add filtering-active class to hide summary views
                    $(".tab-content").addClass("filtering-active");
                    
                    // Apply filter to all Plotly charts
                    $(".js-plotly-plot").each(function() {{
                        const plotDiv = $(this).attr("id");
                        if (!plotDiv) return;
                        
                        // Skip if inside a summary-view div
                        if ($(this).closest(".summary-view").length > 0) {{
                            return; // Don't filter summary views
                        }}
                        
                        // Check if this is the FLW table - special handling
                        const isTable = $(this).attr("data-flw-table") === "true";
                        
                        const plotObj = document.getElementById(plotDiv);
                        if (!plotObj || !window.Plotly) return;
                        
                        // Get the plotly data
                        const data = plotObj.data;
                        if (!data || data.length === 0) return;
                        
                        console.log("Processing chart:", plotDiv, isTable ? "(FLW Table)" : "");
                        
                        // Special handling for FLW table - use name-based filtering
                        if (isTable) {{
                            // For the table, we'll filter the rows
                            const tableData = data[0];
                            
                            // Save original data if not already saved
                            if (!plotObj._originalData) {{
                                plotObj._originalData = JSON.parse(JSON.stringify(data));
                            }}
                            
                            // Get the FLW names column (should be the first column)
                            const flwNames = tableData.cells.values[0];
                            
                            // Create filtered values arrays
                            const newValues = tableData.cells.values.map(column => 
                                column.filter((_, i) => selectedFlws.includes(flwNames[i]))
                            );
                            
                            // Update the table with filtered data
                            Plotly.restyle(plotDiv, {{'cells.values': [newValues]}});
                            return;
                        }}
                        
                        // Save original data if not already saved
                        if (!plotObj._originalData) {{
                            plotObj._originalData = JSON.parse(JSON.stringify(data));
                        }}
                        
                        // The rest of the logic depends on the chart type and how FLW data is displayed
                        // We need to check how FLW data is represented in this specific chart
                        
                        // For charts where traces have FLW names as their 'name' property (like service area progress)
                        if (data[0].name && uniqueFlws.includes(data[0].name)) {{
                            const visibility = data.map(trace => 
                                selectedFlws.includes(trace.name) ? true : 'legendonly'
                            );
                            
                            Plotly.restyle(plotDiv, {{ visible: visibility }});
                        }}
                        // For charts where x-axis has FLW names (like FLW performance chart)
                        else if (data[0].x && uniqueFlws.some(flw => data[0].x.includes(flw))) {{
                            // Create a new data array that only includes the selected FLWs
                            const newData = [];
                            
                            for (let i = 0; i < data.length; i++) {{
                                const trace = data[i];
                                
                                // Create filtered x and y arrays
                                const newX = [];
                                const newY = [];
                                const newText = trace.text ? [] : undefined;
                                const newHovertext = trace.hovertext ? [] : undefined;
                                
                                for (let j = 0; j < trace.x.length; j++) {{
                                    if (selectedFlws.includes(trace.x[j])) {{
                                        newX.push(trace.x[j]);
                                        newY.push(trace.y[j]);
                                        if (newText) newText.push(trace.text[j]);
                                        if (newHovertext) newHovertext.push(trace.hovertext[j]);
                                    }}
                                }}
                                
                                // Create new trace with filtered data
                                const newTrace = {{
                                    x: newX,
                                    y: newY,
                                    type: trace.type,
                                    mode: trace.mode,
                                    name: trace.name,
                                    marker: trace.marker
                                }};
                                
                                if (newText) newTrace.text = newText;
                                if (newHovertext) newTrace.hovertext = newHovertext;
                                if (trace.hoverinfo) newTrace.hoverinfo = trace.hoverinfo;
                                if (trace.textposition) newTrace.textposition = trace.textposition;
                                
                                newData.push(newTrace);
                            }}
                            
                            // Replace the plot with filtered data
                            Plotly.react(plotDiv, newData, plotObj.layout);
                        }}
                    }});
                }});
                
                // Reset filter button click handler
                $("#reset-flw-filter").click(resetFilters);
                
                function resetFilters() {{
                    console.log("Resetting filters");
                    
                    // Reset dropdown
                    $("#flw-selector").val(["all"]);
                    if (typeof $.fn.select2 !== 'undefined') {{
                        $("#flw-selector").trigger('change');
                    }}
                    
                    // Remove filtering-active class to show summary views again
                    $(".tab-content").removeClass("filtering-active");
                    
                    // Reset all charts to show all data
                    $(".js-plotly-plot").each(function() {{
                        // Skip if inside a summary-view div
                        if ($(this).closest(".summary-view").length > 0) {{
                            return; // Don't reset summary views
                        }}
                        
                        const plotDiv = $(this).attr("id");
                        if (!plotDiv) return;
                        
                        const plotObj = document.getElementById(plotDiv);
                        if (!plotObj || !window.Plotly) return;
                        
                        // If we have saved the original data, restore it
                        if (plotObj._originalData) {{
                            Plotly.react(plotDiv, plotObj._originalData, plotObj.layout);
                        }}
                    }});
                }}
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Coverage Statistics Report</h1>
            
            <section>
                <h2>Summary</h2>
                <div class="summary-grid">
                    <div class="stat-card">
                        <h3>Total Delivery Units</h3>
                        <div class="value">{total_units:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Completed DUs</h3>
                        <div class="value">{completed_dus:,}</div>
                        <div>{(completed_dus/total_units*100):.1f}% of total</div>
                    </div>
                    <div class="stat-card">
                        <h3>Visited DUs (not completed)</h3>
                        <div class="value">{visited_dus:,}</div>
                        <div>{(visited_dus/total_units*100):.1f}% of total</div>
                    </div>
                    <div class="stat-card">
                        <h3>Unvisited DUs</h3>
                        <div class="value">{unvisited_dus:,}</div>
                        <div>{(unvisited_dus/total_units*100):.1f}% of total</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Buildings (computed during Segmentation) </h3>
                        <div class="value">{total_buildings:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Service Areas</h3>
                        <div class="value">{unique_service_areas:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Field Workers</h3>
                        <div class="value">{unique_flws:,}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Overall Delivery Progress</h3>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {min(delivery_progress, 100)}%">
                                {delivery_progress:.1f}%
                            </div>
                        </div>
                        <div style="font-size: 14px; margin-top: 5px;">
                            ({completed_dus:,} of {total_units:,} DUs completed)
                        </div>
                    </div>
                </div>
            </section>
            
            <section>
                <h2>Delivery Unit Status</h2>
                <div class="summary-grid">
    """
    
    # Add status cards
    for status, count in coverage_data.delivery_status_counts.items():
        percentage = (count / total_units) * 100
        html_content += f"""
                    <div class="stat-card">
                        <h3>{status.title()}</h3>
                        <div class="value">{count:,}</div>
                        <div>{percentage:.1f}% of total</div>
                    </div>
        """
    
    html_content += """
                </div>
            </section>
    """
    
    # Add service delivery stats if available
    if service_df is not None and service_stats:
        html_content += f"""
            <section>
                <h2>Service Delivery</h2>
                <div class="summary-grid">
                    <div class="stat-card">
                        <h3>Total Service Points</h3>
                        <div class="value">{service_stats.get('total_points', 0):,}</div>
                    </div>
        """
        
        if 'unique_flws' in service_stats:
            html_content += f"""
                    <div class="stat-card">
                        <h3>Field Workers</h3>
                        <div class="value">{service_stats.get('unique_flws', 0):,}</div>
                    </div>
            """
        
        html_content += """
                </div>
            </section>
        """
    
    # Add interactive visualization section with FLW data for filtering
    html_content += """
            <section>
                <h2>Analysis Visualizations</h2>
                
                <div class="filter-controls">
                    <label for="flw-selector">Filter by Field Workers:</label>
                    <select id="flw-selector" multiple>
                        <option value="all" selected>All Field Workers</option>
                        <!-- FLW options will be populated via JavaScript -->
                    </select>
                    <button id="apply-flw-filter">Apply Filter</button>
                    <button id="reset-flw-filter">Reset</button>
                </div>
                
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-button" data-tab="tab-status">Status Distribution</button>
                        <button class="tab-button" data-tab="tab-service-areas">Service Areas</button>
                        <button class="tab-button" data-tab="tab-flw">Field Worker Performance</button>
                        <button class="tab-button" data-tab="tab-density">Building Density</button>
                        <button class="tab-button" data-tab="tab-delivery">Service Delivery</button>
                    </div>
                    
                    <div id="tab-status" class="tab-content">
                        <div class="figure-container">
                            %s
                        </div>
                    </div>
                    
                    <div id="tab-service-areas" class="tab-content">
                        <div class="figure-container">
                            %s
                        </div>
                    </div>
                    
                    <div id="tab-flw" class="tab-content">
                        <input type="text" id="search-flw" placeholder="Search for Field Worker...">
                        <div class="figure-container">
                            %s
                        </div>
                        <div class="figure-container">
                            %s
                        </div>
                    </div>
                    
                    <div id="tab-density" class="tab-content">
                        <div class="figure-container">
                            %s
                        </div>
                    </div>
                    
                    <div id="tab-delivery" class="tab-content">
                        <div class="figure-container">
                            %s
                        </div>
                    </div>
                </div>
            </section>
    """ % tuple(figures[:6] + [''] * (6 - len(figures)))
    
    # Close the HTML
    html_content += f"""
            <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create statistics report from Excel and CSV data")
    parser.add_argument("--excel", help="Excel file containing delivery unit data")
    parser.add_argument("--csv", help="CSV file containing service delivery data")
    args = parser.parse_args()
    
    excel_file = None
    delivery_csv = None
    
    # If arguments are provided, use them
    if args.excel and args.csv:
        excel_file = args.excel
        delivery_csv = args.csv
        
        print(f"\nCreating statistics report using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the statistics report using CoverageData model
        output_file = create_statistics_report(excel_file=excel_file, service_delivery_csv=delivery_csv)
        print(f"Statistics report created: {output_file}")
    else:
        # Interactive selection
        # Get all files in the current directory
        files = glob.glob('*.*')
        
        # Filter for Excel and CSV files
        excel_files = [f for f in files if f.lower().endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        
        # Handle Excel file selection
        if len(excel_files) == 0:
            print("No Excel files found in the current directory.")
            exit(1)
        elif len(excel_files) == 1:
            excel_choice = 1
            print(f"\nAutomatically selected the only available Excel file: {excel_files[0]}")
        else:
            # Display Excel files
            print("\nAvailable Excel files for microplanning:")
            for i, file in enumerate(excel_files, 1):
                print(f"{i}. {file}")
            
            # Get user selection for Excel file
            excel_choice = 0
            while excel_choice < 1 or excel_choice > len(excel_files):
                try:
                    excel_choice = int(input(f"\nEnter the number for the microplanning Excel file (1-{len(excel_files)}): "))
                except ValueError:
                    print("Please enter a valid number.")
        
        # Handle CSV file selection
        if len(csv_files) == 0:
            print("No CSV files found in the current directory.")
            exit(1)
        elif len(csv_files) == 1:
            csv_choice = 1
            print(f"\nAutomatically selected the only available CSV file: {csv_files[0]}")
        else:
            # Display CSV files
            print("\nAvailable CSV files for service delivery data:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            
            # Get user selection for CSV file
            csv_choice = 0
            while csv_choice < 1 or csv_choice > len(csv_files):
                try:
                    csv_choice = int(input(f"\nEnter the number for the service delivery CSV file (1-{len(csv_files)}): "))
                except ValueError:
                    print("Please enter a valid number.")
        
        # Get selected files
        excel_file = excel_files[excel_choice - 1]
        delivery_csv = csv_files[csv_choice - 1]
        
        print(f"\nCreating statistics report using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the statistics report using CoverageData model
        output_file = create_statistics_report(excel_file=excel_file, service_delivery_csv=delivery_csv)
        print(f"Statistics report created: {output_file}") 