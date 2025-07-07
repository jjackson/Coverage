"""
Opportunity Comparison Statistics Module

This module generates comparative statistics and visualizations across multiple
CoverageData objects to analyze differences between opportunities/projects.
"""

import os
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Any
import json

try:
    # When imported as a module
    from .models import CoverageData, DeliveryUnit, ServiceDeliveryPoint
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData, DeliveryUnit, ServiceDeliveryPoint


# Handle imports based on how the module is used



def create_opportunity_comparison_report(coverage_data_objects: Dict[str, CoverageData], clumping_ratio: float = 10.0, lookback_days: int = 10) -> str:
    """
    Generate a comparative statistics report for multiple CoverageData objects.
    
    Args:
        coverage_data_objects: Dictionary mapping project keys to CoverageData objects
        clumping_ratio: Ratio threshold for identifying clumped DUs (services_count / building_count)
        lookback_days: Number of days to look back for unique FLW calculation
        
    Returns:
        str: Filename of the generated HTML report
    """
    if len(coverage_data_objects) < 1:
        print("Warning: No CoverageData objects provided for comparison")
        return None
    
    print(f"Generating opportunity analysis report for {len(coverage_data_objects)} project(s)...")
    
    # Generate comparison statistics
    comparison_stats = _generate_comparison_statistics(coverage_data_objects)
    
    # Generate progress data for charts
    progress_data = _generate_progress_data(coverage_data_objects, clumping_ratio, lookback_days)
    
    # Generate HTML report
    html_content = _generate_html_report(comparison_stats, coverage_data_objects, progress_data, lookback_days)
    
    # Write to file
    filename = "opportunity_comparison_report.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Opportunity analysis report saved as: {filename}")
    return filename


def _generate_comparison_statistics(coverage_data_objects: Dict[str, CoverageData]) -> Dict[str, Any]:
    """
    Generate comparative statistics across all CoverageData objects.
    
    Args:
        coverage_data_objects: Dictionary mapping project keys to CoverageData objects
        
    Returns:
        Dict containing comparison statistics
    """
    stats = {
        'project_count': len(coverage_data_objects),
        'projects': {},
        'summary_comparisons': {}
    }
    
    # Extract basic stats for each project
    for project_key, coverage_data in coverage_data_objects.items():
        project_stats = {
            'opportunity_name': getattr(coverage_data, 'opportunity_name', project_key),
            'project_space': getattr(coverage_data, 'project_space', 'Unknown'),
            'delivery_units_count': len(coverage_data.delivery_units) if coverage_data.delivery_units else 0,
            'service_points_count': len(coverage_data.service_points) if coverage_data.service_points else 0,
            'visits_per_day' : coverage_data.get_average_visits_data()[1],
            'completed_dus_count': len([du for du in coverage_data.delivery_units.values() if du.status == 'completed']) if coverage_data.delivery_units else 0,
            'dus_per_day' : coverage_data.get_average_visits_data()[0],
            'total_flws': len(coverage_data.flws) if coverage_data.flws else 0,
            'total_service_areas': len(coverage_data.service_areas) if coverage_data.service_areas else 0,
            'started_sas_count': len([sa for sa in coverage_data.service_areas.values() if sa.is_started]) if coverage_data.service_areas else 0,
            'completed_sas_count': len([sa for sa in coverage_data.service_areas.values() if sa.is_completed]) if coverage_data.service_areas else 0,
            'active_flw_last7days': coverage_data.get_active_flws_last7days()
        }
        
        # Calculate coverage percentage
        if project_stats['delivery_units_count'] > 0:
            project_stats['coverage_percentage'] = (project_stats['completed_dus_count'] / project_stats['delivery_units_count']) * 100
        else:
            project_stats['coverage_percentage'] = 0.0
        
        # Calculate %age active flws
        if project_stats['total_flws'] > 0:
            project_stats['pct_active_flw_last7days'] = (project_stats['active_flw_last7days'] / project_stats['total_flws']) * 100
        else:
            project_stats['pct_active_flw_last7days'] = 0.0
        
        

        
        stats['projects'][project_key] = project_stats
    
    # Generate cross-project comparisons
    stats['summary_comparisons'] = {
        'total_delivery_units': sum(p['delivery_units_count'] for p in stats['projects'].values()),
        'total_service_points': sum(p['service_points_count'] for p in stats['projects'].values()),
        'total_completed_dus': sum(p['completed_dus_count'] for p in stats['projects'].values()),
        'total_service_areas': sum(p['total_service_areas'] for p in stats['projects'].values()),
        'total_started_sas': sum(p['started_sas_count'] for p in stats['projects'].values()),
        'total_completed_sas': sum(p['completed_sas_count'] for p in stats['projects'].values()),
        'average_coverage': sum(p['coverage_percentage'] for p in stats['projects'].values()) / len(stats['projects']) if stats['projects'] else 0,
        'total_flws': sum(p['total_flws'] for p in stats['projects'].values()),
    }
    
    return stats


def _generate_progress_data(coverage_data_objects: Dict[str, CoverageData], clumping_ratio: float = 10.0, lookback_days: int = 10) -> Dict[str, Any]:
    """
    Generate progress data for charting service deliveries and completed DUs over time.
    
    Args:
        coverage_data_objects: Dictionary mapping project keys to CoverageData objects
        clumping_ratio: Ratio threshold for identifying clumped DUs (services_count / building_count)
        lookback_days: Number of days to look back for unique FLW calculation
        
    Returns:
        Dict containing progress data for charts
    """
    progress_data = {
        'daily': {
            'service_delivery': {},
            'du_completion': {},
            'clumped_dus': {},
            'service_to_building_ratio': {}
        },
        'cumulative': {
            'service_delivery': {},
            'du_completion': {},
            'clumped_dus': {},
            'service_to_building_ratio': {}
        }
    }
    
    for project_key, coverage_data in coverage_data_objects.items():
        opportunity_name = getattr(coverage_data, 'opportunity_name', project_key)
        
        # Process service delivery data
        service_delivery_by_day = {}
        for point in coverage_data.service_points:
            if point.visit_date:
                visit_date = pd.to_datetime(point.visit_date).date()
                if visit_date not in service_delivery_by_day:
                    service_delivery_by_day[visit_date] = 0
                service_delivery_by_day[visit_date] += 1
        
        # Process DU completion data and identify clumped DUs
        # JJ: When this was first written was having a lot of issues with the NaT and str issues, I think now resolved.-
        du_completion_by_day = {}
        clumped_dus_by_day = {}
        
        for du in coverage_data.delivery_units.values():
            if du.status == 'completed':
                if isinstance(du.computed_du_completion_date, datetime):
                    completion_date = du.computed_du_completion_date.date()
                    # Add this check to catch NaT that slipped through
                    if pd.isna(completion_date):
                        #print(f"DU {du.du_name} is marked as completed but has no computed completion date, ignoring this DU in oppurtunity statistics")
                        continue
                    
                    # Track regular DU completions
                    if completion_date not in du_completion_by_day:
                        du_completion_by_day[completion_date] = 0
                    du_completion_by_day[completion_date] += 1
                    
                    # Check if this is a clumped DU
                    is_clumped = False
                    clumping_value = len(du.service_points) / du.buildings
                    if clumping_value > clumping_ratio:
                        is_clumped = True

                    # Track clumped DU completions
                    if is_clumped:
                        if completion_date not in clumped_dus_by_day:
                            clumped_dus_by_day[completion_date] = []
                        clumped_dus_by_day[completion_date].append(du)
                else:
                    #print(f"DU {du.du_name} is marked as completed but has no computed completion date, ignoring this DU in oppurtunity statistics")    
                    continue
                 
        # Convert to days since start for each opportunity
        if service_delivery_by_day:
            first_service_date = min(service_delivery_by_day.keys())
            daily_service_data = []
            cumulative_service_data = []
            cumulative_services = 0
            
            # Create a complete date range
            if service_delivery_by_day:
                last_service_date = max(service_delivery_by_day.keys())
                current_date = first_service_date
                while current_date <= last_service_date:
                    day_number = (current_date - first_service_date).days
                    daily_count = service_delivery_by_day.get(current_date, 0)
                    cumulative_services += daily_count
                    
                    daily_service_data.append({
                        'day': day_number,
                        'count': daily_count
                    })
                    
                    cumulative_service_data.append({
                        'day': day_number,
                        'count': cumulative_services
                    })
                    
                    current_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
                    current_date = current_date.date()
            
            progress_data['daily']['service_delivery'][opportunity_name] = daily_service_data
            progress_data['cumulative']['service_delivery'][opportunity_name] = cumulative_service_data
        
        if du_completion_by_day:
            first_completion_date = min(du_completion_by_day.keys())
            daily_du_data = []
            cumulative_du_data = []
            cumulative_dus = 0
            
            # Create a complete date range
            if du_completion_by_day:
                last_completion_date = max(du_completion_by_day.keys())
                current_date = first_completion_date
                while current_date <= last_completion_date:
                    day_number = (current_date - first_completion_date).days
                    daily_count = du_completion_by_day.get(current_date, 0)
                    cumulative_dus += daily_count
                    
                    daily_du_data.append({
                        'day': day_number,
                        'count': daily_count
                    })
                    
                    cumulative_du_data.append({
                        'day': day_number,
                        'count': cumulative_dus
                    })
                    
                    current_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
                    current_date = current_date.date()
            
            progress_data['daily']['du_completion'][opportunity_name] = daily_du_data
            progress_data['cumulative']['du_completion'][opportunity_name] = cumulative_du_data
        
        # Process clumped DUs progress data
        if clumped_dus_by_day:
            daily_clumped_data = []
            cumulative_clumped_data = []
            cumulative_clumped = 0
            
            # Create a complete date range
            if clumped_dus_by_day:
                last_clumped_date = max(clumped_dus_by_day.keys())
                current_date = first_completion_date
                while current_date <= last_clumped_date:
                    day_number = (current_date - first_completion_date).days
                    daily_clumped_dus = clumped_dus_by_day.get(current_date, [])
                    daily_count = len(daily_clumped_dus)
                    cumulative_clumped += daily_count
                    
                    # Calculate unique FLWs who completed clumped DUs in the past N days
                    lookback_start_date = current_date - pd.Timedelta(days=lookback_days-1)
                    unique_flws_in_lookback = set()
                    
                    # Look through all dates in the lookback window
                    check_date = lookback_start_date
                    while check_date <= current_date:
                        if check_date in clumped_dus_by_day:
                            for clumped_du in clumped_dus_by_day[check_date]:
                                # Get FLW associated with this DU
                                flw = coverage_data.flws[clumped_du.flw_commcare_id]
                                unique_flws_in_lookback.add(flw.id)
                        check_date += pd.Timedelta(days=1)
                    
                    daily_clumped_data.append({
                        'day': day_number,
                        'count': daily_count,
                        'clumped_dus': daily_clumped_dus,
                        'unique_flws_in_lookback': list(unique_flws_in_lookback),
                        'unique_flws_count_in_lookback': len(unique_flws_in_lookback)
                    })
                    
                    cumulative_clumped_data.append({
                        'day': day_number,
                        'count': cumulative_clumped,
                        'clumped_dus': daily_clumped_dus,
                        'unique_flws_in_lookback': list(unique_flws_in_lookback),
                        'unique_flws_count_in_lookback': len(unique_flws_in_lookback)
                    })
                    
                    current_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
                    current_date = current_date.date()
            
            progress_data['daily']['clumped_dus'][opportunity_name] = daily_clumped_data
            progress_data['cumulative']['clumped_dus'][opportunity_name] = cumulative_clumped_data
        
        # Calculate service-to-building ratio progress
        if du_completion_by_day:
            daily_ratio_data = []
            cumulative_ratio_data = []
            
            # Sort completion dates to process chronologically
            completion_dates = sorted(du_completion_by_day.keys())
            
            for current_date in completion_dates:
                day_number = (current_date - first_completion_date).days
                
                # Calculate daily ratio (services delivered on this day / buildings in DUs completed this day)
                daily_services = 0
                daily_buildings = 0
                
                # Calculate cumulative ratio (all services up to this date / all buildings in completed DUs up to this date)
                cumulative_services = 0
                cumulative_buildings = 0
                
                for du in coverage_data.delivery_units.values():
                    if (du.status == 'completed' and 
                        isinstance(du.computed_du_completion_date, datetime) and 
                        not pd.isna(du.computed_du_completion_date)):
                        
                        du_completion_date = du.computed_du_completion_date.date()
                        
                        # For daily: include this DU if it was completed on current_date
                        if du_completion_date == current_date:
                            daily_services += len(du.service_points)
                            daily_buildings += du.buildings
                        
                        # For cumulative: include this DU if it was completed on or before current_date
                        if du_completion_date <= current_date:
                            cumulative_services += len(du.service_points)
                            cumulative_buildings += du.buildings
                
                # Calculate ratios (avoid division by zero)
                daily_ratio = daily_services / daily_buildings if daily_buildings > 0 else 0
                cumulative_ratio = cumulative_services / cumulative_buildings if cumulative_buildings > 0 else 0
                
                daily_ratio_data.append({
                    'day': day_number,
                    'count': daily_ratio,
                    'total_services': daily_services,
                    'total_buildings': daily_buildings
                })
                
                cumulative_ratio_data.append({
                    'day': day_number,
                    'count': cumulative_ratio,
                    'total_services': cumulative_services,
                    'total_buildings': cumulative_buildings
                })
            
            progress_data['daily']['service_to_building_ratio'][opportunity_name] = daily_ratio_data
            progress_data['cumulative']['service_to_building_ratio'][opportunity_name] = cumulative_ratio_data
    
    return progress_data


def _generate_html_report(comparison_stats: Dict[str, Any], coverage_data_objects: Dict[str, CoverageData], progress_data: Dict[str, Any], lookback_days: int = 10) -> str:
    """
    Generate HTML content for the comparison report.
    
    Args:
        comparison_stats: Dictionary containing comparison statistics
        coverage_data_objects: Dictionary mapping project keys to CoverageData objects
        progress_data: Dictionary containing progress data for charts
        lookback_days: Number of days to look back for unique FLW calculation
        
    Returns:
        str: HTML content for the report
    """
    
    # Generate project comparison table
    project_rows = ""
    for project_key, project_stats in comparison_stats['projects'].items():
        project_rows += f"""
        <tr>
            <td>{project_stats['opportunity_name']}</td>
            <td>{project_stats['project_space']}</td>
            <td>{project_stats['delivery_units_count']}</td>
            <td>{project_stats['completed_dus_count']}</td>
            <td>{project_stats['dus_per_day']}</td>
            <td>{project_stats['service_points_count']}</td>
            <td>{project_stats['visits_per_day']}</td>
            <td>{project_stats['total_service_areas']}</td>
            <td>{project_stats['started_sas_count']}</td>
            <td>{project_stats['completed_sas_count']}</td>
            <td>{project_stats['total_flws']}</td>
            <td>{project_stats['active_flw_last7days']}</td>
            <td>{project_stats['pct_active_flw_last7days']:.1f}%</td>
            <td>{project_stats['coverage_percentage']:.1f}%</td>
        </tr>
        """
    
    # Convert progress data to JSON for JavaScript
    progress_data_json = json.dumps(progress_data, default=str)
    
    # Determine if this is a single project or comparison
    is_single_project = len(coverage_data_objects) == 1
    report_title = "Opportunity Analysis Report" if is_single_project else "Opportunity Comparison Report"
    note_text = "Progress charts show days since the opportunity's first active day (Day 0 = first service delivery or DU completion)." if is_single_project else "Progress charts show days since each opportunity's first active day (Day 0 = first service delivery or DU completion for that opportunity)."
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid #4CAF50;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 20px 0;
        }}
        .chart-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .timestamp {{
            color: #777;
            font-size: 0.9em;
            margin-top: 30px;
        }}
        .note {{
            background-color: #e3f2fd;
            border: 1px solid #2196F3;
            color: #1565C0;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .chart-toggle {{
            text-align: center;
            margin: 20px 0;
        }}
        .toggle-btn {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            color: #495057;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }}
        .toggle-btn:hover {{
            background-color: #e9ecef;
        }}
        .toggle-btn.active {{
            background-color: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_title}</h1>
        
        <div class="note">
            <strong>Note:</strong> {note_text}
        </div>
        
        <h2>Summary Statistics</h2>
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['project_count']}</div>
                <div class="stat-label">Total Projects</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_delivery_units']}</div>
                <div class="stat-label">Total Delivery Units</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_completed_dus']}</div>
                <div class="stat-label">Completed DUs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_service_areas']}</div>
                <div class="stat-label">Total Service Areas</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_started_sas']}</div>
                <div class="stat-label">Started SAs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_completed_sas']}</div>
                <div class="stat-label">Completed SAs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_service_points']}</div>
                <div class="stat-label">Total Service Points</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['average_coverage']:.1f}%</div>
                <div class="stat-label">Average Coverage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{comparison_stats['summary_comparisons']['total_flws']}</div>
                <div class="stat-label">Total FLWs</div>
            </div>
        </div>
        
        <h2>Project Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Opportunity Name</th>
                    <th>Project Space</th>
                    <th>Total DUs</th>
                    <th>Completed DUs</th>
                    <th>DUs per Day</th>
                    <th>Service Points</th>
                    <th>Forms per Day</th>
                    <th>Total SA</th>
                    <th>Started SAs</th>
                    <th>Completed SAs</th>
                    <th>FLWs</th>
                    <th>Active FLWs</th>
                    <th>Active FLWs %</th>
                    <th>Coverage %</th>
                </tr>
            </thead>
            <tbody>
                {project_rows}
            </tbody>
        </table>
        
        <h2>Progress Comparison Charts</h2>
        
        <div class="chart-toggle">
            <button id="toggle-daily" class="toggle-btn active">Show Daily</button>
            <button id="toggle-cumulative" class="toggle-btn">Show Cumulative</button>
        </div>
        
        <div class="chart-grid">
            <div class="chart-item">
                <div class="chart-title" id="service-chart-title">Daily Service Deliveries</div>
                <div id="service-chart" style="height: 400px;"></div>
            </div>
            <div class="chart-item">
                <div class="chart-title" id="du-chart-title">Daily DU Completions</div>
                <div id="du-chart" style="height: 400px;"></div>
            </div>
            <div class="chart-item">
                <div class="chart-title" id="flws-chart-title">FLWs clumping in trailing {lookback_days} days</div>
                <div id="flws-clumping-chart" style="height: 400px;"></div>
            </div>
            <div class="chart-item">
                <div class="chart-title" id="ratio-chart-title">Service-to-Building Ratio in Completed DUs</div>
                <div id="service-building-ratio-chart" style="height: 400px;"></div>
            </div>
        </div>
        
        <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <script>
        // Progress data from Python
        const progressData = {progress_data_json};
        
        // Color palette for different opportunities
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
        
        // Current view mode
        let currentView = 'daily';
        
        // Create service deliveries chart
        function createServiceChart() {{
            const traces = [];
            let colorIndex = 0;
            
            for (const [opportunity, data] of Object.entries(progressData[currentView]['service_delivery'])) {{
                if (data && data.length > 0) {{
                    traces.push({{
                        x: data.map(d => d.day),
                        y: data.map(d => d.count),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: opportunity,
                        line: {{ color: colors[colorIndex % colors.length] }},
                        marker: {{ size: 6 }}
                    }});
                    colorIndex++;
                }}
            }}
            
            const title = currentView === 'daily' ? 'Daily Service Deliveries' : 'Cumulative Service Deliveries';
            const yAxisTitle = currentView === 'daily' ? 'Number of Service Deliveries' : 'Cumulative Service Deliveries';
            
            const layout = {{
                xaxis: {{ title: 'Days Since First Active Day' }},
                yaxis: {{ title: yAxisTitle }},
                hovermode: 'x unified',
                showlegend: true,
                margin: {{ l: 50, r: 50, t: 30, b: 50 }}
            }};
            
            Plotly.newPlot('service-chart', traces, layout, {{responsive: true}});
        }}
        
        // Create DU completions chart
        function createDUChart() {{
            const traces = [];
            let colorIndex = 0;
            
            for (const [opportunity, data] of Object.entries(progressData[currentView]['du_completion'])) {{
                if (data && data.length > 0) {{
                    traces.push({{
                        x: data.map(d => d.day),
                        y: data.map(d => d.count),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: opportunity,
                        line: {{ color: colors[colorIndex % colors.length] }},
                        marker: {{ size: 6 }}
                    }});
                    colorIndex++;
                }}
            }}
            
            const title = currentView === 'daily' ? 'Daily DU Completions' : 'Cumulative DU Completions';
            const yAxisTitle = currentView === 'daily' ? 'Number of DUs Completed' : 'Cumulative DUs Completed';
            
            const layout = {{
                xaxis: {{ title: 'Days Since First Active Day' }},
                yaxis: {{ title: yAxisTitle }},
                hovermode: 'x unified',
                showlegend: true,
                margin: {{ l: 50, r: 50, t: 30, b: 50 }}
            }};
            
            Plotly.newPlot('du-chart', traces, layout, {{responsive: true}});
        }}
        
        // Create FLWs clumping chart
        function createFLWsClumpingChart() {{
            const traces = [];
            let colorIndex = 0;
            
            for (const [opportunity, data] of Object.entries(progressData[currentView]['clumped_dus'])) {{
                if (data && data.length > 0) {{
                    traces.push({{
                        x: data.map(d => d.day),
                        y: data.map(d => d.unique_flws_count_in_lookback),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: opportunity,
                        line: {{ color: colors[colorIndex % colors.length] }},
                        marker: {{ size: 6 }}
                    }});
                    colorIndex++;
                }}
            }}
            
            const title = currentView === 'daily' ? 'Daily FLWs Clumping' : 'Cumulative FLWs Clumping';
            const yAxisTitle = currentView === 'daily' ? 'Number of Unique FLWs' : 'Cumulative Unique FLWs';
            
            const layout = {{
                xaxis: {{ title: 'Days Since First Active Day' }},
                yaxis: {{ title: yAxisTitle }},
                hovermode: 'x unified',
                showlegend: true,
                margin: {{ l: 50, r: 50, t: 30, b: 50 }}
            }};
            
            Plotly.newPlot('flws-clumping-chart', traces, layout, {{responsive: true}});
        }}
        
        // Create service-to-building ratio chart
        function createServiceBuildingRatioChart() {{
            const traces = [];
            let colorIndex = 0;
            
            for (const [opportunity, data] of Object.entries(progressData[currentView]['service_to_building_ratio'])) {{
                if (data && data.length > 0) {{
                    traces.push({{
                        x: data.map(d => d.day),
                        y: data.map(d => d.count),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: opportunity,
                        line: {{ color: colors[colorIndex % colors.length] }},
                        marker: {{ size: 6 }},
                        customdata: data.map(d => [d.total_services, d.total_buildings]),
                        hovertemplate: 
                            '<b>%{{fullData.name}}</b><br>' +
                            'Day: %{{x}}<br>' +
                            'Ratio: %{{y:.2f}}<br>' +
                            'Total Services: %{{customdata[0]}}<br>' +
                            'Total Buildings: %{{customdata[1]}}' +
                            '<extra></extra>'
                    }});
                    colorIndex++;
                }}
            }}
            
            const title = currentView === 'daily' ? 'Daily Service-to-Building Ratio' : 'Cumulative Service-to-Building Ratio';
            const yAxisTitle = currentView === 'daily' ? 'Daily Services per Building' : 'Cumulative Services per Building';
            
            const layout = {{
                xaxis: {{ title: 'Days Since First Active Day' }},
                yaxis: {{ title: yAxisTitle }},
                hovermode: 'x unified',
                showlegend: true,
                margin: {{ l: 50, r: 50, t: 30, b: 50 }}
            }};
            
            Plotly.newPlot('service-building-ratio-chart', traces, layout, {{responsive: true}});
        }}
        
        // Function to update all charts based on current view
        function updateAllCharts() {{
            createServiceChart();
            createDUChart();
            createFLWsClumpingChart();
            createServiceBuildingRatioChart();
            updateChartTitles();
        }}
        
        // Function to update chart titles
        function updateChartTitles() {{
            document.getElementById('service-chart-title').textContent = 
                currentView === 'daily' ? 'Daily Service Deliveries' : 'Cumulative Service Deliveries';
            document.getElementById('du-chart-title').textContent = 
                currentView === 'daily' ? 'Daily DU Completions' : 'Cumulative DU Completions';
            document.getElementById('flws-chart-title').textContent = 
                currentView === 'daily' ? 'FLWs clumping in trailing {lookback_days} days (Daily)' : 'FLWs clumping in trailing {lookback_days} days (Cumulative)';
            document.getElementById('ratio-chart-title').textContent = 
                currentView === 'daily' ? 'Daily Service-to-Building Ratio in Completed DUs' : 'Cumulative Service-to-Building Ratio in Completed DUs';
        }}
        
        // Toggle event handlers
        function setupToggleHandlers() {{
            document.getElementById('toggle-daily').addEventListener('click', function() {{
                currentView = 'daily';
                document.getElementById('toggle-daily').classList.add('active');
                document.getElementById('toggle-cumulative').classList.remove('active');
                updateAllCharts();
            }});
            
            document.getElementById('toggle-cumulative').addEventListener('click', function() {{
                currentView = 'cumulative';
                document.getElementById('toggle-cumulative').classList.add('active');
                document.getElementById('toggle-daily').classList.remove('active');
                updateAllCharts();
            }});
        }}
        
        // Initialize all charts when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            setupToggleHandlers();
            updateAllCharts();
        }});
    </script>
</body>
</html>"""
    
    return html_content 