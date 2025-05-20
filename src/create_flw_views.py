import pandas as pd
import os
import argparse
from datetime import datetime
import json

# Handle imports based on how the module is used
try:
    # When imported as a module
    from .models import CoverageData
except ImportError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CoverageData

def process_flw_timeline_data(coverage_data):
    """
    Process FLW visit data to create timeline data for ECharts visualization.
    
    Args:
        coverage_data: CoverageData object containing the loaded data
        
    Returns:
        A dictionary with FLW timeline data and stats
    """
    # Get the service points dataframe
    service_df = coverage_data.create_service_points_dataframe()
    
    # Make sure we have the necessary columns
    if 'visit_date' not in service_df.columns or 'flw_name' not in service_df.columns:
        raise ValueError("Service points data missing required columns: 'visit_date' or 'flw_name'")
    
    # Convert visit_date to datetime if it's not already
    if service_df['visit_date'].dtype != 'datetime64[ns]':
        service_df['visit_date'] = pd.to_datetime(service_df['visit_date'], errors='coerce')
    
    # Drop rows with invalid dates
    service_df = service_df.dropna(subset=['visit_date'])
    
    # Create a dictionary to store timeline data for each FLW
    timeline_data = {}
    
    # Get unique FLWs
    flws = service_df['flw_name'].unique().tolist()
    
    # Get global date range
    min_date = service_df['visit_date'].min()
    max_date = service_df['visit_date'].max()
    
    # Process data for each FLW
    for flw in flws:
        flw_data = service_df[service_df['flw_name'] == flw].sort_values('visit_date')
        
        # Create individual visit entries
        visits = []
        
        for _, row in flw_data.iterrows():
            # Create visit info with all available metadata
            visit_info = {
                'timestamp': row['visit_date'].strftime('%Y-%m-%dT%H:%M:%S'),
                'visit_id': str(row.get('visit_id', '')),
                'status': row.get('status', 'unknown'),
                'du_name': row.get('du_name', ''),
                'service_area_id': row.get('service_area_id', ''),
                'accuracy_in_m': float(row.get('accuracy_in_m', 0)),
                'flagged': bool(row.get('flagged', False)),
                'time': row['visit_date'].strftime('%H:%M:%S'),
                'date': row['visit_date'].strftime('%Y-%m-%d'),
            }
            
            visits.append(visit_info)
        
        # Calculate stats for this FLW
        timeline_data[flw] = {
            'visits': visits,
            'total_visits': len(visits),
            'active_days': len(set(v['date'] for v in visits))
        }
    
    return {
        'flws': flws,
        'timeline_data': timeline_data,
        'min_date': min_date.strftime('%Y-%m-%d') if not pd.isnull(min_date) else '',
        'max_date': max_date.strftime('%Y-%m-%d') if not pd.isnull(max_date) else ''
    }

def extract_flw_table_data(coverage_data):
    """
    Extract detailed FLW data for table view showing productivity metrics.
    
    Args:
        coverage_data: CoverageData object containing the loaded data
        
    Returns:
        A list of dictionaries with FLW metrics
    """
    flw_data = []
    
    for flw_id, flw in coverage_data.flws.items():
        # Format dates nicely or use placeholder if None
        first_date = flw.first_service_delivery_date.strftime('%Y-%m-%d') if flw.first_service_delivery_date else 'N/A'
        last_date = flw.last_service_delivery_date.strftime('%Y-%m-%d') if flw.last_service_delivery_date else 'N/A'
        
        # Create entry for this FLW
        entry = {
            'flw_name': flw.name,
            'service_deliveries': len(flw.service_points),
            'first_service_delivery_date': first_date,
            'last_service_delivery_date': last_date,
            'days_active': flw.days_active,
            'completed_units': flw.completed_units,
            'assigned_units': flw.assigned_units,
            'completion_rate': round(flw.completion_rate, 1),
            'units_per_day': round(flw.delivery_units_completed_per_day, 2)
        }
        flw_data.append(entry)
    
    return flw_data

def create_flw_views_report(excel_file=None, service_delivery_csv=None, coverage_data=None):
    """
    Create Field-Level Worker (FLW) views from the DU Export Excel file and service delivery CSV
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
    
    # Process the FLW timeline data
    try:
        timeline_data = process_flw_timeline_data(coverage_data)
    except Exception as e:
        print(f"Error processing timeline data: {e}")
        timeline_data = {
            'flws': [],
            'timeline_data': {},
            'min_date': '',
            'max_date': ''
        }

    # Prepare initial heatmap data (full range, all FLWs)
    try:
        heatmap_data = coverage_data.get_completed_du_heatmap_data()
    except Exception as e:
        print(f"Error processing heatmap data: {e}")
        heatmap_data = {'flws': [], 'flw_names': [], 'dates': [], 'matrix': []}
        
    # Extract FLW table data for productivity metrics
    try:
        flw_table_data = extract_flw_table_data(coverage_data)
    except Exception as e:
        print(f"Error processing FLW table data: {e}")
        flw_table_data = []

    # Create the JavaScript file content
    js_content = f"""
// FLW Timeline Chart Script
// This file is generated automatically - do not edit manually

document.addEventListener('DOMContentLoaded', function() {{
    // Timeline chart setup
    const chartElement = document.getElementById('timeline-chart');
    const chart = echarts.init(chartElement);
    const timelineData = {json.dumps(timeline_data)};
    const heatmapData = {json.dumps(heatmap_data)};
    const flwTableData = {json.dumps(flw_table_data)};

    // Populate FLW select
    const flwSelect = document.getElementById('flw-select');
    timelineData.flws.forEach(flw => {{
        const option = document.createElement('option');
        option.value = flw;
        option.textContent = flw;
        flwSelect.appendChild(option);
    }});
    // Select all FLWs by default
    if (flwSelect.options.length > 0) {{
        for (let i = 0; i < flwSelect.options.length; i++) {{
            flwSelect.options[i].selected = true;
        }}
    }}
    // Set default date range
    document.getElementById('start-date').value = timelineData.min_date;
    document.getElementById('end-date').value = timelineData.max_date;

    // Timeline chart logic
    function formatTooltip(params) {{
        const seriesName = params.seriesName;
        const data = params.data;
        const visitInfo = data[3];
        let content = "<div style='font-weight:bold;margin-bottom:5px;'>" + seriesName + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Date:</b> " + visitInfo.date + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Time:</b> " + visitInfo.time + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Status:</b> " + (visitInfo.status || 'unknown') + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Delivery Unit:</b> " + (visitInfo.du_name || 'N/A') + "</div>";
        if (visitInfo.service_area_id) {{
            content += "<div style='margin-bottom:5px;'><b>Service Area:</b> " + visitInfo.service_area_id + "</div>";
        }}
        if (visitInfo.accuracy_in_m) {{
            content += "<div style='margin-bottom:5px;'><b>GPS Accuracy:</b> " + visitInfo.accuracy_in_m.toFixed(1) + " m</div>";
        }}
        if (visitInfo.flagged) {{
            content += "<div style='margin-bottom:5px;'><b>Flagged:</b> Yes</div>";
        }}
        return content;
    }}
    function updateChart() {{
        const selectedFLWs = Array.from(flwSelect.selectedOptions).map(option => option.value);
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const noDataMessage = document.getElementById('no-data-message');
        if (selectedFLWs.length === 0) {{
            noDataMessage.style.display = 'block';
            chart.clear();
            return;
        }}
        noDataMessage.style.display = 'none';
        const series = [];
        selectedFLWs.forEach((flw, index) => {{
            const flwData = timelineData.timeline_data[flw];
            if (!flwData || !flwData.visits || flwData.visits.length === 0) return;
            const filteredVisits = flwData.visits.filter(visit => {{
                return visit.date >= startDate && visit.date <= endDate;
            }});
            const data = filteredVisits.map(visit => [visit.timestamp, flw, 15, visit]);
            const byDateDU = {{}};
            const specialPoints = new Set();
            filteredVisits.forEach(visit => {{
                const date = visit.date;
                const du = visit.du_name || 'unknown';
                const key = date + '|' + du;
                if (!byDateDU[key]) byDateDU[key] = [];
                byDateDU[key].push(visit);
            }});
            Object.keys(byDateDU).forEach(key => {{
                const visits = byDateDU[key];
                if (visits.length === 1) {{
                    specialPoints.add(visits[0].timestamp);
                }} else if (visits.length >= 2) {{
                    visits.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
                    specialPoints.add(visits[0].timestamp);
                    specialPoints.add(visits[visits.length - 1].timestamp);
                }}
            }});
            if (data.length > 0) {{
                series.push({{
                    name: flw,
                    type: 'scatter',
                    symbol: 'circle',
                    symbolSize: 15,
                    itemStyle: {{
                        color: function(params) {{
                            const pointData = params.data;
                            const timestamp = pointData[0];
                            if (specialPoints.has(timestamp)) {{
                                return {{
                                    type: 'radial',
                                    x: 0.5, y: 0.5, r: 0.5,
                                    colorStops: [
                                        {{ offset: 0, color: '#000' }},
                                        {{ offset: 0.7, color: '#000' }},
                                        {{ offset: 0.71, color: '#fff' }},
                                        {{ offset: 0.85, color: params.color }}
                                    ],
                                    global: false
                                }};
                            }} else {{
                                return params.color;
                            }}
                        }}
                    }},
                    data: data,
                    tooltip: {{ formatter: formatTooltip }}
                }});
                Object.keys(byDateDU).forEach(key => {{
                    const visits = byDateDU[key];
                    if (visits.length >= 2) {{
                        visits.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
                        const firstVisit = visits[0];
                        const lastVisit = visits[visits.length - 1];
                        const parts = key.split('|');
                        const date = parts[0];
                        const du = parts[1];
                        series.push({{
                            name: 'Connection: ' + date + ' - ' + du,
                            type: 'line',
                            data: [
                                [firstVisit.timestamp, flw],
                                [lastVisit.timestamp, flw]
                            ],
                            showSymbol: false,
                            lineStyle: {{ width: 2, color: 'rgba(100, 100, 100, 0.5)', type: 'solid' }},
                            silent: true,
                            tooltip: {{ show: false }}
                        }});
                    }}
                }});
            }}
        }});
        if (series.length === 0 || series.every(s => s.data.length === 0)) {{
            noDataMessage.style.display = 'block';
            chart.clear();
            return;
        }}
        const option = {{
            title: {{ text: 'FLW Activity Timeline', left: 'center' }},
            tooltip: {{ trigger: 'item', enterable: true, confine: true, position: function (point, params, dom, rect, size) {{
                const spaceRight = size.viewSize[0] - point[0];
                const spaceLeft = point[0];
                if (spaceRight < 300 && spaceLeft > 300) return [point[0] - 300, point[1]];
                return [point[0] + 30, point[1]];
            }}, formatter: formatTooltip, backgroundColor: 'rgba(255, 255, 255, 0.9)', borderColor: '#ccc', borderWidth: 1, extraCssText: 'max-width: 300px; max-height: 300px; overflow-y: auto;' }},
            dataZoom: [
                {{ type: 'inside', xAxisIndex: 0, filterMode: 'filter' }},
                {{ type: 'slider', xAxisIndex: 0, height: 20, bottom: 50, filterMode: 'filter' }}
            ],
            grid: {{ left: '10%', right: '5%', top: 60, bottom: 100 }},
            xAxis: {{
                type: 'time',
                splitLine: {{ show: true, lineStyle: {{ type: 'dashed', color: '#ddd' }} }},
                axisLabel: {{
                    formatter: function (value) {{
                        const date = new Date(value);
                        if (date.getHours() === 0 && date.getMinutes() === 0) return date.toLocaleDateString();
                        const hours = date.getHours().toString().padStart(2, '0');
                        const minutes = date.getMinutes().toString().padStart(2, '0');
                        return hours + ':' + minutes;
                    }},
                    rotate: 45, fontSize: 10
                }},
                minInterval: 3600 * 1000,
                maxInterval: 86400 * 1000
            }},
            yAxis: {{
                type: 'category',
                data: selectedFLWs,
                axisLine: {{ lineStyle: {{ color: '#ccc' }} }},
                axisLabel: {{ formatter: function(value) {{ return value; }}, show: true, inside: false, margin: 8, color: '#333', fontWeight: 'normal', rich: {{ hover: {{ color: '#000', fontWeight: 'bold' }} }} }},
                axisTick: {{ alignWithLabel: true }},
                splitLine: {{ show: true, lineStyle: {{ type: 'dashed', color: '#eee' }} }},
                z: 10,
                inverse: true
            }},
            series: series
        }};
        chart.setOption(option);
    }}

    // Heatmap chart setup
    const heatmapElement = document.getElementById('heatmap-chart');
    const heatmapChart = echarts.init(heatmapElement);
    function updateHeatmap() {{
        // Use the matrix and axes as provided by Python (using FLW Name (flw) for the Y axis and tooltip)
        console.log("DEBUG: heatmapData structure:", JSON.stringify(heatmapData, null, 2));
        console.log("DEBUG: flw_names array:", heatmapData.flw_names);
        console.log("DEBUG: flws array:", heatmapData.flws);
        console.log("DEBUG: Are arrays equal length?", (heatmapData.flw_names || []).length === (heatmapData.flws || []).length);
        
        let values = [];
        // Iterate using the same order as the matrix data to ensure alignment
        (heatmapData.flws || []).forEach((flw, i) => {{
            (heatmapData.dates || []).forEach((date, j) => {{
                let count = (heatmapData.matrix && heatmapData.matrix[i]) ? heatmapData.matrix[i][j] : 0;
                values.push([j, i, count]);
            }});
        }});
        const option = {{
            tooltip: {{ position: 'top', formatter: function(params) {{ 
                console.log("DEBUG: Tooltip params:", params);
                // Get the FLW name using the same index as used in the matrix
                const flwName = heatmapData.flw_names[params.value[1]];
                return (flwName || 'Unknown') + '<br>' + (heatmapData.dates[params.value[0]] || '') + ': <b>' + params.value[2] + '</b> completed DUs'; 
            }} }},
            grid: {{ left: 80, bottom: 80, right: 40, top: 40 }},
            xAxis: {{ type: 'category', data: heatmapData.dates, axisLabel: {{ rotate: 45 }} }},
            yAxis: {{ 
                type: 'category', 
                data: heatmapData.flw_names, 
                inverse: true,
                axisLabel: {{ fontSize: 8 }}
            }},
            visualMap: {{ min: 0, max: Math.max(1, ...values.map(v => v[2])), calculable: true, orient: 'horizontal', left: 'center', bottom: 10, inRange: {{ color: ['#ffffe0', '#ffd080', '#ff8040', '#d73027'] }} }},
            series: [{{ name: 'Completed DUs', type: 'heatmap', data: values, label: {{ show: true, color: '#222', fontWeight: 'bold', formatter: function(p) {{ return p.value[2] > 0 ? p.value[2] : ''; }} }}, emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }} }} }}]
        }};
        heatmapChart.setOption(option);
    }}

    // FLW Table functionality
    function populateFlwTable() {{
        const tableBody = document.getElementById('flw-table-body');
        if (!tableBody) return;

        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Sort the data initially by service deliveries (descending)
        const sortedData = [...flwTableData].sort((a, b) => b.service_deliveries - a.service_deliveries);
        
        // Add rows to the table
        sortedData.forEach(flw => {{
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${{flw.flw_name}}</td>
                <td>${{flw.service_deliveries}}</td>
                <td>${{flw.first_service_delivery_date}}</td>
                <td>${{flw.last_service_delivery_date}}</td>
                <td>${{flw.days_active}}</td>
                <td>${{flw.completed_units}}</td>
                <td>${{flw.assigned_units}}</td>
                <td>${{flw.completion_rate}}%</td>
                <td>${{flw.units_per_day}}</td>
            `;
            
            tableBody.appendChild(row);
        }});
    }}
    
    // Table sorting functionality
    function setupTableSorting() {{
        const table = document.getElementById('flw-table');
        const headers = table.querySelectorAll('th');
        const tableBody = document.getElementById('flw-table-body');
        const rows = tableBody.querySelectorAll('tr');
        
        // Define the sort direction for each column (initially all ascending)
        const directions = Array.from(headers).map(() => {{
            return '';
        }});
        
        // Add click listeners to all headers
        headers.forEach((header, index) => {{
            header.addEventListener('click', () => {{
                // Get all rows from the table body
                const rows = Array.from(tableBody.querySelectorAll('tr'));
                // Get the header text
                const headerText = header.textContent.trim();
                // Get the column type to determine sort method
                const isNumber = header.dataset.type === 'number';
                const isDate = header.dataset.type === 'date';
                
                // Direction changes each time header is clicked
                // Empty -> ascending -> descending -> Empty (back to original order)
                const direction = directions[index] === '' ? 'asc' : directions[index] === 'asc' ? 'desc' : '';
                
                // Reset all other directions
                directions.forEach((dir, i) => {{
                    directions[i] = i !== index ? '' : direction;
                }});
                
                // Remove class from all headers
                headers.forEach(header => {{
                    header.classList.remove('asc', 'desc');
                }});
                
                if (direction) {{
                    header.classList.add(direction);
                }}
                
                // Sort the rows
                const sortedRows = [...rows].sort((rowA, rowB) => {{
                    const cellA = rowA.querySelectorAll('td')[index].textContent;
                    const cellB = rowB.querySelectorAll('td')[index].textContent;
                    
                    // Compare based on data type
                    let comparison = 0;
                    if (isNumber) {{
                        // Number comparison: convert to numeric values first
                        const valueA = parseFloat(cellA.replace('%', ''));
                        const valueB = parseFloat(cellB.replace('%', ''));
                        comparison = valueA - valueB;
                    }} else if (isDate) {{
                        // Date comparison: convert to Date objects for comparison
                        // Check if dates are N/A first
                        if (cellA === 'N/A' && cellB === 'N/A') {{
                            comparison = 0;
                        }} else if (cellA === 'N/A') {{
                            comparison = -1;
                        }} else if (cellB === 'N/A') {{
                            comparison = 1;
                        }} else {{
                            const dateA = new Date(cellA);
                            const dateB = new Date(cellB);
                            comparison = dateA - dateB;
                        }}
                    }} else {{
                        // String comparison
                        comparison = cellA.localeCompare(cellB);
                    }}
                    
                    return direction === 'asc' ? comparison : -comparison;
                }});
                
                // Remove existing rows
                rows.forEach(row => {{
                    tableBody.removeChild(row);
                }});
                
                // Add new rows in the right order
                sortedRows.forEach(row => {{
                    tableBody.appendChild(row);
                }});
            }});
        }});
    }}
    
    // Export functionality
    function setupExportButton() {{
        const exportBtn = document.getElementById('export-flw-table');
        if (!exportBtn) return;
        
        exportBtn.addEventListener('click', () => {{
            // Create CSV content
            let csv = 'FLW Name,Service Deliveries,First Service Date,Last Service Date,Days Active,Completed Units,Assigned Units,Completion Rate,Units Per Day\\n';
            
            flwTableData.forEach(flw => {{
                // Properly escape fields with quotes if they contain commas
                const row = [
                    flw.flw_name, 
                    flw.service_deliveries,
                    flw.first_service_delivery_date,
                    flw.last_service_delivery_date,
                    flw.days_active,
                    flw.completed_units,
                    flw.assigned_units,
                    flw.completion_rate,
                    flw.units_per_day
                ];
                csv += row.join(',') + '\\n';
            }});
            
            // Create download link
            const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.setAttribute('href', url);
            link.setAttribute('download', 'flw_productivity_data.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }});
    }}

    // Update both charts together
    document.getElementById('update-chart').addEventListener('click', function() {{
        updateChart();
        updateHeatmap();
    }});
    
    // Initial render
    updateChart();
    updateHeatmap();
    populateFlwTable();
    setupTableSorting();
    setupExportButton();
    
    window.addEventListener('resize', function() {{ 
        chart.resize(); 
        heatmapChart.resize(); 
    }});
}});
"""

    # Create JavaScript file
    js_filename = "flw_timeline.js"
    with open(js_filename, "w", encoding="utf-8") as f:
        f.write(js_content)
    
    # Generate the HTML file with ECharts visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Field-Level Worker (FLW) Views</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js"></script>
        <script src="{js_filename}"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
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
            .timestamp {{
                color: #777;
                font-size: 0.9em;
                margin-top: 30px;
            }}
            .control-panel {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: flex-end;
            }}
            .control-section {{
                flex: 1;
                min-width: 200px;
            }}
            .control-section h3 {{
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 16px;
            }}
            select, input {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }}
            .date-inputs {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            #timeline-chart {{
                width: 100%;
                height: 600px;
                margin-top: 20px;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 5px 0;
            }}
            .stat-label {{
                font-size: 14px;
                color: #666;
            }}
            .tooltip-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 12px;
            }}
            .tooltip-table th, .tooltip-table td {{
                border: 1px solid #ddd;
                padding: 4px 8px;
                text-align: left;
            }}
            .tooltip-table th {{
                background-color: #f2f2f2;
            }}
            .chart-container {{
                position: relative;
            }}
            .chart-message {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                text-align: center;
                display: none;
            }}
            .flw-row-highlight {{
                pointer-events: none;
                position: absolute;
                background-color: rgba(255, 255, 0, 0.15);
                border-top: 1px solid rgba(220, 220, 0, 0.4);
                border-bottom: 1px solid rgba(220, 220, 0, 0.4);
                border-left: 3px solid rgba(220, 220, 0, 0.7);
                z-index: 1;
                box-shadow: 0 0 8px rgba(200, 200, 0, 0.3);
                animation: highlightPulse 2s infinite;
            }}
            
            @keyframes highlightPulse {{
                0% {{ background-color: rgba(255, 255, 0, 0.15); }}
                50% {{ background-color: rgba(255, 255, 0, 0.25); }}
                100% {{ background-color: rgba(255, 255, 0, 0.15); }}
            }}
            #heatmap-chart {{
                width: 100%;
                height: 400px;
                margin-top: 40px;
            }}
            .table-container {{
                margin-top: 40px;
                overflow-x: auto;
            }}
            .flw-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                table-layout: auto;
            }}
            .flw-table th, .flw-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .flw-table th {{
                background-color: #f5f5f5;
                position: relative;
                cursor: pointer;
                padding-right: 25px;
            }}
            .flw-table th::after {{
                content: '⇕';
                position: absolute;
                right: 8px;
                color: #999;
            }}
            .flw-table th.asc::after {{
                content: '↑';
                color: #333;
            }}
            .flw-table th.desc::after {{
                content: '↓';
                color: #333;
            }}
            .flw-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .flw-table tr:hover {{
                background-color: #f0f7ff;
            }}
            .export-button {{
                margin-top: 10px;
                text-align: right;
            }}
            .export-button button {{
                background-color: #2196F3;
                margin-left: 10px;
            }}
            .export-button button:hover {{
                background-color: #0b7dda;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Field-Level Worker (FLW) Activity Timeline</h1>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-label">Total FLWs</div>
                    <div class="stat-value">{coverage_data.total_flws}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Service Deliveries</div>
                    <div class="stat-value">{len(coverage_data.create_service_points_dataframe())}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Service Areas</div>
                    <div class="stat-value">{coverage_data.total_service_areas}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Date Range</div>
                    <div class="stat-value" style="font-size: 16px;">{timeline_data['min_date']} to {timeline_data['max_date']}</div>
                </div>
            </div>
            
            <div class="control-panel">
                <div class="control-section">
                    <h3>Select FLWs</h3>
                    <select id="flw-select" multiple size="5">
                        <!-- FLW options will be populated via JavaScript -->
                    </select>
                </div>
                
                <div class="control-section">
                    <h3>Date Range</h3>
                    <div class="date-inputs">
                        <div>
                            <label for="start-date">Start Date</label>
                            <input type="date" id="start-date">
                        </div>
                        <div>
                            <label for="end-date">End Date</label>
                            <input type="date" id="end-date">
                        </div>
                    </div>
                </div>
                
                <div class="control-section" style="max-width: 150px;">
                    <button id="update-chart">Update Chart</button>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="timeline-chart"></div>
                <div id="no-data-message" class="chart-message">
                    <h3>No Data Available</h3>
                    <p>Please select at least one FLW and ensure the date range contains activity data.</p>
                </div>
            </div>
            <div class="chart-container">
                <div id="heatmap-chart"></div>
            </div>
            
            <div class="table-container">
                <h2>FLW Productivity Metrics</h2>
                <div class="export-button">
                    <button id="export-flw-table">Export to CSV</button>
                </div>
                <table id="flw-table" class="flw-table">
                    <thead>
                        <tr>
                            <th>FLW Name</th>
                            <th data-type="number">Service Deliveries</th>
                            <th data-type="date">First Service Date</th>
                            <th data-type="date">Last Service Date</th>
                            <th data-type="number">Days Active</th>
                            <th data-type="number">Completed Units</th>
                            <th data-type="number">Assigned Units</th>
                            <th data-type="number">Completion Rate</th>
                            <th data-type="number">Units Per Day</th>
                        </tr>
                    </thead>
                    <tbody id="flw-table-body">
                        <!-- Table rows will be populated by JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    output_filename = "flw_views.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"FLW Views report has been created: {output_filename}")
    print(f"JavaScript file created: {js_filename}")
    return output_filename

# Execute the function if run directly
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create FLW views from Excel and CSV data")
    parser.add_argument("--excel", help="Excel file containing delivery unit data")
    parser.add_argument("--csv", help="CSV file containing service delivery data")
    args = parser.parse_args()
    
    excel_file = args.excel
    delivery_csv = args.csv
    
    if excel_file and delivery_csv:
        print(f"\nCreating FLW views using:")
        print(f"Microplanning file: {excel_file}")
        print(f"Service delivery file: {delivery_csv}")
        
        # Create the views
        create_flw_views_report(excel_file=excel_file, service_delivery_csv=delivery_csv)
    else:
        print("Please provide both Excel and CSV files using the --excel and --csv arguments.") 