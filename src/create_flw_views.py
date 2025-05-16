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
    
    # Create the JavaScript file content
    js_content = """// FLW Timeline Chart Script
// This file is generated automatically - do not edit manually

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize ECharts instance
    const chartElement = document.getElementById('timeline-chart');
    const chart = echarts.init(chartElement);
    
    // Load timeline data
    const timelineData = """ + json.dumps(timeline_data) + """;
    
    // Populate FLW select
    const flwSelect = document.getElementById('flw-select');
    timelineData.flws.forEach(flw => {
        const option = document.createElement('option');
        option.value = flw;
        option.textContent = flw;
        flwSelect.appendChild(option);
    });
    
    // Select all FLWs by default
    if (flwSelect.options.length > 0) {
        for (let i = 0; i < flwSelect.options.length; i++) {
            flwSelect.options[i].selected = true;
        }
    }
    
    // Set default date range
    document.getElementById('start-date').value = timelineData.min_date;
    document.getElementById('end-date').value = timelineData.max_date;
    
    // Function to format tooltip content
    function formatTooltip(params) {
        const seriesName = params.seriesName;
        const data = params.data;
        
        const visitInfo = data[3]; // Custom data with visit details
        
        let content = "<div style='font-weight:bold;margin-bottom:5px;'>" + seriesName + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Date:</b> " + visitInfo.date + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Time:</b> " + visitInfo.time + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Status:</b> " + (visitInfo.status || 'unknown') + "</div>";
        content += "<div style='margin-bottom:5px;'><b>Delivery Unit:</b> " + (visitInfo.du_name || 'N/A') + "</div>";
        
        if (visitInfo.service_area_id) {
            content += "<div style='margin-bottom:5px;'><b>Service Area:</b> " + visitInfo.service_area_id + "</div>";
        }
        
        if (visitInfo.accuracy_in_m) {
            content += "<div style='margin-bottom:5px;'><b>GPS Accuracy:</b> " + visitInfo.accuracy_in_m.toFixed(1) + " m</div>";
        }
        
        if (visitInfo.flagged) {
            content += "<div style='margin-bottom:5px;'><b>Flagged:</b> Yes</div>";
        }
        
        return content;
    }
    
    // Function to update the chart
    function updateChart() {
        // Get selected FLWs
        const selectedFLWs = Array.from(flwSelect.selectedOptions).map(option => option.value);
        
        // Get date range
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        // Show "no data" message if no FLWs are selected
        const noDataMessage = document.getElementById('no-data-message');
        if (selectedFLWs.length === 0) {
            noDataMessage.style.display = 'block';
            chart.clear();
            return;
        }
        
        noDataMessage.style.display = 'none';
        
        // Prepare the series data for each selected FLW
        const series = [];
        
        selectedFLWs.forEach((flw, index) => {
            const flwData = timelineData.timeline_data[flw];
            
            if (!flwData || !flwData.visits || flwData.visits.length === 0) return;
            
            // Filter visits by date range
            const filteredVisits = flwData.visits.filter(visit => {
                return visit.date >= startDate && visit.date <= endDate;
            });
            
            // Create data points for the chart - one point per visit
            const data = filteredVisits.map(visit => {
                return [
                    visit.timestamp,   // x-axis: exact timestamp
                    flw,               // y-axis: FLW name
                    15,                // fixed symbolSize for individual visits
                    visit              // custom data for tooltip
                ];
            });
            
            // Group visits by date and DU name to mark first/last/standalone visits
            const byDateDU = {};
            const specialPoints = new Set(); // Store keys for first/last/standalone visits
            
            // First, group visits
            filteredVisits.forEach(visit => {
                const date = visit.date;
                const du = visit.du_name || 'unknown';
                const key = date + '|' + du;
                
                if (!byDateDU[key]) {
                    byDateDU[key] = [];
                }
                
                byDateDU[key].push(visit);
            });
            
            // Now identify special points
            Object.keys(byDateDU).forEach(key => {
                const visits = byDateDU[key];
                
                if (visits.length === 1) {
                    // Standalone visit - mark it
                    specialPoints.add(visits[0].timestamp);
                } else if (visits.length >= 2) {
                    // Sort by timestamp to find first and last
                    visits.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
                    
                    // Mark first and last
                    specialPoints.add(visits[0].timestamp);
                    specialPoints.add(visits[visits.length - 1].timestamp);
                }
            });
            
            // Add main scatter series
            if (data.length > 0) {
                series.push({
                    name: flw,
                    type: 'scatter',
                    symbol: 'circle',
                    symbolSize: 15,    // Fixed size for all visits
                    itemStyle: {
                        color: function(params) {
                            const pointData = params.data;
                            const timestamp = pointData[0];
                            
                            if (specialPoints.has(timestamp)) {
                                // Return gradient for special points (black center with colored border)
                                return {
                                    type: 'radial',
                                    x: 0.5,
                                    y: 0.5,
                                    r: 0.5,
                                    colorStops: [
                                        { offset: 0, color: '#000' }, // black center
                                        { offset: 0.7, color: '#000' }, // black until 70% radius
                                        { offset: 0.71, color: '#fff' }, // white ring
                                        { offset: 0.85, color: params.color } // original color at edge
                                    ],
                                    global: false
                                };
                            } else {
                                // Regular visits keep their default color
                                return params.color;
                            }
                        }
                    },
                    data: data,
                    tooltip: {
                        formatter: formatTooltip
                    }
                });
                
                // Create line series for connecting first and last visit in each DU per day
                Object.keys(byDateDU).forEach(key => {
                    const visits = byDateDU[key];
                    
                    if (visits.length >= 2) {
                        // Sort by timestamp
                        visits.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
                        
                        const firstVisit = visits[0];
                        const lastVisit = visits[visits.length - 1];
                        const parts = key.split('|');
                        const date = parts[0];
                        const du = parts[1];
                        
                        // Add a line series
                        series.push({
                            name: 'Connection: ' + date + ' - ' + du,
                            type: 'line',
                            data: [
                                [firstVisit.timestamp, flw], 
                                [lastVisit.timestamp, flw]
                            ],
                            showSymbol: false,
                            lineStyle: {
                                width: 2,
                                color: 'rgba(100, 100, 100, 0.5)',
                                type: 'solid'
                            },
                            silent: true,
                            tooltip: {
                                show: false
                            }
                        });
                    }
                });
            }
        });
        
        // If no data after filtering, show message
        if (series.length === 0 || series.every(s => s.data.length === 0)) {
            noDataMessage.style.display = 'block';
            chart.clear();
            return;
        }
        
        // Set chart options
        const option = {
            title: {
                text: 'FLW Activity Timeline',
                left: 'center'
            },
            tooltip: {
                trigger: 'item',
                enterable: true, // Allow mouse to enter the tooltip
                confine: true,   // Keep tooltip inside chart area
                position: function (point, params, dom, rect, size) {
                    // Position tooltip to avoid being cut off
                    const spaceRight = size.viewSize[0] - point[0];
                    const spaceLeft = point[0];
                    
                    if (spaceRight < 300 && spaceLeft > 300) {
                        // If more space on the left, position left
                        return [point[0] - 300, point[1]];
                    }
                    // Default position (to the right)
                    return [point[0] + 30, point[1]];
                },
                formatter: formatTooltip,
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderColor: '#ccc',
                borderWidth: 1,
                extraCssText: 'max-width: 300px; max-height: 300px; overflow-y: auto;',
            },
            dataZoom: [
                {
                    type: 'inside',
                    xAxisIndex: 0,
                    filterMode: 'filter'
                },
                {
                    type: 'slider',
                    xAxisIndex: 0,
                    height: 20,
                    bottom: 50,
                    filterMode: 'filter'
                }
            ],
            grid: {
                left: '10%',
                right: '5%',
                top: 60,
                bottom: 100
            },
            xAxis: {
                type: 'time',
                splitLine: {
                    show: true,
                    lineStyle: {
                        type: 'dashed',
                        color: '#ddd'
                    }
                },
                axisLabel: {
                    formatter: function (value) {
                        const date = new Date(value);
                        
                        // Check if this is midnight (start of day)
                        if (date.getHours() === 0 && date.getMinutes() === 0) {
                            // For day boundaries, show the date
                            return date.toLocaleDateString();
                        } else {
                            // For other times, just show the time (HH:MM format)
                            const hours = date.getHours().toString().padStart(2, '0');
                            const minutes = date.getMinutes().toString().padStart(2, '0');
                            return hours + ':' + minutes;
                        }
                    },
                    rotate: 45,
                    fontSize: 10
                },
                minInterval: 3600 * 1000, // Minimum interval of 1 hour
                maxInterval: 86400 * 1000 // Maximum interval of 1 day
            },
            yAxis: {
                type: 'category',
                data: selectedFLWs,
                axisLine: {
                    lineStyle: {
                        color: '#ccc'
                    }
                },
                axisLabel: {
                    formatter: function(value) {
                        return value;
                    },
                    show: true,
                    inside: false,
                    margin: 8,
                    color: '#333',
                    fontWeight: 'normal',
                    // Show hover effect
                    rich: {
                        hover: {
                            color: '#000',
                            fontWeight: 'bold'
                        }
                    }
                },
                axisTick: {
                    alignWithLabel: true
                },
                splitLine: {
                    show: true,
                    lineStyle: {
                        type: 'dashed',
                        color: '#eee'
                    }
                },
                z: 10
            },
            series: series
        };
        
        // Generate day boundary vertical lines
        if (startDate && endDate) {
            const markLines = [];
            const startDt = new Date(startDate);
            const endDt = new Date(endDate);
            
            // Set to start of day
            startDt.setHours(0, 0, 0, 0);
            
            // Create a date for each day boundary
            let currentDay = new Date(startDt);
            currentDay.setDate(currentDay.getDate() + 1); // Start with the day after start
            
            while (currentDay < endDt) {
                markLines.push({
                    xAxis: currentDay.toISOString(),
                    lineStyle: {
                        color: '#666',
                        width: 1,
                        type: 'solid'
                    },
                    label: {
                        formatter: function(params) {
                            // Format date as MM/DD
                            const date = new Date(params.value);
                            return (date.getMonth() + 1) + '/' + date.getDate();
                        },
                        position: 'insideTopLeft',
                        distance: 5,
                        fontSize: 10,
                        fontWeight: 'bold'
                    }
                });
                
                // Move to next day
                currentDay.setDate(currentDay.getDate() + 1);
            }
            
            // Add markLines with global z-index higher than the scatter points
            if (markLines.length > 0) {
                option.series.push({
                    type: 'line',
                    showSymbol: false,
                    data: [],
                    markLine: {
                        silent: true,
                        symbol: 'none',
                        lineStyle: {
                            type: 'solid'
                        },
                        data: markLines,
                        zlevel: 1
                    }
                });
            }
        }
        
        // Apply options to chart
        chart.setOption(option);
        
        // Track highlighted FLWs
        if (!window.highlightedFLWs) {
            window.highlightedFLWs = new Set();
        }
        
        // Create a separate container for our highlights to avoid z-index issues
        const highlightContainer = document.createElement('div');
        highlightContainer.className = 'highlight-container';
        highlightContainer.style.position = 'absolute';
        highlightContainer.style.top = '0';
        highlightContainer.style.left = '0';
        highlightContainer.style.width = '100%';
        highlightContainer.style.height = '100%';
        highlightContainer.style.pointerEvents = 'none';
        highlightContainer.style.zIndex = '1';
        chartElement.style.position = 'relative';
        chartElement.appendChild(highlightContainer);
        
        // Add custom handler for axis label clicks
        // Using direct DOM event since ECharts doesn't provide built-in axis label click events
        chartElement.addEventListener('click', function(e) {
            // Get chart's position relative to the window
            const chartRect = chartElement.getBoundingClientRect();
            
            // Calculate click position relative to chart
            const x = e.clientX - chartRect.left;
            const y = e.clientY - chartRect.top;
            
            // Only process clicks near the y-axis labels (left side of chart)
            if (x < chartRect.width * 0.15) { // Adjust this value as needed
                // We need to manually find which FLW label was clicked
                // First get all text elements in the chart
                const textElements = Array.from(chartElement.querySelectorAll('text'));
                
                // Filter to likely y-axis labels (those on the left side)
                const yAxisLabels = textElements.filter(el => {
                    const rect = el.getBoundingClientRect();
                    return (rect.left - chartRect.left) < chartRect.width * 0.15;
                });
                
                // Find the clicked label
                let clickedFLW = null;
                yAxisLabels.forEach(label => {
                    const labelRect = label.getBoundingClientRect();
                    if (e.clientY >= labelRect.top && e.clientY <= labelRect.bottom) {
                        // Check if this label text matches an FLW name
                        const labelText = label.textContent;
                        if (selectedFLWs.includes(labelText)) {
                            clickedFLW = labelText;
                        }
                    }
                });
                
                // If we found a matching FLW, toggle its highlight
                if (clickedFLW) {
                    console.log('Clicked on FLW:', clickedFLW);
                    
                    // Toggle highlight
                    if (window.highlightedFLWs.has(clickedFLW)) {
                        window.highlightedFLWs.delete(clickedFLW);
                    } else {
                        window.highlightedFLWs.add(clickedFLW);
                    }
                    
                    // Update the highlights
                    updateHighlighting();
                }
            }
        });
        
        // Function to update the row highlights
        function updateHighlighting() {
            // Clear existing highlights
            highlightContainer.innerHTML = '';
            
            // Get the grid position and dimensions
            const gridComponent = chart.getModel().getComponent('grid');
            const grid = gridComponent.coordinateSystem.getRect();
            
            // Add highlight for each selected FLW
            window.highlightedFLWs.forEach(flwName => {
                const flwIndex = selectedFLWs.indexOf(flwName);
                if (flwIndex >= 0) {
                    // Convert index to pixel position
                    const yPos = chart.convertToPixel({yAxisIndex: 0}, flwIndex);
                    
                    // Create highlight element
                    const highlightElem = document.createElement('div');
                    highlightElem.className = 'flw-row-highlight';
                    highlightElem.style.position = 'absolute';
                    highlightElem.style.left = `${grid.x}px`;
                    highlightElem.style.width = `${grid.width}px`;
                    highlightElem.style.top = `${yPos - 15}px`; // Center on the row
                    highlightElem.style.height = '30px'; // Approximate row height
                    
                    // Add a label showing which FLW is highlighted
                    const label = document.createElement('div');
                    label.className = 'highlight-label';
                    label.style.position = 'absolute';
                    label.style.right = '10px';
                    label.style.top = '5px';
                    label.style.padding = '2px 5px';
                    label.style.background = 'rgba(255, 255, 0, 0.7)';
                    label.style.borderRadius = '3px';
                    label.style.fontSize = '11px';
                    label.textContent = flwName;
                    highlightElem.appendChild(label);
                    
                    // Add to container
                    highlightContainer.appendChild(highlightElem);
                    
                    // Log for debugging
                    console.log('Highlighted:', flwName, 'at y-position:', yPos);
                }
            });
        }
        
        // Call updateHighlighting initially
        updateHighlighting();
    }
    
    // Add event listener for the update button
    document.getElementById('update-chart').addEventListener('click', updateChart);
    
    // Initialize chart on load
    updateChart();
    
    // Handle window resize
    window.addEventListener('resize', function() {
        chart.resize();
    });
});
"""

    # Create JavaScript file
    js_filename = "flw_timeline.js"
    with open(js_filename, "w") as f:
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
            h1 {{
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
            
            <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    output_filename = "flw_views.html"
    with open(output_filename, "w") as f:
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