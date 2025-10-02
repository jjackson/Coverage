"""
Visit Time Period Report

Groups visits by time periods (weekly or monthly) and creates interactive maps
showing visits colored by their time period for each OPP ID.
"""

import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from .base_report import BaseReport


class VisitTimePeriodReport(BaseReport):
    """Report that groups visits by time periods and creates temporal maps"""
    
    @staticmethod
    def setup_parameters(parent_frame):
        """Set up parameters for visit time period report"""
        
        # Time period selection
        ttk.Label(parent_frame, text="Time period:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        time_period_var = tk.StringVar(value="Week")
        period_frame = ttk.Frame(parent_frame)
        period_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(period_frame, text="Weekly", variable=time_period_var, value="Week").pack(side=tk.LEFT)
        ttk.Radiobutton(period_frame, text="Monthly", variable=time_period_var, value="Month").pack(side=tk.LEFT, padx=(10,0))
        
        # Store variables
        parent_frame.time_period_var = time_period_var
        
    def generate(self):
        """Generate visit time period reports for all OPP IDs"""
        output_files = []
        
        # Get parameters
        time_period = self.get_parameter_value('time_period', 'Week')
        
        self.log(f"Starting visit time period analysis for all OPP IDs")
        self.log(f"Time period: {time_period}")
        
        # Create output directory with today's date
        today = datetime.now().strftime('%Y_%m_%d')
        period_dir = os.path.join(self.output_dir, f"visit_timeline_{today}")
        os.makedirs(period_dir, exist_ok=True)
        self.log(f"Created output directory: {os.path.basename(period_dir)}")
        
        # Validate and prepare data
        visits_data = self._prepare_visit_data()
        
        if len(visits_data) == 0:
            raise ValueError("No valid visit data found")
        
        # Group visits by time periods
        grouped_data = self._group_by_time_period(visits_data, time_period)
        
        # Get unique OPP IDs
        opp_ids = grouped_data['opp_id'].unique()
        self.log(f"Found {len(opp_ids)} unique OPP IDs: {list(opp_ids)}")
        
        # Collect summary data
        all_summary_data = []
        all_details_data = []
        
        # Process each OPP ID
        for current_opp_id in opp_ids:
            self.log(f"Processing OPP ID: {current_opp_id}")
            
            # Filter data for this OPP ID
            opp_data = grouped_data[grouped_data['opp_id'] == current_opp_id].copy()
            
            if len(opp_data) == 0:
                self.log(f"  Skipping {current_opp_id}: no visits found")
                continue
                
            self.log(f"  Found {len(opp_data)} visits across {len(opp_data['time_period'].unique())} time periods")
            
            # Get opportunity name for file naming
            opp_name_prefix = self._get_opportunity_name_prefix(opp_data, current_opp_id)
            
            # Generate map for this OPP ID
            map_file = self._create_time_period_map(opp_data, current_opp_id, period_dir, time_period, opp_name_prefix)
            if map_file:
                output_files.append(map_file)
                self.log(f"  Created: {os.path.basename(map_file)}")
            
            # Collect summary data for this OPP
            opp_summary = self._extract_opp_summary(opp_data, current_opp_id, opp_name_prefix, time_period)
            all_summary_data.extend(opp_summary)
            
            # Collect details data
            opp_details = opp_data.copy()
            opp_details['opp_name'] = opp_name_prefix
            all_details_data.append(opp_details)
        
        # Create summary files
        if all_summary_data:
            summary_file = self._save_summary_file(all_summary_data, period_dir, time_period)
            if summary_file:
                output_files.append(summary_file)
        
        if all_details_data:
            details_file = self._save_details_file(all_details_data, period_dir, time_period)
            if details_file:
                output_files.append(details_file)
        
        self.log(f"Visit time period analysis complete! Generated {len(output_files)} files in {os.path.basename(period_dir)}")
        
        return output_files
    
    def _prepare_visit_data(self):
        """Prepare and validate visit data"""
        
        # Make a copy of the data
        data = self.df.copy()
        
        # Standardize column names (case-insensitive)
        data.columns = data.columns.str.lower().str.strip()
        
        # Check for required columns
        required_cols = ['opp_id', 'latitude', 'longitude']
        missing_cols = []
        
        # Try different possible column name variations
        col_variations = {
            'opp_id': ['opp_id', 'oppid', 'opportunity_id', 'campaign_id'],
            'latitude': ['latitude', 'lat', 'y'],
            'longitude': ['longitude', 'lon', 'lng', 'x']
        }
        
        final_cols = {}
        for req_col in required_cols:
            found = False
            for variation in col_variations[req_col]:
                if variation in data.columns:
                    final_cols[req_col] = variation
                    found = True
                    break
            if not found:
                missing_cols.append(req_col)
        
        if missing_cols:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {available_cols}")
        
        # Rename columns to standard names
        data = data.rename(columns={v: k for k, v in final_cols.items()})
        
        # Find date column
        date_col = self._find_date_column(data)
        if not date_col:
            raise ValueError("No date/time column found. Looking for columns containing: date, time, visit, created, updated")
        
        self.log(f"Using date column: {date_col}")
        self.log(f"Processing {len(data)} total visit records")
        
        # Clean data
        before_clean = len(data)
        data = data.dropna(subset=['latitude', 'longitude', date_col])
        after_clean = len(data)
        if before_clean != after_clean:
            self.log(f"Removed {before_clean - after_clean} records with missing coordinates or dates")
        
        # Validate coordinate ranges
        invalid_coords = (
            (data['latitude'] < -90) | (data['latitude'] > 90) |
            (data['longitude'] < -180) | (data['longitude'] > 180)
        )
        if invalid_coords.any():
            invalid_count = invalid_coords.sum()
            self.log(f"Warning: Found {invalid_count} records with invalid coordinates")
            data = data[~invalid_coords]
        
        # Add unique visit ID if not present
        if 'visit_id' not in data.columns:
            data['visit_id'] = [f"visit_{i+1}" for i in range(len(data))]
        
        # Store the date column name for later use
        data['visit_date'] = data[date_col]
        
        return data.reset_index(drop=True)
    
    def _find_date_column(self, data):
        """Find the most likely date column"""
        
        # First priority: look for exact 'visit_date' column
        if 'visit_date' in data.columns:
            return 'visit_date'
        
        date_patterns = ['visit_date', 'date', 'time', 'created', 'updated', 'timestamp']
        
        # Look for columns that might contain dates
        date_candidates = []
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in date_patterns):
                # Skip visit_id specifically - it's not a date
                if 'visit_id' in col_lower.replace('_', '').replace(' ', ''):
                    continue
                date_candidates.append(col)
        
        if not date_candidates:
            return None
        
        # Test each candidate to see if it contains parseable dates
        for col in date_candidates:
            try:
                # Try to parse a few sample values
                sample = data[col].dropna().head(10)
                if len(sample) > 0:
                    # Try parsing with pandas
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() >= len(sample) * 0.8:  # At least 80% parseable
                        return col
            except:
                continue
        
        # If no good candidate found, return the first one
        return date_candidates[0] if date_candidates else None
    
    def _group_by_time_period(self, data, time_period):
        """Group visits by time period and assign colors"""
        
        self.log(f"Grouping visits by {time_period.lower()}...")
        
        # Use visit_date directly - it should already be parsed as datetime
        # (following the same pattern as basic_analyzer.py and other working reports)
        try:
            # Check if visit_date is already datetime, if not convert it
            if not pd.api.types.is_datetime64_any_dtype(data['visit_date']):
                self.log("Converting visit_date to datetime...")
                data['parsed_date'] = pd.to_datetime(data['visit_date'], errors='coerce')
            else:
                self.log("Using existing datetime visit_date column...")
                data['parsed_date'] = data['visit_date']
                
        except Exception as e:
            raise ValueError(f"Could not use visit_date column: {str(e)}")
        
        # Log sample of parsed dates for debugging
        valid_dates = data['parsed_date'].dropna()
        if len(valid_dates) > 0:
            self.log(f"Sample parsed dates: {valid_dates.head(3).dt.strftime('%Y-%m-%d').tolist()}")
            self.log(f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
        else:
            # Show sample raw values for debugging
            sample_raw = data['visit_date'].dropna().head(5).tolist()
            self.log(f"Failed to parse any dates. Sample raw values: {sample_raw}")
        
        # Remove rows with unparseable dates
        before_parse = len(data)
        data = data.dropna(subset=['parsed_date'])
        after_parse = len(data)
        if before_parse != after_parse:
            self.log(f"Removed {before_parse - after_parse} records with unparseable dates")
        
        if len(data) == 0:
            raise ValueError("No visits with valid dates found")
        
        # Group by time period
        if time_period == "Week":
            # Use Monday as week start (ISO week)
            data['time_period'] = data['parsed_date'].dt.to_period('W-MON').astype(str)
            data['time_period_label'] = data['parsed_date'].dt.strftime('%Y-W%U')  # Year-Week format
        else:  # Month
            data['time_period'] = data['parsed_date'].dt.to_period('M').astype(str)
            data['time_period_label'] = data['parsed_date'].dt.strftime('%Y-%m')  # Year-Month format
        
        # Get unique time periods and assign colors
        unique_periods = sorted(data['time_period'].unique())
        period_colors = self._generate_period_colors(len(unique_periods))
        color_map = dict(zip(unique_periods, period_colors))
        
        # Add colors to data
        data['period_color'] = data['time_period'].map(color_map)
        
        self.log(f"Found {len(unique_periods)} unique time periods: {unique_periods[:5]}..." if len(unique_periods) > 5 else f"Found {len(unique_periods)} unique time periods: {unique_periods}")
        
        return data
    
    def _get_opportunity_name_prefix(self, opp_data, opp_id):
        """Extract the first word of opportunity name for file naming"""
        
        # Look for opportunity name columns
        name_columns = [col for col in opp_data.columns if 'opportunity' in col.lower() and 'name' in col.lower()]
        if not name_columns:
            name_columns = [col for col in opp_data.columns if 'opp' in col.lower() and 'name' in col.lower()]
        if not name_columns:
            name_columns = [col for col in opp_data.columns if 'campaign' in col.lower() and 'name' in col.lower()]
        
        # If we found a name column, extract the first word
        if name_columns:
            name_col = name_columns[0]
            # Get the first non-null opportunity name
            opp_names = opp_data[name_col].dropna()
            if len(opp_names) > 0:
                full_name = str(opp_names.iloc[0]).strip()
                # Get first word (until first space)
                first_word = full_name.split()[0] if full_name else f"opp{opp_id}"
                # Clean the first word for use in filename
                first_word = ''.join(c for c in first_word if c.isalnum() or c in '-_')
                return first_word
        
        # Fallback if no opportunity name found
        return f"opp{opp_id}"
    
    def _create_time_period_map(self, opp_data, opp_id, period_dir, time_period, opp_name_prefix):
        """Create interactive Leaflet map showing visits colored by time period"""
        
        # Prepare data for map
        map_data = []
        for _, row in opp_data.iterrows():
            map_data.append({
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'time_period': str(row['time_period']),
                'time_period_label': str(row['time_period_label']),
                'visit_id': str(row.get('visit_id', 'Unknown')),
                'visit_date': str(row['parsed_date'].date()) if pd.notna(row['parsed_date']) else 'Unknown',
                'color': str(row['period_color'])
            })
        
        # Get unique time periods and their colors
        unique_periods = opp_data['time_period'].unique()
        period_colors = {}
        period_counts = {}
        for period in unique_periods:
            period_data = opp_data[opp_data['time_period'] == period]
            period_colors[period] = period_data['period_color'].iloc[0]
            period_counts[period] = len(period_data)
        
        # Calculate map center
        center_lat = opp_data['latitude'].mean()
        center_lon = opp_data['longitude'].mean()
        
        # Generate HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Visit Timeline - OPP {opp_id}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS and JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, Helvetica, sans-serif;
        }}
        #map {{
            width: 100%;
            height: 85vh;
        }}
        .control-panel {{
            background: white;
            padding: 10px;
            height: 15vh;
            border-bottom: 1px solid #ccc;
            overflow-y: auto;
        }}
        .control-row {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 10px;
        }}
        .period-toggles {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            max-height: 60px;
            overflow-y: auto;
        }}
        .period-toggle {{
            display: inline-block;
            cursor: pointer;
            padding: 3px 8px;
            border-radius: 3px;
            border: 1px solid #ddd;
            background: #f9f9f9;
            font-size: 12px;
        }}
        .period-toggle input {{
            margin-right: 3px;
        }}
        .color-swatch {{
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 5px;
            border: 1px solid #999;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div class="control-panel">
        <div class="control-row">
            <div>
                <strong>OPP ID:</strong> {opp_id} | 
                <strong>Period:</strong> {time_period} | 
                <strong>Total visits:</strong> {len(opp_data)} | 
                <strong>Time periods:</strong> {len(unique_periods)}
            </div>
            <div>
                <button onclick="toggleAllPeriods(true)">Show All</button>
                <button onclick="toggleAllPeriods(false)">Hide All</button>
            </div>
        </div>
        <div class="control-row">
            <strong>Toggle Time Periods:</strong>
            <div class="period-toggles" id="period-toggles">
                <!-- Period toggles will be added here -->
            </div>
        </div>
    </div>
    
    <div id="map"></div>
    
    <script>
        // Map data
        const visitData = {json.dumps(map_data)};
        const periodColors = {json.dumps(period_colors)};
        const periodCounts = {json.dumps(period_counts)};
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 13);
        
        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);
        
        // Create layer groups for each time period
        const periodLayers = {{}};
        
        // Group visits by time period
        visitData.forEach(visit => {{
            const period = visit.time_period;
            
            if (!periodLayers[period]) {{
                periodLayers[period] = L.layerGroup().addTo(map);
            }}
            
            // Create marker
            const marker = L.circleMarker([visit.latitude, visit.longitude], {{
                radius: 6,
                fillColor: visit.color,
                color: '#fff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }});
            
            // Add popup
            marker.bindPopup(`
                <strong>Visit ID:</strong> ${{visit.visit_id}}<br>
                <strong>Date:</strong> ${{visit.visit_date}}<br>
                <strong>Time Period:</strong> ${{visit.time_period_label}}<br>
                <strong>Coordinates:</strong> ${{visit.latitude.toFixed(6)}}, ${{visit.longitude.toFixed(6)}}
            `);
            
            periodLayers[period].addLayer(marker);
        }});
        
        // Create period toggles
        const togglesContainer = document.getElementById('period-toggles');
        Object.keys(periodColors).sort().forEach(period => {{
            const label = document.createElement('label');
            label.className = 'period-toggle';
            
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = true;
            input.addEventListener('change', function() {{
                if (this.checked) {{
                    periodLayers[period].addTo(map);
                }} else {{
                    map.removeLayer(periodLayers[period]);
                }}
            }});
            
            const colorSwatch = document.createElement('span');
            colorSwatch.className = 'color-swatch';
            colorSwatch.style.backgroundColor = periodColors[period];
            
            label.appendChild(input);
            label.appendChild(colorSwatch);
            label.appendChild(document.createTextNode(`${{period}} (${{periodCounts[period]}})`));
            
            togglesContainer.appendChild(label);
        }});
        
        // Toggle all periods function
        function toggleAllPeriods(show) {{
            document.querySelectorAll('.period-toggle input').forEach(input => {{
                input.checked = show;
                const event = new Event('change');
                input.dispatchEvent(event);
            }});
        }}
        
        // Fit map to all visits
        if (visitData.length > 0) {{
            const group = new L.featureGroup(Object.values(periodLayers));
            map.fitBounds(group.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>"""
        
        # Save HTML file
        period_type_lower = time_period.lower()
        output_filename = os.path.join(period_dir, f"visit_timeline_map_{period_type_lower}_{opp_name_prefix}_{opp_id}.html")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_filename
    
    def _generate_period_colors(self, n_periods):
        """Generate distinct colors for time periods"""
        
        # Use a color palette that works well for temporal data
        base_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
            "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d3", "#c7c7c7",
            "#dbdb8d", "#9edae5"
        ]
        
        if n_periods <= len(base_colors):
            return base_colors[:n_periods]
        else:
            # Generate additional colors using HSV space for better distribution
            import colorsys
            additional_colors = []
            for i in range(n_periods - len(base_colors)):
                hue = (i * 0.618033988749895) % 1  # Golden ratio for good distribution
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
                additional_colors.append(hex_color)
            return base_colors + additional_colors
    
    def _extract_opp_summary(self, opp_data, opp_id, opp_name_prefix, time_period):
        """Extract summary data for this OPP"""
        
        summary_data = []
        
        # Get summary for each time period
        for period in opp_data['time_period'].unique():
            period_data = opp_data[opp_data['time_period'] == period]
            period_label = period_data['time_period_label'].iloc[0]
            
            # Calculate some basic stats
            visit_count = len(period_data)
            date_range = f"{period_data['parsed_date'].min().date()} to {period_data['parsed_date'].max().date()}"
            
            summary_data.append({
                'opp_id': opp_id,
                'opp_name': opp_name_prefix,
                'time_period_type': time_period,
                'time_period': period,
                'time_period_label': period_label,
                'visit_count': visit_count,
                'date_range': date_range,
                'first_visit_date': period_data['parsed_date'].min().date(),
                'last_visit_date': period_data['parsed_date'].max().date()
            })
        
        return summary_data
    
    def _save_summary_file(self, all_summary_data, period_dir, time_period):
        """Save summary CSV file"""
        
        try:
            summary_df = pd.DataFrame(all_summary_data)
            summary_df = summary_df.sort_values(['opp_id', 'time_period'])
            
            period_type_lower = time_period.lower()
            summary_file = os.path.join(period_dir, f"visit_timeline_summary_{period_type_lower}.csv")
            summary_df.to_csv(summary_file, index=False)
            
            self.log(f"Created summary file: {os.path.basename(summary_file)} ({len(summary_df)} records)")
            return summary_file
            
        except Exception as e:
            self.log(f"Error saving summary file: {str(e)}")
            return None
    
    def _save_details_file(self, all_details_data, period_dir, time_period):
        """Save details CSV file with all visits"""
        
        try:
            details_df = pd.concat(all_details_data, ignore_index=True)
            
            # Select relevant columns for the details file
            detail_columns = [
                'opp_id', 'opp_name', 'visit_id', 'latitude', 'longitude',
                'visit_date', 'parsed_date', 'time_period', 'time_period_label',
                'period_color'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in detail_columns if col in details_df.columns]
            details_df = details_df[available_columns]
            
            details_df = details_df.sort_values(['opp_id', 'parsed_date'])
            
            period_type_lower = time_period.lower()
            details_file = os.path.join(period_dir, f"visit_timeline_details_{period_type_lower}.csv")
            details_df.to_csv(details_file, index=False)
            
            self.log(f"Created details file: {os.path.basename(details_file)} ({len(details_df)} visits)")
            return details_file
            
        except Exception as e:
            self.log(f"Error saving details file: {str(e)}")
            return None
"""
Time Analysis - calculates visit duration and timing metrics
Handles form_start_time and form_end_time as time-only data (no midnight crossover)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta


class TimeAnalyzer:
    def __init__(self, df, flw_id_col, log_func):
        self.df = df
        self.flw_id_col = flw_id_col
        self.log = log_func
    
    def analyze(self):
        """Generate time-based analysis for all FLWs"""
        
        # Check if required columns exist
        if not self._has_required_columns():
            self.log("Warning: Missing form_start_time or form_end_time columns - skipping time analysis")
            return None
        
        try:
            # Prepare time data (much simpler now)
            time_df = self._prepare_time_data()
            
            if len(time_df) == 0:
                self.log("No time data found")
                return None
            
            # Log timing data quality
            total_visits = len(time_df)
            visits_with_issues = time_df['has_timing_issues'].sum()
            valid_visits = total_visits - visits_with_issues
            
            self.log(f"Time data: {total_visits} total visits, {valid_visits} with valid timing, {visits_with_issues} with issues")
            
            if valid_visits == 0:
                self.log("No visits with valid timing data")
                return None
            
            # Analyze each FLW
            flw_list = time_df[self.flw_id_col].unique()
            flw_list = flw_list[pd.notna(flw_list)]
            
            results = []
            
            for flw_id in flw_list:
                try:
                    flw_data = time_df[time_df[self.flw_id_col] == flw_id].copy()
                    flw_metrics = self._analyze_flw_timing(flw_id, flw_data)
                    results.append(flw_metrics)
                except Exception as e:
                    self.log(f"Warning: Could not analyze FLW {flw_id}: {str(e)}")
                    continue
            
            if not results:
                self.log("No FLW time analysis results generated")
                return None
            
            results_df = pd.DataFrame(results)
            self.log(f"Time analysis complete: {len(results_df)} FLWs analyzed")
            return results_df
            
        except Exception as e:
            self.log(f"Error in time analysis: {str(e)}")
            return None
    
    def _has_required_columns(self):
        """Check if required time columns exist"""
        required_cols = ['form_start_time', 'form_end_time']
        has_cols = all(col in self.df.columns for col in required_cols)
        if has_cols:
            self.log(f"Found time columns: {required_cols}")
        else:
            missing = [col for col in required_cols if col not in self.df.columns]
            self.log(f"Missing time columns: {missing}")
        return has_cols
    
    def _prepare_time_data(self):
        """Prepare and clean time data - simplified approach"""
        time_df = self.df.copy()
        
        # Identify timing issues
        time_df['has_timing_issues'] = self._identify_timing_issues(time_df)
        
        # Parse time strings and calculate duration
        time_df['visit_duration_minutes'] = self._calculate_visit_duration(time_df)
        
        return time_df
    
    def _identify_timing_issues(self, df):
        """Identify visits with timing issues - simplified for time-only data"""
        issues = pd.Series(False, index=df.index)
        
        # Convert to string and check for missing/empty values
        start_strings = df['form_start_time'].astype(str).str.strip()
        end_strings = df['form_end_time'].astype(str).str.strip()
        
        # Check for missing/empty/invalid values
        invalid_values = {'', 'nan', 'None', 'NaT', 'nat'}
        
        missing_start = (
            df['form_start_time'].isna() | 
            start_strings.isin(invalid_values)
        )
        
        missing_end = (
            df['form_end_time'].isna() | 
            end_strings.isin(invalid_values)
        )
        
        issues |= missing_start | missing_end
        
        # For non-missing times, try to parse and check if start > end
        valid_mask = ~missing_start & ~missing_end
        
        if valid_mask.any():
            try:
                # Parse time strings
                start_times = pd.to_datetime(start_strings[valid_mask], format='%H:%M:%S', errors='coerce')
                end_times = pd.to_datetime(end_strings[valid_mask], format='%H:%M:%S', errors='coerce')
                
                # Mark failed parsing as issues
                parse_failed = start_times.isna() | end_times.isna()
                issues.loc[valid_mask] |= parse_failed
                
                # Check for start > end (no midnight crossover assumed)
                successfully_parsed = valid_mask & ~issues
                if successfully_parsed.any():
                    start_subset = pd.to_datetime(start_strings[successfully_parsed], format='%H:%M:%S')
                    end_subset = pd.to_datetime(end_strings[successfully_parsed], format='%H:%M:%S')
                    invalid_order = start_subset > end_subset
                    issues.loc[successfully_parsed] |= invalid_order
                
            except Exception as e:
                self.log(f"Warning: Could not validate time format: {str(e)}")
                # Mark questionable entries as having issues
                issues.loc[valid_mask] = True
        
        return issues
    
    def _calculate_visit_duration(self, df):
        """Calculate visit duration in minutes - simple time arithmetic"""
        duration = pd.Series(np.nan, index=df.index)
        
        # Only process visits without timing issues
        valid_mask = ~df['has_timing_issues']
        
        if not valid_mask.any():
            return duration
        
        try:
            # Get clean time strings
            start_strings = df.loc[valid_mask, 'form_start_time'].astype(str).str.strip()
            end_strings = df.loc[valid_mask, 'form_end_time'].astype(str).str.strip()
            
            # Parse as datetime objects (we only care about the time part)
            start_times = pd.to_datetime(start_strings, format='%H:%M:%S', errors='coerce')
            end_times = pd.to_datetime(end_strings, format='%H:%M:%S', errors='coerce')
            
            # Calculate duration in minutes
            time_diff = end_times - start_times
            duration_minutes = time_diff.dt.total_seconds() / 60
            
            # Only keep positive durations (should be guaranteed by our validation)
            duration.loc[valid_mask] = duration_minutes
            
        except Exception as e:
            self.log(f"Warning: Could not calculate visit duration: {str(e)}")
        
        return duration
    
    def _analyze_flw_timing(self, flw_id, flw_data):
        """Analyze timing metrics for a single FLW"""
        
        total_visits = len(flw_data)
        timing_issues = flw_data['has_timing_issues'].sum()
        
        result = {
            self.flw_id_col: flw_id,
            'percent_visits_with_timing_issues': round(timing_issues / total_visits, 3) if total_visits > 0 else 0
        }
        
        # Duration metrics (only for visits without timing issues)
        valid_durations = flw_data['visit_duration_minutes'].dropna()
        
        if len(valid_durations) > 0:
            result.update({
                'avg_visit_duration_minutes': round(valid_durations.mean(), 1),
                'median_visit_duration_minutes': round(valid_durations.median(), 1),
                'min_visit_duration_minutes': round(valid_durations.min(), 1),
                'max_visit_duration_minutes': round(valid_durations.max(), 1)
            })
        else:
            result.update({
                'avg_visit_duration_minutes': None,
                'median_visit_duration_minutes': None,
                'min_visit_duration_minutes': None,
                'max_visit_duration_minutes': None
            })
        
        # Time between consecutive visits
        avg_gap = self._calculate_avg_time_between_visits(flw_data)
        result['avg_minutes_between_consecutive_visits'] = avg_gap
        
        return result
    
    def _calculate_avg_time_between_visits(self, flw_data):
        """Calculate average time between consecutive visits on the same day for this FLW"""
        
        # Filter to visits with valid timing and duration data
        valid_visits = flw_data[
            ~flw_data['has_timing_issues'] & 
            flw_data['visit_duration_minutes'].notna()
        ].copy()
        
        if len(valid_visits) < 2:
            return None
        
        try:
            # Add visit_day column for grouping
            valid_visits['visit_day'] = pd.to_datetime(valid_visits['visit_date']).dt.date
            
            # Parse times for each visit
            start_time_strings = valid_visits['form_start_time'].astype(str).str.strip()
            end_time_strings = valid_visits['form_end_time'].astype(str).str.strip()
            
            start_times = pd.to_datetime(start_time_strings, format='%H:%M:%S', errors='coerce')
            end_times = pd.to_datetime(end_time_strings, format='%H:%M:%S', errors='coerce')
            
            valid_visits = valid_visits.assign(
                start_time_parsed=start_times,
                end_time_parsed=end_times
            )
            
            # Calculate gaps within each day
            all_gaps = []
            
            for visit_day, day_data in valid_visits.groupby('visit_day'):
                if len(day_data) < 2:
                    continue  # Need at least 2 visits on same day
                
                # Sort visits by start time within this day
                day_data = day_data.sort_values('start_time_parsed').dropna(subset=['start_time_parsed', 'end_time_parsed'])
                
                if len(day_data) < 2:
                    continue
                
                # Calculate gaps between consecutive visits on this day
                for i in range(len(day_data) - 1):
                    current_end = day_data.iloc[i]['end_time_parsed']
                    next_start = day_data.iloc[i + 1]['start_time_parsed']
                    
                    # Calculate gap in minutes (same day, so just time difference)
                    gap_minutes = (next_start - current_end).total_seconds() / 60
                    
                    # Only include reasonable gaps (positive and less than 12 hours)
                    if 0 <= gap_minutes <= 720:  # 12 hours = 720 minutes
                        all_gaps.append(gap_minutes)
            
            if all_gaps:
                return round(np.mean(all_gaps), 1)
            else:
                return None
                
        except Exception as e:
            self.log(f"Warning: Could not calculate same-day time between visits for FLW {flw_data[self.flw_id_col].iloc[0]}: {str(e)}")
            return None
    
    def get_timing_quality_summary(self, time_results_df):
        """Generate a summary of timing data quality across all FLWs"""
        
        if time_results_df is None or len(time_results_df) == 0:
            return None
        
        timing_issues = time_results_df['percent_visits_with_timing_issues']
        
        summary = {
            'total_flws_analyzed': len(time_results_df),
            'flws_with_no_timing_issues': (timing_issues == 0).sum(),
            'flws_with_some_timing_issues': (timing_issues > 0).sum(),
            'avg_percent_timing_issues': round(timing_issues.mean(), 3),
            'median_percent_timing_issues': round(timing_issues.median(), 3),
            'max_percent_timing_issues': round(timing_issues.max(), 3)
        }
        
        return summary

