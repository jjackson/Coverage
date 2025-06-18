#!/usr/bin/env python3
"""
Exploratory Statistics Generator

This module generates HTML output with various tables and statistics
based on KMC visit data loaded from Superset.
"""

import pandas as pd
import os
import sys
import math
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import webbrowser

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import export_superset_query_with_pagination
from src.sqlqueries.sql_queries import SQL_QUERIES
import constants


def load_kmc_visit_data(superset_url: str, username: str, password: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load KMC visit data from Superset using the kmc_visit_query.
    
    Args:
        superset_url: Base URL of the Superset instance
        username: Superset username for authentication
        password: Superset password for authentication
        verbose: Whether to show detailed progress output
        
    Returns:
        DataFrame with KMC visit data
    """
    print("Loading KMC visit data from Superset...")
    
    # Get the SQL query
    sql_query = SQL_QUERIES["kmc_visit_query"]
    
    # Export data from Superset
    csv_file_path = export_superset_query_with_pagination(
        superset_url=superset_url,
        sql_query=sql_query,
        username=username,
        password=password,
        verbose=verbose
    )
    
    # Load the exported CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    print(f"Successfully loaded {len(df):,} rows of KMC visit data")
    print(f"Columns: {list(df.columns)}")
    
    return df


def create_html_table(df: pd.DataFrame, title: str = "Data Table") -> str:
    """
    Create an HTML table from a DataFrame.
    
    Args:
        df: DataFrame to convert to HTML table
        title: Title for the table section
        
    Returns:
        HTML string with the table
    """
    html = f"""
    <div class="table-section">
        <h2>{title}</h2>
        <div class="table-container">
            {df.to_html(classes=['data-table', 'table', 'table-striped', 'table-hover'], 
                       index=False, 
                       float_format='%.2f')}
        </div>
    </div>
    """
    return html


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates using the Haversine formula.
    
    Args:
        lat1, lon1: First coordinate pair
        lat2, lon2: Second coordinate pair
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r


def create_case_visit_table(df: pd.DataFrame) -> str:
    """
    Create a table showing visits per case_id with daily columns and distance calculations.
    
    Args:
        df: DataFrame with KMC visit data
        
    Returns:
        HTML string with the case visit table
    """
    print("Creating case-based visit table...")
    
    # Ensure we have the required columns
    required_cols = ['case_id', 'visit_date', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return "<p>Error: Missing required columns for case visit table</p>"
    
    # Convert visit_date to datetime if it's not already
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    
    # Convert latitude and longitude to numeric, handling any non-numeric values
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Remove rows with missing coordinates
    before_count = len(df)
    df = df.dropna(subset=['latitude', 'longitude', 'case_id'])
    after_count = len(df)
    if before_count != after_count:
        print(f"Removed {before_count - after_count} rows with missing coordinates or case_id")
    
    # Get the date range
    min_date = df['visit_date'].min().date()
    max_date = df['visit_date'].max().date()
    
    # Create date range for all days
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Group by case_id and sort visits by date
    case_visits = df.groupby('case_id').apply(
        lambda x: x.sort_values('visit_date').reset_index(drop=True)
    ).reset_index(drop=True)
    
    # Create the case visit table
    case_table_data = []
    
    for case_id in case_visits['case_id'].unique():
        case_data = case_visits[case_visits['case_id'] == case_id].copy()
        
        # Get the most common values for additional columns
        opportunity_name = case_data['opportunity_name'].mode().iloc[0] if 'opportunity_name' in case_data.columns and len(case_data['opportunity_name'].mode()) > 0 else ''
        flw_id = case_data['flw_id'].mode().iloc[0] if 'flw_id' in case_data.columns and len(case_data['flw_id'].mode()) > 0 else ''
        flw_name = case_data['flw_name'].mode().iloc[0] if 'flw_name' in case_data.columns and len(case_data['flw_name'].mode()) > 0 else ''
        
        # Create row for this case
        row = {
            'case_id': case_id,
            'opportunity_name': opportunity_name,
            'flw_id': flw_id,
            'flw_name': flw_name
        }
        
        # Add daily visit indicators
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            has_visit = case_data['visit_date'].dt.date.eq(date.date()).any()
            row[date_str] = 'X' if has_visit else ''
        
        # Calculate distances between consecutive visits
        if len(case_data) > 1:
            for i in range(len(case_data) - 1):
                visit1 = case_data.iloc[i]
                visit2 = case_data.iloc[i + 1]
                
                try:
                    distance = calculate_distance(
                        visit1['latitude'], visit1['longitude'],
                        visit2['latitude'], visit2['longitude']
                    )
                    row[f'Distance {i+1}-{i+2}'] = f"{distance:.2f} km"
                except (ValueError, TypeError):
                    row[f'Distance {i+1}-{i+2}'] = 'N/A'
        else:
            # Single visit - no distances to calculate
            pass
        
        # Fill in missing distance columns (up to 5)
        for i in range(len(case_data) - 1, 5):
            row[f'Distance {i+1}-{i+2}'] = ''
        
        case_table_data.append(row)
    
    # Create DataFrame and sort by case_id
    case_df = pd.DataFrame(case_table_data)
    if len(case_df) > 0:
        case_df = case_df.sort_values('case_id')
    
    # Create HTML table
    html = f"""
    <div class="table-section">
        <h2>Case Visit Timeline</h2>
        <p>Showing visits per case_id from {min_date} to {max_date}. 'X' indicates a visit on that date.</p>
        <p>Distance columns show the distance in kilometers between consecutive visits.</p>
        <div class="table-container">
            {case_df.to_html(classes=['data-table', 'table', 'table-striped', 'table-hover'], 
                           index=False, 
                           float_format='%.2f')}
        </div>
    </div>
    """
    
    print(f"Created case visit table with {len(case_df)} cases")
    return html


def generate_exploratory_stats_html(df: pd.DataFrame, output_file: str = None) -> str:
    """
    Generate HTML output with exploratory statistics and tables.
    
    Args:
        df: DataFrame with KMC visit data
        output_file: Optional output file path
        
    Returns:
        Path to the generated HTML file
    """
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create basic statistics
    total_visits = len(df)
    unique_flws = df['flw_id'].nunique()
    unique_opportunities = df['opportunity_id'].nunique()
    unique_cases = df['case_id'].nunique() if 'case_id' in df.columns else 0
    
    # Generate the case visit table
    case_visit_table_html = create_case_visit_table(df)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>KMC Visit Data - Exploratory Statistics</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .stats-summary {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }}
            .stat-card {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #007bff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                color: #666;
                font-size: 14px;
            }}
            .table-section {{
                margin: 30px 0;
            }}
            .table-container {{
                overflow-x: auto;
                margin: 15px 0;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            .data-table th {{
                background-color: #007bff;
                color: white;
                padding: 12px 8px;
                text-align: left;
                font-weight: bold;
            }}
            .data-table td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            .data-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .data-table tr:hover {{
                background-color: #e9ecef;
            }}
            .timestamp {{
                color: #666;
                font-size: 12px;
                text-align: center;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>KMC Visit Data - Exploratory Statistics</h1>
            
            <div class="stats-summary">
                <h2>Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_visits:,}</div>
                        <div class="stat-label">Total Visits</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{unique_flws}</div>
                        <div class="stat-label">Unique FLWs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{unique_opportunities}</div>
                        <div class="stat-label">Unique Opportunities</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{unique_cases}</div>
                        <div class="stat-label">Unique Cases</div>
                    </div>
                </div>
            </div>
            
            <div class="table-section">
                <h2>Raw Data Table</h2>
                <p>Showing all {len(df)} records from the KMC visit data:</p>
                <div class="table-container">
                    {df.to_html(classes=['data-table', 'table', 'table-striped', 'table-hover'], 
                               index=False, 
                               float_format='%.2f')}
                </div>
            </div>
            
            {case_visit_table_html}
            
            <div class="timestamp">
                Report generated on: {timestamp}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Determine output file path
    if output_file is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"kmc_exploratory_stats_{timestamp_str}.html"
    
    # Ensure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_path}")
    return str(output_path)


def main():
    """
    Main function to run the exploratory statistics generator.
    """
    print("KMC Visit Data - Exploratory Statistics Generator")
    print("=" * 50)
    
    # Load environment variables from .env file
    find_dotenv()
    load_dotenv(override=True,verbose=True)
    
    # Configuration - you may want to move these to environment variables or config file
    SUPERSET_URL = os.environ.get("SUPERSET_URL")
    USERNAME = os.environ.get("SUPERSET_USERNAME")
    PASSWORD = os.environ.get("SUPERSET_PASSWORD")
    
    if not all([SUPERSET_URL, USERNAME, PASSWORD]):
        print("Error: All credentials are required.")
        return
    
    try:
        # Load the data
        df = load_kmc_visit_data(
            superset_url=SUPERSET_URL,
            username=USERNAME,
            password=PASSWORD,
            verbose=True
        )
        
        # Generate HTML report
        output_file = generate_exploratory_stats_html(df)
        
        print(f"\n✅ Success! HTML report generated: {output_file}")
        print("You can open this file in your web browser to view the statistics.")
        webbrowser.open(output_file)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return


if __name__ == "__main__":
    main() 