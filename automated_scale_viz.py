#!/usr/bin/env python3
"""
Scale Visualization Pipeline

Downloads visit data from multiple Superset queries and generates 
monthly stacked bar charts showing scale by country, plus Excel stats.

Usage:
    python scale_visualization_pipeline.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("? Loaded environment variables from .env file")
except ImportError:
    print("??  python-dotenv not installed - using system environment variables only")

# Add src to path for imports
sys.path.append('src')
from src.utils.data_loader import export_superset_query_with_pagination


# Configuration - Multiple analysis runs
SCALE_CONFIGS = [
    {
        "name": "All Delivery Types",
        "enabled": True,
        "file_suffix": "all",
        "start_date": "2024-10-01",
        "end_date": None,
        "superset_queries": [216],
        "delivery_type_filter": None,  # None = all types
        "output_format": "png",
        "required_columns": ['flw_id', 'visit_id', 'visit_date', 'opp_id', 
                            'country', 'delivery_type', 'llo']
    },
    {
        "name": "CHC Only",
        "enabled": True,
        "file_suffix": "chc",
        "start_date": "2024-10-01",
        "end_date": None,
        "superset_queries": [216],
        "delivery_type_filter": "CHC",
        "output_format": "png",
        "required_columns": ['flw_id', 'visit_id', 'visit_date', 'opp_id', 
                            'country', 'delivery_type', 'llo']
    }
]


class ScaleVisualizationPipeline:
    """Pipeline for generating scale visualizations from visit data"""
    
    def __init__(self, base_output_dir=r"C:\Users\Neal Lesh\Coverage\automated_pipeline_output"):
        """Initialize pipeline with base output directory"""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create today's directory
        today = datetime.now().strftime("%Y_%m_%d")
        self.today_dir = self.base_output_dir / today
        self.today_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.superset_data_dir = self.today_dir / "superset_data"
        self.superset_data_dir.mkdir(exist_ok=True)
        
        self.viz_output_dir = self.today_dir / "scale_visualizations"
        self.viz_output_dir.mkdir(exist_ok=True)
        
        # Load Superset credentials
        self.superset_url = os.getenv('SUPERSET_URL')
        self.superset_username = os.getenv('SUPERSET_USERNAME')
        self.superset_password = os.getenv('SUPERSET_PASSWORD')
        
        self._validate_credentials()
        
        print(f"?? Working directory: {self.today_dir}")
    
    def _validate_credentials(self):
        """Validate Superset credentials"""
        missing = []
        if not self.superset_url:
            missing.append('SUPERSET_URL')
        if not self.superset_username:
            missing.append('SUPERSET_USERNAME')
        if not self.superset_password:
            missing.append('SUPERSET_PASSWORD')
        
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        print(f"? Superset credentials loaded: {self.superset_url}")
    
    def run_pipeline(self, configs):
        """Run the complete pipeline for all configurations"""
        # Filter to enabled configs
        enabled_configs = [c for c in configs if c.get('enabled', True)]
        
        print(f"\n?? Starting Scale Visualization Pipeline")
        print(f"Processing {len(enabled_configs)} configurations")
        print("=" * 60)
        
        try:
            # Phase 1: Download Superset data (collect unique query IDs)
            all_query_ids = set()
            for config in enabled_configs:
                all_query_ids.update(config['superset_queries'])
            
            csv_files = self._download_all_queries(list(all_query_ids))
            
            # Phase 2: Process each configuration
            results = []
            for i, config in enumerate(enabled_configs, 1):
                print(f"\n?? Configuration {i}/{len(enabled_configs)}: {config['name']}")
                print("-" * 40)
                
                # Get CSV files for this config
                config_csv_files = [csv_files[qid] for qid in config['superset_queries']]
                
                # Combine and clean data
                combined_df = self._combine_and_clean(config_csv_files, config)
                
                # Generate visualizations
                output_files = self._generate_visualizations(combined_df, config)
                
                # Generate Excel stats (only once for the full dataset config)
                if config['delivery_type_filter'] is None:
                    excel_file = self._generate_stats_excel(combined_df, config)
                    output_files.append(excel_file)
                
                results.append({
                    'config': config,
                    'status': 'success',
                    'files': output_files
                })
                
                print(f"  ? Generated {len(output_files)} files")
            
            print(f"\n? Pipeline completed successfully!")
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            print(f"\n? Pipeline failed: {str(e)}")
            raise
    
    def _download_all_queries(self, query_ids):
        """Download data from all Superset queries (cache-aware)"""
        print(f"\n?? Phase 1: Downloading Superset Data")
        print("-" * 40)
        
        csv_files = {}
        
        for query_id in query_ids:
            csv_file = self.superset_data_dir / f"query_{query_id}_data.csv"
            
            if csv_file.exists():
                print(f"  ? Query {query_id}: Using cached data")
                csv_files[query_id] = csv_file
            else:
                print(f"  ?? Query {query_id}: Downloading...")
                try:
                    # Get SQL from saved query
                    sql_query = self._get_sql_from_saved_query(query_id)
                    
                    # Download data
                    downloaded_file = export_superset_query_with_pagination(
                        superset_url=self.superset_url,
                        sql_query=sql_query,
                        username=self.superset_username,
                        password=self.superset_password,
                        output_filename=str(csv_file.with_suffix(''))
                    )
                    
                    if os.path.exists(downloaded_file):
                        df = pd.read_csv(downloaded_file)
                        print(f"  ? Query {query_id}: Downloaded {len(df):,} rows")
                        csv_files[query_id] = Path(downloaded_file)
                    else:
                        print(f"  ? Query {query_id}: Download failed")
                        
                except Exception as e:
                    print(f"  ? Query {query_id}: Error - {str(e)}")
                    raise
        
        return csv_files
    
    def _get_sql_from_saved_query(self, query_id):
        """Get SQL query from Superset saved query ID"""
        try:
            session = requests.Session()
            
            # Login
            auth_url = f'{self.superset_url}/api/v1/security/login'
            auth_data = {
                'username': self.superset_username,
                'password': self.superset_password,
                'provider': 'db',
                'refresh': True
            }
            
            response = session.post(auth_url, json=auth_data, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Authentication failed: {response.text}")
            
            auth_data = response.json()
            access_token = auth_data.get('access_token')
            if not access_token:
                raise RuntimeError("No access token received")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Get CSRF token
            csrf_url = f'{self.superset_url}/api/v1/security/csrf_token/'
            csrf_response = session.get(csrf_url, headers=headers, timeout=30)
            if csrf_response.status_code == 200:
                csrf_data = csrf_response.json()
                csrf_token = csrf_data.get('result')
                if csrf_token:
                    headers['x-csrftoken'] = csrf_token
                    headers['Referer'] = self.superset_url + "/sqllab"
            
            # Get saved query
            saved_query_url = f'{self.superset_url}/api/v1/saved_query/{query_id}'
            response = session.get(saved_query_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get saved query {query_id}: {response.text}")
            
            query_data = response.json()
            result = query_data.get('result', {})
            sql_query = result.get('sql', '')
            
            if not sql_query:
                raise RuntimeError(f"No SQL found in saved query {query_id}")
            
            return sql_query
            
        except Exception as e:
            raise RuntimeError(f"Failed to get SQL from query {query_id}: {str(e)}")
    
    def _combine_and_clean(self, csv_files, config):
        """Combine all CSV files and clean data"""
        print(f"  ?? Combining and cleaning data...")
        
        # Load and combine all files
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"    ?? Combined: {len(combined_df):,} rows")
        
        # Validate required columns
        missing_cols = set(config['required_columns']) - set(combined_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert visit_date to datetime (handles ISO 8601 format with/without microseconds)
        combined_df['visit_date'] = pd.to_datetime(combined_df['visit_date'], format='ISO8601', utc=True)
        
        # Show date range in data
        min_date = combined_df['visit_date'].min()
        max_date = combined_df['visit_date'].max()
        print(f"    ? Date range: {min_date.date()} to {max_date.date()}")
        
        # Filter by date range
        if config['start_date']:
            start_date = pd.to_datetime(config['start_date']).tz_localize('UTC')
            combined_df = combined_df[combined_df['visit_date'] >= start_date]
            print(f"    ? Filtered to >= {config['start_date']}: {len(combined_df):,} rows")
        
        if config['end_date']:
            end_date = pd.to_datetime(config['end_date']).tz_localize('UTC')
            combined_df = combined_df[combined_df['visit_date'] <= end_date]
            print(f"    ? Filtered to <= {config['end_date']}: {len(combined_df):,} rows")
        
        # Filter by delivery type if specified
        if config['delivery_type_filter']:
            combined_df = combined_df[combined_df['delivery_type'] == config['delivery_type_filter']]
            print(f"    ? Filtered to delivery_type={config['delivery_type_filter']}: {len(combined_df):,} rows")
        
        print(f"    ? Final dataset: {len(combined_df):,} rows")
        
        return combined_df
    
    def _generate_visualizations(self, df, config):
        """Generate both standard and report versions of the visualization"""
        print(f"  ?? Generating visualizations...")
        
        output_files = []
        today = datetime.now().strftime("%Y_%m_%d")
        
        # Generate both aspect ratios
        aspect_ratios = [
            ('standard', (14, 8)),
            ('report', (16, 5))
        ]
        
        for ratio_name, figsize in aspect_ratios:
            output_file = self._create_stacked_bar_chart(df, config, figsize, ratio_name, today)
            output_files.append(output_file)
        
        return output_files
    
    def _create_stacked_bar_chart(self, df, config, figsize, ratio_name, today):
        """Create a single stacked bar chart with specified dimensions"""
        
        # Add year-month column for grouping
        df_copy = df.copy()
        df_copy['year_month'] = df_copy['visit_date'].dt.to_period('M')
        
        # Get current month (in UTC to match data)
        current_month = pd.Timestamp.now(tz='UTC').to_period('M')
        
        # Drop future months
        df_copy = df_copy[df_copy['year_month'] <= current_month]
        
        # Calculate monthly visits by country
        monthly_by_country = df_copy.groupby(['year_month', 'country']).size().unstack(fill_value=0)
        
        # Calculate total unique LLOs per country (across entire dataset)
        llo_counts = df_copy.groupby('country')['llo'].nunique().to_dict()
        
        # Get delivery types per country (for unrestricted configs)
        delivery_types_by_country = {}
        if config['delivery_type_filter'] is None:
            for country in monthly_by_country.columns:
                country_df = df_copy[df_copy['country'] == country]
                dtypes = sorted(country_df['delivery_type'].unique())
                delivery_types_by_country[country] = dtypes
        
        # Create country labels
        country_labels = {}
        for country in monthly_by_country.columns:
            llo_count = llo_counts[country]
            llo_text = "LLO" if llo_count == 1 else "LLOs"
            
            if config['delivery_type_filter'] is None and country in delivery_types_by_country:
                dtypes = ', '.join(delivery_types_by_country[country])
                country_labels[country] = f"{country} ({llo_count} {llo_text} | {dtypes})"
            else:
                country_labels[country] = f"{country} ({llo_count} {llo_text})"
        
        # Sort countries by total visits (descending)
        country_order = monthly_by_country.sum().sort_values(ascending=False).index
        monthly_by_country = monthly_by_country[country_order]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert Period index to timestamp for plotting
        x_dates = monthly_by_country.index.to_timestamp()
        
        # Create stacked bar chart
        bottom = None
        colors = plt.cm.tab10.colors
        
        for i, country in enumerate(country_order):
            values = monthly_by_country[country].values
            label = country_labels[country]
            
            ax.bar(x_dates, values, bottom=bottom, label=label, 
                   color=colors[i % len(colors)], width=25)
            
            if bottom is None:
                bottom = values
            else:
                bottom = bottom + values
        
        # Format x-axis with custom labels for partial month
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Custom formatter to add "(partial)" to current month
        def format_month(x, pos):
            date = mdates.num2date(x)
            month_period = pd.Timestamp(date).to_period('M')
            label = date.strftime('%b %Y')
            if month_period == current_month:
                label += ' (partial)'
            return label
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_month))
        plt.xticks(rotation=45, ha='right')
        
        # Labels and title
        ax.set_ylabel('Individuals receiving verified services', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Monthly Visits by Country', fontsize=16, fontweight='bold', pad=20)
        
        # Format y-axis with thousand separators
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Legend
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        suffix = config['file_suffix']
        filename = f"monthly_visits_stacked_{suffix}_{today}"
        if ratio_name != 'standard':
            filename += f"_{ratio_name}"
        filename += f".{config['output_format']}"
        
        output_file = self.viz_output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ? Saved {ratio_name}: {output_file.name}")
        
        return output_file
    
    
    def _print_summary(self, results):
        """Print pipeline execution summary"""
        print(f"\n?? Pipeline Summary")
        print("=" * 60)
        
        for result in results:
            config_name = result['config']['name']
            file_count = len(result['files'])
            print(f"  ? {config_name}: {file_count} files generated")
            for file_path in result['files']:
                print(f"       {Path(file_path).name}")


    def _generate_stats_excel(self, df, config):
        """Generate Excel file with statistical breakdowns"""
        print(f"  ?? Generating Excel statistics...")
        
        df_copy = df.copy()
        df_copy['year_month'] = df_copy['visit_date'].dt.to_period('M').astype(str)
        
        today = datetime.now().strftime("%Y_%m_%d")
        excel_file = self.viz_output_dir / f"visit_statistics_{today}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Tab 1: Monthly by Country
            monthly_country = pd.pivot_table(
                df_copy, 
                values='visit_id', 
                index='country',
                columns='year_month',
                aggfunc='count',
                fill_value=0,
                margins=True,
                margins_name='TOTAL'
            )
            
            # Add metadata columns for countries
            country_delivery_types = df_copy.groupby('country')['delivery_type'].apply(
                lambda x: ', '.join(sorted(x.unique()))
            )
            country_llo_counts = df_copy.groupby('country')['llo'].nunique()
            
            monthly_country.insert(0, 'Number LLOs', monthly_country.index.map(
                lambda x: country_llo_counts.get(x, '') if x != 'TOTAL' else ''
            ))
            monthly_country.insert(0, 'Delivery_Types', monthly_country.index.map(
                lambda x: country_delivery_types.get(x, '') if x != 'TOTAL' else ''
            ))
            monthly_country.insert(0, 'Countries', monthly_country.index.map(
                lambda x: x if x != 'TOTAL' else ''
            ))
            
            monthly_country.to_excel(writer, sheet_name='Monthly by Country')
            
            # Tab 2: Monthly by LLO
            monthly_llo = pd.pivot_table(
                df_copy,
                values='visit_id',
                index='llo',
                columns='year_month',
                aggfunc='count',
                fill_value=0,
                margins=True,
                margins_name='TOTAL'
            )
            
            # Add metadata columns for LLOs
            llo_countries = df_copy.groupby('llo')['country'].apply(
                lambda x: ', '.join(sorted(x.unique()))
            )
            llo_delivery_types = df_copy.groupby('llo')['delivery_type'].apply(
                lambda x: ', '.join(sorted(x.unique()))
            )
            
            monthly_llo.insert(0, 'Delivery_Types', monthly_llo.index.map(
                lambda x: llo_delivery_types.get(x, '') if x != 'TOTAL' else ''
            ))
            monthly_llo.insert(0, 'Countries', monthly_llo.index.map(
                lambda x: llo_countries.get(x, '') if x != 'TOTAL' else ''
            ))
            
            monthly_llo.to_excel(writer, sheet_name='Monthly by LLO')
            
            # Tab 3: Monthly by Delivery Type
            monthly_delivery = pd.pivot_table(
                df_copy,
                values='visit_id',
                index='delivery_type',
                columns='year_month',
                aggfunc='count',
                fill_value=0,
                margins=True,
                margins_name='TOTAL'
            )
            
            # Add metadata columns for delivery types
            delivery_countries = df_copy.groupby('delivery_type')['country'].apply(
                lambda x: ', '.join(sorted(x.unique()))
            )
            delivery_llo_counts = df_copy.groupby('delivery_type')['llo'].nunique()
            
            monthly_delivery.insert(0, 'Number LLOs', monthly_delivery.index.map(
                lambda x: delivery_llo_counts.get(x, '') if x != 'TOTAL' else ''
            ))
            monthly_delivery.insert(0, 'Countries', monthly_delivery.index.map(
                lambda x: delivery_countries.get(x, '') if x != 'TOTAL' else ''
            ))
            
            monthly_delivery.to_excel(writer, sheet_name='Monthly by Delivery Type')
            
            # Tab 4: LLO by Delivery Type
            llo_delivery = pd.pivot_table(
                df_copy,
                values='visit_id',
                index='llo',
                columns='delivery_type',
                aggfunc='count',
                fill_value=0,
                margins=True,
                margins_name='TOTAL'
            )
            
            # Add countries column
            llo_countries = df_copy.groupby('llo')['country'].apply(
                lambda x: ', '.join(sorted(x.unique()))
            )
            
            llo_delivery.insert(0, 'Countries', llo_delivery.index.map(
                lambda x: llo_countries.get(x, '') if x != 'TOTAL' else ''
            ))
            
            llo_delivery.to_excel(writer, sheet_name='LLO by Delivery Type')
        
        print(f"    ? Saved Excel: {excel_file.name}")
        return excel_file

def main():
    """Main entry point"""
    print("?? Scale Visualization Pipeline")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = ScaleVisualizationPipeline()
        
        # Run the pipeline
        results = pipeline.run_pipeline(SCALE_CONFIGS)
        
        print(f"\n?? Success!")
        
    except Exception as e:
        print(f"\n?? Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
