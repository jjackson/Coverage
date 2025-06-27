# run_flw_dashboard.py

import os
from src.utils import data_loader
from src.coverage_master import load_opportunity_domain_mapping
from src.flw_summary_dashboard import create_flw_dashboard
import pandas as pd
from src.create_flw_views import create_flw_views_report
from src.create_statistics import create_statistics_report
from src.create_delivery_map import create_leaflet_map
from src.opportunity_comparison_statistics import create_opportunity_comparison_report
from dotenv import load_dotenv
import webbrowser
import argparse
from datetime import datetime

def load_coverage_data_objects():
    opportunity_to_domain_mapping = load_opportunity_domain_mapping()

    # Get latest CSV
    def get_latest_csv_file(folder="data"):
        csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in 'data/' folder.")

        # Sort by modification time descending, return first
        csv_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = csv_files[0]
        print(f"📁 Latest CSV file found: {latest_file}")
        return latest_file

    csv_path = get_latest_csv_file()
    excel_files = data_loader.get_available_files("data", "xlsx")
    # Load the CSV file into a DataFrame first
    service_delivery_df = pd.read_csv(csv_path)
    service_delivery_by_opportunity_df = data_loader.group_service_delivery_df_by_opportunity(service_delivery_df)

    coverage_data_objects = {}

    for opportunity_name, service_df in service_delivery_by_opportunity_df.items():
        domain_name = opportunity_to_domain_mapping.get(opportunity_name)
        if not domain_name:
            continue

        matching_excels = [f for f in excel_files if domain_name in f]
        if matching_excels:
            # Sort matching files by modification time, latest first
            matching_excels.sort(key=os.path.getmtime, reverse=True)
            matching_excel = matching_excels[0]
        else:
            matching_excel = None
        if not matching_excel:
            continue

        coverage_data = data_loader.get_coverage_data_from_excel_and_csv(matching_excel, None)
        coverage_data.load_service_delivery_from_datafame(service_df)
        coverage_data.project_space = domain_name
        coverage_data.opportunity_name = opportunity_name

        coverage_data_objects[domain_name] = coverage_data

    return coverage_data_objects


if __name__ == "__main__":
    coverage_data_objects = load_coverage_data_objects()

    if coverage_data_objects:
        print(f"\n🚀 Launching dashboard for {len(coverage_data_objects)} opportunities...")
        create_flw_dashboard(coverage_data_objects)
    else:
        print("❌ No valid opportunities found.")
