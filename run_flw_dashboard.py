# run_flw_dashboard.py

import os
from src.utils import data_loader
from src.flw_summary_dashboard import create_flw_dashboard
from src.coverage_master import load_opportunity_domain_mapping

# Load environment mapping
opportunity_to_domain_mapping = load_opportunity_domain_mapping()

# Define paths

#Get the most recent CSV file in the 'data' folder
def get_latest_csv_file(folder="data"):
    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in 'data/' folder.")
    return max(csv_files, key=os.path.getmtime)

csv_path = get_latest_csv_file()
print(f"📁 Using latest CSV file: {csv_path}")

#Get all the excel files
excel_files = data_loader.get_available_files("data", "xlsx")

# Load service delivery grouped by opportunity
service_delivery_by_opportunity_df = data_loader.load_service_delivery_df_by_opportunity(csv_path)

coverage_data_objects = {}

for opportunity_name, service_df in service_delivery_by_opportunity_df.items():
    print(f"\n🔍 Processing opportunity: {opportunity_name}")
    domain_name = opportunity_to_domain_mapping.get(opportunity_name)

    if not domain_name:
        print(f"  ⚠️ Skipping: No domain mapping found for '{opportunity_name}'")
        continue

    # Try to find a matching Excel file
    matching_excel = None
    for excel_file in excel_files:
        if domain_name in excel_file:
            matching_excel = excel_file
            break

    if not matching_excel:
        print(f"  ⚠️ Skipping: No Excel file found containing domain '{domain_name}'")
        continue

    print(f"  ✅ Found Excel file: {matching_excel}")

    coverage_data = data_loader.get_coverage_data_from_excel_and_csv(matching_excel, None)
    coverage_data.load_service_delivery_from_datafame(service_df)
    coverage_data.project_space = domain_name
    coverage_data.opportunity_name = opportunity_name

    coverage_data_objects[domain_name] = coverage_data

# Launch the dashboard
if coverage_data_objects:
    print(f"\n🚀 Launching dashboard for {len(coverage_data_objects)} opportunities...")
    create_flw_dashboard(coverage_data_objects)
else:
    print("❌ No valid opportunities found.")
