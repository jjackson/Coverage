import os
from pprint import pprint
import pickle
import pandas as pd
import json
from dotenv import load_dotenv, find_dotenv
import src.utils.data_loader as data_loader
from sqlqueries.sql_queries import SQL_QUERIES
from typing import Dict
from pathlib import Path

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

def load_opportunity_domain_mapping() -> Dict[str, str]:
    """Load opportunity to domain mapping from environment variable."""
    mapping_str = os.environ.get('OPPORTUNITY_DOMAIN_MAPPING', '')
    if not mapping_str:
        # Return default mapping if no environment variable is set
        return {
           "ZEGCAWIS | CHC Givewell Scale Up": "ccc-chc-zegcawis-2024-25",
           "COWACDI | CHC Givewell Scale Up": "ccc-chc-cowacdi-2024-25"
        }
    
    try:
        # Try to parse as JSON first
        return json.loads(mapping_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, try pipe-separated format
        try:
            mapping = {}
            pairs = mapping_str.split('|')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)  # Split only on first colon
                    mapping[key.strip()] = value.strip()
            return mapping
        except Exception as e:
            raise ValueError(f"Could not parse OPPORTUNITY_DOMAIN_MAPPING from environment variable. Please check the format. Error: {e}")

def output_as_excel_in_downloads(df, file_name):
    if(df is not None and not df.empty):
        output_path = os.path.join(DOWNLOADS_DIR, file_name +".xlsx")
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name="Merged", index=False)
        print(f"Generated file: {output_path}")
    else : 
        print("DF is either empty or Null")

def init_microplanning():
    # Load JSON from a file or string
    current_dir = Path(__file__).parent
    json_path = current_dir /'src' / 'opportunity_target' / 'opportunity_target.json'
    if not json_path.exists():
        print(f"Error: {json_path} not found. Please ensure the file exists.")
        return

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    rows = []
    for domain, domain_data in json_data['domains'].items():
        for ward, targets in domain_data['wards'].items():
            row = {
            'domain': domain,
            'ward': ward,
            'visit_target': targets['visit_target'],
            'building_target': targets['building_target'],
            'du_target': targets['du_target']
            }
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Display the DataFrame
    print(df)
    return df

def main():
    final_df = init_microplanning()
    # Load environment variables from .env file
    dotenv_path=find_dotenv("./src/.env")
    load_dotenv(dotenv_path,override=True)

    opportunity_to_domain_mapping = load_opportunity_domain_mapping()
    valid_opportunities = list(opportunity_to_domain_mapping.values())
    
    for domain in valid_opportunities:
        domain_df = pd.DataFrame()
         # Get the directory of the current script
        _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
        # Load coverage data from pickle file for each domain
        pickle_file_name = domain.replace('"', '')+ ".pkl"
        pickle_path = os.path.join(_dir, pickle_file_name)
        if not os.path.exists(pickle_path):
            print(f"Error: {pickle_file_name} not found. Please run run_coverage.py first.")
            return
        
        try:
            with open(pickle_path, 'rb') as f:
                service_df = pickle.load(f)
        except Exception as e:
            print(f"Error : {str(e)}")

        if service_df is not None and not service_df.empty :
            # Convert coverage data to DataFrame if not already
            if isinstance(service_df, pd.DataFrame):
                service_df = service_df
            else:
                service_df = pd.DataFrame(service_df)

            # Output the DataFrame to Excel in Downloads folder
            output_as_excel_in_downloads(service_df, f"{domain}_coverage_data")

 

if __name__ == "__main__":
    main()