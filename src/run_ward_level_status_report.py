import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pickle
import pandas as pd
import json
from dotenv import load_dotenv, find_dotenv
import src.utils.data_loader as data_loader
from sqlqueries.sql_queries import SQL_QUERIES
from typing import Dict
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from src.sqlqueries.sql_queries import SQL_QUERIES
import constants

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")


def get_superset_data(sql):
    
    dotenv_path=find_dotenv("./src/.env")
    load_dotenv(dotenv_path,override=True)
    superset_url = os.environ.get('SUPERSET_URL')
    superset_username = os.environ.get('SUPERSET_USERNAME')
    superset_password = os.environ.get('SUPERSET_PASSWORD')
    chunk_size: int = constants.SQL_CHUNK_SIZE
    timeout: int = constants.API_TIMEOUT_LIMIT
    verbose: bool = True
    if superset_username is None or superset_password is None or superset_url is None:
        raise ValueError("USERNAME, PASSWORD or SUPERSET_URL must be set in the environment or .env file")

    output_df = data_loader.superset_query_with_pagination_as_df(superset_url, sql, superset_username, superset_password, chunk_size, timeout, verbose
    )
    return output_df

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

def init_microplanning_ward_level_data_frame():
    # Load JSON from a file or string
    current_dir = Path(__file__).parent
    json_path = current_dir / 'opportunity_target' / 'ward_level_targets.json'
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
    df['visits_completed'] = 0
    df['visits_completed_last_week'] = 0
    df['pct_visits_completed'] = 0.0
    df['pct_visits_completed_last_week'] = 0.0
    df['buildings_completed'] = 0
    df['buildings_completed_last_week'] = 0
    df['pct_buildings_completed'] = 0.0
    df['pct_buildings_completed_last_week'] = 0.0
    df['du_completed'] = 0
    df['du_completed_last_week'] = 0
    df['pct_du_completed'] = 0.0
    df['pct_du_completed_last_week'] = 0.0

    # Display the DataFrame
    return df

def init_microplannin_opp_level_data_frame():
    # Load JSON from a file or string
    current_dir = Path(__file__).parent
    json_path = current_dir / 'opportunity_target' / 'opp_level_targets.json'
    if not json_path.exists():
        print(f"Error: {json_path} not found. Please ensure the file exists.")
        return

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    rows = []
    for domain, targets in json_data['domains'].items():
        row = {
            'domain': domain,
            'visit_target': targets['visit_target'],
            'building_target': targets['building_target'],
            'du_target': targets['du_target'],
            'start_date': targets['start_date'],
            'end_date': targets['end_date']
        }
        rows.append(row)
    # Create DataFrame
    df = pd.DataFrame(rows)
    df['visits_completed'] = 0
    df['visits_completed_last_week'] = 0
    df['pct_visits_completed'] = 0.0
    df['pct_visits_completed_last_week'] = 0.0
    df['buildings_completed'] = 0
    df['buildings_completed_last_week'] = 0
    df['pct_buildings_completed'] = 0.0
    df['pct_buildings_completed_last_week'] = 0.0
    df['du_completed'] = 0
    df['du_completed_last_week'] = 0
    df['pct_du_completed'] = 0.0
    df['pct_du_completed_last_week'] = 0.0

    # Display the DataFrame
    return df

def find_ward_column_name(domain):
    match domain:
        case 'ccc-chc-ahti-2024-25':
            return "ward_name"
        case 'ccc-chc-ruwoyd-2024-25':
            return "ward"
        case 'ccc-chc-jhf-2024-25':
            return "ward"
        case 'ccc-chc-rwyd-2024-25':
            return "ward_name"
        case 'ccc-chc-eha-c-2024-25':
            return "ward_name"
        case 'ccc-chc-isodaf-2024-25':
            return "ward_name"
        case 'ccc-chc-zegcawis-2024-25':
            return "ward_name"
        case 'ccc-chc-cowacdi-2024-25':
            return "ward_name"
        case _:
            return ""


def main():
    
    #Load environment variables from .env file
    dotenv_path=find_dotenv("./src/.env")
    load_dotenv(dotenv_path,override=True)
    
    # Load opportunity to domain mapping from environment variable
    opportunity_to_domain_mapping = load_opportunity_domain_mapping()
    valid_opportunities = list(opportunity_to_domain_mapping.values())

    # update visit related columns
    #get details of visits from visit SQL query with delivery-unit case_ids for using it later
    print("Fetching data from SQL for visit related columns...")

    #####prepping visit data######
    visit_sql = SQL_QUERIES['opp_user_visit_du_case_id_mapping']
    visit_data_df = get_superset_data(visit_sql)
    visit_data_df.rename(columns={'du_case_id': 'case_id'}, inplace=True)
    visit_data_df.rename(columns={'time_end': 'visit_date'}, inplace=True)
    visit_data_df.rename(columns={'name': 'domain'}, inplace=True)

    # Convert visit_date to datetime
    visit_data_df['visit_date'] = pd.to_datetime(visit_data_df['visit_date'],format='mixed', utc=True, errors='coerce')
    # Extract only the date component from visit_date
    visit_data_df['visit_date'] = visit_data_df['visit_date'].dt.date

    #conver domain name values from actual name to cchq domain names
    visit_data_df['domain'] = visit_data_df['domain'].map(opportunity_to_domain_mapping)
    print("Visit data fetched successfully.")
    output_as_excel_in_downloads(visit_data_df, "visit_data_df")

    ward_level_df = init_microplanning_ward_level_data_frame()
    ward_level_final_df = generate_ward_level_status_report(valid_opportunities,visit_data_df, ward_level_df)

    opp_level_df = init_microplannin_opp_level_data_frame()
    opp_level_final_df = generate_opp_level_status_report(valid_opportunities,visit_data_df, opp_level_df)

    timline_df = generate_timeline_based_status_report(valid_opportunities, visit_data_df, ward_level_df)


def generate_opp_level_status_report(valid_opportunities,visit_data_df,final_df):
    #####Get all the required metrics##### 
    for domain in valid_opportunities:
        domain_df = pd.DataFrame()
         # Get the directory of the current script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(project_root, 'data')
        # Load coverage data from pickle file for each domain
        pickle_file_name = domain.replace('"', '')+ ".pkl"
        pickle_path = os.path.join(data_path, pickle_file_name)
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
            #output_as_excel_in_downloads(service_df, "domain_"+ domain+"_pickel_data")
            domain_df = service_df.copy()
            domain_df_last_week = service_df.copy()
            # Filter the DataFrame for the current domain
            domain_df_last_week['last_modified'] = pd.to_datetime(domain_df_last_week['last_modified'], utc=True)
             # Define analysis window
            today_utc = datetime.now(pytz.UTC)
            seven_days_ago = today_utc - timedelta(days=7)

            # Filter rows modified in the last 7 days
            domain_df_last_week = domain_df_last_week[domain_df_last_week['last_modified'] >= seven_days_ago]
            print(f"Processing domain: {domain}")

            #Update DU relateed columns
            final_df.loc[(final_df['domain'] == domain) , 'du_completed'] = domain_df[ (domain_df['du_status'] == 'completed')].shape[0]
            final_df.loc[(final_df['domain'] == domain) , 'du_completed_last_week'] = domain_df_last_week[ (domain_df_last_week['du_status'] == 'completed')].shape[0]
            final_df.loc[(final_df['domain'] == domain) , 'pct_du_completed'] = 100*final_df.loc[(final_df['domain'] == domain) , 'du_completed'] / final_df.loc[(final_df['domain'] == domain) , 'du_target']
            final_df.loc[(final_df['domain'] == domain) , 'pct_du_completed_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) , 'du_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) , 'du_target']
                    
            #Update building related columns
            buildings_completed = domain_df[ (domain_df['du_status'] == 'completed')]['buildings'].sum()
            final_df.loc[(final_df['domain'] == domain) ,'buildings_completed'] = buildings_completed
            buildings_completed_last_week = domain_df_last_week[ (domain_df_last_week['du_status'] == 'completed')]['buildings'].sum()
            final_df.loc[(final_df['domain'] == domain) ,'buildings_completed_last_week'] = buildings_completed_last_week
            final_df.loc[(final_df['domain'] == domain) , 'pct_buildings_completed'] = 100*final_df.loc[(final_df['domain'] == domain), 'buildings_completed'] / final_df.loc[(final_df['domain'] == domain) , 'building_target']
            final_df.loc[(final_df['domain'] == domain) , 'pct_buildings_completed_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) , 'buildings_completed_last_week'] / final_df.loc[(final_df['domain'] == domain), 'building_target']
                    
            #Update visit related columns
            subset_visit_data_df = visit_data_df[visit_data_df['domain'] == domain]
            subset_visit_data_df = pd.merge(subset_visit_data_df, domain_df, on="case_id",how="left")
            subset_visit_data_df = subset_visit_data_df[['case_id','visit_date','domain']]

            subset_visit_data_df_last_week = subset_visit_data_df.copy()
            today_utc = datetime.now(pytz.UTC)
            seven_days_ago = (today_utc - timedelta(days=7)).date()
            # Filter rows modified in the last 7 days
            subset_visit_data_df_last_week = subset_visit_data_df_last_week[subset_visit_data_df_last_week['visit_date'] >= seven_days_ago]
                    
            # Count visits completed total
            visits_completed = subset_visit_data_df.shape[0]
            final_df.loc[(final_df['domain'] == domain) , 'visits_completed'] = visits_completed
            # Count visits completed last week  
            visits_completed_last_week = subset_visit_data_df_last_week.shape[0]
            final_df.loc[(final_df['domain'] == domain) , 'visits_completed_last_week'] = visits_completed_last_week
            # Calculate percentage of visits completed

            final_df.loc[(final_df['domain'] == domain), 'pct_visits_completed'] = 100*final_df.loc[(final_df['domain'] == domain) , 'visits_completed'] / final_df.loc[(final_df['domain'] == domain) , 'visit_target']
            final_df.loc[(final_df['domain'] == domain) , 'pct_visits_completed_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) , 'visits_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) , 'visit_target']
            
            #Update percentage completion for the domain
            start_date_to_date = datetime.strptime(final_df.loc[(final_df['domain'] == domain) , 'start_date'].iloc[0], '%Y-%m-%d').date()
            end_date_to_date = datetime.strptime(final_df.loc[(final_df['domain'] == domain) , 'end_date'].iloc[0], '%Y-%m-%d').date()
            total_days = (end_date_to_date - start_date_to_date).days
            total_days_till_now = (today_utc - pd.to_datetime(final_df['start_date'].min(), utc=True)).days
            final_df.loc[(final_df['domain'] == domain), 'pct_completion'] = 100*total_days_till_now/total_days

            final_df.loc[(final_df['domain'] == domain), 'pct_building_microplanning_completion_rate'] = 100*final_df.loc[(final_df['domain'] == domain) , 'pct_buildings_completed'] / final_df.loc[(final_df['domain'] == domain) , 'pct_visits_completed']
            final_df.loc[(final_df['domain'] == domain), 'pct_building_microplanning_completion_rate_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) , 'pct_buildings_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) , 'visits_completed_last_week']

            final_df.loc[(final_df['domain'] == domain), 'pct_du_microplanning_completion_rate'] = 100*final_df.loc[(final_df['domain'] == domain) , 'pct_du_completed'] / final_df.loc[(final_df['domain'] == domain) , 'pct_visits_completed']
            final_df.loc[(final_df['domain'] == domain), 'pct_du_microplanning_completion_rate_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) , 'pct_du_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) , 'visits_completed_last_week']
            
        else:
            print(f"No data found for domain {domain}. Run the coverage for all the domains. For now, we are skipping the domain {domain}...")
    
    # Round all pct_ columns to two decimal places
    pct_cols = [col for col in final_df.columns if col.startswith('pct_')]
    final_df.loc[:, pct_cols] = final_df[pct_cols].round(2)
    
    output_as_excel_in_downloads(final_df, "opp_level_status_report")
    return final_df

def generate_ward_level_status_report(valid_opportunities,visit_data_df,final_df):
    #####Get all the required metrics##### 
    for domain in valid_opportunities:
        domain_df = pd.DataFrame()
         # Get the directory of the current script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(project_root, 'data')
        # Load coverage data from pickle file for each domain
        pickle_file_name = domain.replace('"', '')+ ".pkl"
        pickle_path = os.path.join(data_path, pickle_file_name)
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
            #output_as_excel_in_downloads(service_df, "domain_"+ domain+"_pickel_data")
            domain_df = service_df.copy()
            domain_df_last_week = service_df.copy()
            # Filter the DataFrame for the current domain
            domain_df_last_week['last_modified'] = pd.to_datetime(domain_df_last_week['last_modified'], utc=True)
             # Define analysis window
            today_utc = datetime.now(pytz.UTC)
            seven_days_ago = today_utc - timedelta(days=7)

            # Filter rows modified in the last 7 days
            domain_df_last_week = domain_df_last_week[domain_df_last_week['last_modified'] >= seven_days_ago]

            ward_column = find_ward_column_name(domain)
            if(ward_column != ""):
                wards = final_df.loc[final_df['domain'] == domain, 'ward'].unique()   
                for ward in wards:
                    print(f"Processing domain: {domain}, ward: {ward}")

                    #Update DU relateed columns
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'du_completed'] = domain_df[(domain_df[ward_column] == ward) & (domain_df['du_status'] == 'completed')].shape[0]
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'du_completed_last_week'] = domain_df_last_week[(domain_df_last_week[ward_column] == ward) & (domain_df_last_week['du_status'] == 'completed')].shape[0]
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'pct_du_completed'] = 100*final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'du_completed'] / final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'du_target']
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'pct_du_completed_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'du_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'du_target']
                    
                    #Update building related columns
                    buildings_completed = domain_df[(domain_df[ward_column] == ward) & (domain_df['du_status'] == 'completed')]['buildings'].sum()
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward),'buildings_completed'] = buildings_completed
                    buildings_completed_last_week = domain_df_last_week[(domain_df_last_week[ward_column] == ward) & (domain_df_last_week['du_status'] == 'completed')]['buildings'].sum()
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward),'buildings_completed_last_week'] = buildings_completed_last_week
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'pct_buildings_completed'] = 100*final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'buildings_completed'] / final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'building_target']
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'pct_buildings_completed_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'buildings_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'building_target']
                    
                    #Update visit related columns
                    subset_visit_data_df = visit_data_df[visit_data_df['domain'] == domain]
                    subset_visit_data_df = pd.merge(subset_visit_data_df, domain_df, on="case_id",how="left")
                    subset_visit_data_df = subset_visit_data_df[['case_id','visit_date','domain',ward_column]]

                    subset_visit_data_df_last_week = subset_visit_data_df.copy()
                    today_utc = datetime.now(pytz.UTC)
                    seven_days_ago = (today_utc - timedelta(days=7)).date()
                    # Filter rows modified in the last 7 days
                    subset_visit_data_df_last_week = subset_visit_data_df_last_week[subset_visit_data_df_last_week['visit_date'] >= seven_days_ago]
                    
                    # Count visits completed total
                    visits_completed = subset_visit_data_df[subset_visit_data_df[ward_column] == ward].shape[0]
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'visits_completed'] = visits_completed
                    # Count visits completed last week  
                    visits_completed_last_week = subset_visit_data_df_last_week[subset_visit_data_df_last_week[ward_column] == ward].shape[0]
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'visits_completed_last_week'] = visits_completed_last_week
                    # Calculate percentage of visits completed

                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'pct_visits_completed'] = 100*final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'visits_completed'] / final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'visit_target']
                    final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'pct_visits_completed_last_week'] = 100*final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'visits_completed_last_week'] / final_df.loc[(final_df['domain'] == domain) & (final_df['ward'] == ward), 'visit_target']

            else:
                print(f"Warning: No ward column found for domain {domain}. Skipping DU completion updates.")
                # @TODO: Handle the case of ccc-chc-zegcawis-2024-25 and ccc-chc-cowacdi-2024-25
            
        else:
            print(f"No data found for domain {domain}. Run the coverage for all the domains. For now, we are skipping the domain {domain}...")
     # Round all pct_ columns to two decimal places
    pct_cols = [col for col in final_df.columns if col.startswith('pct_')]
    final_df.loc[:, pct_cols] = final_df[pct_cols].round(2)
    output_as_excel_in_downloads(final_df, "ward_level_status_report")
    return final_df

def generate_timeline_based_status_report(valid_opportunities, visit_data_df, ward_level_df):
    """
    Generate timeline-based status report showing cumulative and last 7 days metrics
    for each visit date, domain, and ward combination.
    
    Args:
        valid_opportunities: List of valid domain names
        visit_data_df: DataFrame containing visit data with columns: visit_date, domain, case_id
    
    Returns:
        DataFrame with timeline-based metrics
    """
    timeline_rows = []
    
    for domain in valid_opportunities:
        print(f"Processing timeline for domain: {domain}")
        
        # Get the directory of the current script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(project_root, 'data')
        
        # Load coverage data from pickle file for each domain
        pickle_file_name = domain.replace('"', '') + ".pkl"
        pickle_path = os.path.join(data_path, pickle_file_name)
        
        if not os.path.exists(pickle_path):
            print(f"Error: {pickle_file_name} not found. Please run run_coverage.py first.")
            continue
        
        try:
            with open(pickle_path, 'rb') as f:
                service_df = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
            continue
        
        if service_df is None or service_df.empty:
            print(f"No data found for domain {domain}. Skipping...")
            continue
        
        # Convert coverage data to DataFrame if not already
        if not isinstance(service_df, pd.DataFrame):
            service_df = pd.DataFrame(service_df)
        
        # Get ward column name for this domain
        ward_column = find_ward_column_name(domain)
        if ward_column == "":
            print(f"Warning: No ward column found for domain {domain}. Skipping...")
            continue
        
        # Filter visit data for current domain
        domain_visit_data = visit_data_df[visit_data_df['domain'] == domain].copy()
        
        # Merge visit data with service data to get building and DU information
        domain_visit_data = pd.merge(domain_visit_data, service_df, on="case_id", how="left")
        
        # Get unique visit dates for this domain
        unique_visit_dates = sorted(domain_visit_data['visit_date'].unique())
        
        # Get unique wards for this domain
        unique_wards = domain_visit_data[ward_column].unique()
        
        for visit_date in unique_visit_dates:
            for ward in unique_wards:
                # Filter data for current ward
                ward_visit_data = domain_visit_data[domain_visit_data[ward_column] == ward].copy()
                
                if ward_visit_data.empty:
                    continue
                
                # a) visits_so_far: number of visits performed on or before current visit_date
                visits_so_far = len(ward_visit_data[ward_visit_data['visit_date'] <= visit_date])
                
                # b) visits_last7days: number of visits performed in last 7 days from current visit_date
                seven_days_before = visit_date - timedelta(days=7)
                visits_last7days = len(ward_visit_data[
                    (ward_visit_data['visit_date'] >= seven_days_before) & 
                    (ward_visit_data['visit_date'] <= visit_date)
                ])
                
                # c) buildings_so_far: count of unique buildings covered on or before current visit_date
                buildings_so_far_data = ward_visit_data[
                    (ward_visit_data['visit_date'] <= visit_date) & 
                    (ward_visit_data['du_status'] == 'completed')
                ]
                # Count unique case_ids (buildings/DUs), not visit count
                buildings_so_far = buildings_so_far_data['case_id'].nunique() if not buildings_so_far_data.empty else 0
                
                # d) buildings_last7days: count of unique buildings covered in last 7 days from current visit_date
                buildings_last7days_data = ward_visit_data[
                    (ward_visit_data['visit_date'] >= seven_days_before) & 
                    (ward_visit_data['visit_date'] <= visit_date) & 
                    (ward_visit_data['du_status'] == 'completed')
                ]
                # Count unique case_ids (buildings/DUs), not visit count
                buildings_last7days = buildings_last7days_data['case_id'].nunique() if not buildings_last7days_data.empty else 0
                
                # e) dus_so_far: count of unique DUs covered on or before current visit_date
                dus_so_far_data = ward_visit_data[
                    (ward_visit_data['visit_date'] <= visit_date) & 
                    (ward_visit_data['du_status'] == 'completed')
                ]
                # Count unique case_ids (DUs), not visit count
                dus_so_far = dus_so_far_data['case_id'].nunique() if not dus_so_far_data.empty else 0
                
                # f) dus_last7days: count of unique DUs covered in last 7 days from current visit_date
                dus_last7days_data = ward_visit_data[
                    (ward_visit_data['visit_date'] >= seven_days_before) & 
                    (ward_visit_data['visit_date'] <= visit_date) & 
                    (ward_visit_data['du_status'] == 'completed')
                ]
                # Count unique case_ids (DUs), not visit count
                dus_last7days = dus_last7days_data['case_id'].nunique() if not dus_last7days_data.empty else 0
                
                # Get targets from ward_level_df for percentage calculations
                ward_targets = ward_level_df[
                    (ward_level_df['domain'] == domain) & 
                    (ward_level_df['ward'] == ward)
                ]
                

                if not ward_targets.empty:
                    visit_target = ward_targets['visit_target'].iloc[0]
                    du_target = ward_targets['du_target'].iloc[0]
                    building_target = ward_targets['building_target'].iloc[0]
                    
                    # Calculate percentage completions
                    pct_visits_completed = (visits_so_far / visit_target * 100) if visit_target > 0 else 0
                    pct_visits_completed_last7days = (visits_last7days / visit_target * 100) if visit_target > 0 else 0
                    pct_dus_completed = (dus_so_far / du_target * 100) if du_target > 0 else 0
                    pct_dus_completed_last7days = (dus_last7days / du_target * 100) if du_target > 0 else 0
                    pct_buildings_completed = (buildings_so_far / building_target * 100) if building_target > 0 else 0
                    pct_buildings_completed_last7days = (buildings_last7days / building_target * 100) if building_target > 0 else 0
                     
                     # Calculate microplanning completion rates
                    building_microplanning_completion_rate = (pct_buildings_completed / pct_visits_completed * 100) if pct_visits_completed > 0 else 0
                    du_microplanning_completion_rate = (pct_dus_completed / pct_visits_completed * 100) if pct_visits_completed > 0 else 0
                    building_microplanning_completion_rate_last7days = (pct_buildings_completed_last7days / pct_visits_completed_last7days * 100) if pct_visits_completed_last7days > 0 else 0
                    du_microplanning_completion_rate_last7days = (pct_dus_completed_last7days / pct_visits_completed_last7days * 100) if pct_visits_completed_last7days > 0 else 0
                    
                    # Debug logging for extreme values (optional - remove after testing)
                    if building_microplanning_completion_rate > 1000 or du_microplanning_completion_rate > 1000:
                        print(f"DEBUG - {domain}/{ward}/{visit_date}:")
                        print(f"  Visits: {visits_so_far}/{visit_target} ({pct_visits_completed:.1f}%)")
                        print(f"  Buildings: {buildings_so_far}/{building_target} ({pct_buildings_completed:.1f}%)")
                        print(f"  DUs: {dus_so_far}/{du_target} ({pct_dus_completed:.1f}%)")
                        print(f"  Building Rate: {building_microplanning_completion_rate:.1f}%")
                        print(f"  DU Rate: {du_microplanning_completion_rate:.1f}%")
                else:
                    # Set to 0 if no targets found
                    pct_visits_completed = 0
                    pct_visits_completed_last7days = 0
                    pct_dus_completed = 0
                    pct_dus_completed_last7days = 0
                    pct_buildings_completed = 0
                    pct_buildings_completed_last7days = 0
                    building_microplanning_completion_rate = 0
                    du_microplanning_completion_rate = 0
                    building_microplanning_completion_rate_last7days = 0
                    du_microplanning_completion_rate_last7days = 0
                
                # Create row for timeline
                timeline_row = {
                     'visit_date': visit_date,
                     'domain': domain,
                     'ward': ward,
                     'visits_so_far': visits_so_far,
                     'buildings_so_far': buildings_so_far,
                     'dus_so_far': dus_so_far,
                     'visits_last7days': visits_last7days,
                     'buildings_last7days': buildings_last7days,
                     'dus_last_7days': dus_last7days,
                     'pct_visits_completed': pct_visits_completed,
                     'pct_visits_completed_last7days': pct_visits_completed_last7days,
                     'pct_dus_completed': pct_dus_completed,
                     'pct_dus_completed_last7days': pct_dus_completed_last7days,
                     'pct_buildings_completed': pct_buildings_completed,
                     'pct_buildings_completed_last7days': pct_buildings_completed_last7days,
                     'building_microplanning_completion_rate': building_microplanning_completion_rate,
                     'du_microplanning_completion_rate': du_microplanning_completion_rate,
                     'building_microplanning_completion_rate_last7days': building_microplanning_completion_rate_last7days,
                     'du_microplanning_completion_rate_last7days': du_microplanning_completion_rate_last7days
                 }
                
                timeline_rows.append(timeline_row)
    
    # Create DataFrame from collected rows
    timeline_df = pd.DataFrame(timeline_rows)
    
    # Sort by visit_date, domain, and ward
    if not timeline_df.empty:
        timeline_df = timeline_df.sort_values(['visit_date', 'domain', 'ward']).reset_index(drop=True)
        
        # Round all pct_ columns to two decimal places
        pct_cols = [col for col in timeline_df.columns if col.startswith('pct_')]
        timeline_df.loc[:, pct_cols] = timeline_df[pct_cols].round(2)
    
    # Save to Excel
    output_as_excel_in_downloads(timeline_df, "timeline_based_status_report")
    
    print(f"Timeline report generated with {len(timeline_df)} rows")
    return timeline_df


if __name__ == "__main__":
    main()