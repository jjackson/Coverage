from dotenv import load_dotenv, find_dotenv
import os, constants
from datetime import date
import numpy
import pickle
from src.utils import data_loader
from src.sqlqueries.sql_queries import SQL_QUERIES
import pandas as pd
from src.reports.flw_data_quality_report import FLWDataQualityReport
from src.org_summary import generate_summary
from src.coverage_master import load_opportunity_domain_mapping
from datetime import datetime, timedelta
import pytz
from types import SimpleNamespace
from functools import reduce

# --- Constants ---
FLW_ID = 'flw_id'
CCHQ_USER_ID = 'cchq_user_id'
LAST_MODIFIED = 'last_modified'
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")


def output_as_excel_in_downloads(df, file_name):
    if(df is not None and not df.empty):
        output_path = os.path.join(DOWNLOADS_DIR, file_name +".xlsx")
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name="Merged", index=False)
        print(f"Generated file: {output_path}")
    else : 
        print("DF is either empty or Null")

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

def load_pickel_data_for_summary():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load coverage data from pickle file
    pickle_path = os.path.join(script_dir, 'coverage_data.pkl')
    
    if not os.path.exists(pickle_path):
        print("Error: coverage_data.pkl not found. Please run run_coverage.py first.")
        return

    try:
        with open(pickle_path, 'rb') as f:
            coverage_data_objects = pickle.load(f)

        # Create the dashboard app
        summary_df, topline_stats = generate_summary(coverage_data_objects, group_by='flw')

        opportunity_to_domain_mapping = load_opportunity_domain_mapping()
        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame(summary_df)
        valid_opportunities = list(opportunity_to_domain_mapping.values())
        #filter summary_df to contain datqa only about opps present in .env
        summary_df = summary_df[summary_df['opportunity'].isin(valid_opportunities)]
        
        # #export to excel
        # if summary_df is not None and not summary_df.empty:
        #     output_path = os.path.join(downloads_dir, "summary_df_only.xlsx")
        #     with pd.ExcelWriter(output_path) as writer:
        #         summary_df.to_excel(writer, sheet_name="Summary", index=False)
        #     print(f"Generated file: {output_path}")
        
    except Exception as e:
        print(f"Error : {str(e)}")
    return summary_df

def main():

    #fetching opps username details from superset
    print("Starting the process...")
    opps_username_details_sql =  SQL_QUERIES["opp_user_details_mapping"]
    final_df = get_superset_data(opps_username_details_sql) 
    opps_username_df = final_df.copy(deep=True)

    final_df['flw_id'] = final_df['flw_id'].astype(str)  # converting flw_id column values as string

    #fetching data quality details from superset
    data_quality_sql = SQL_QUERIES["flw_data_quality_analysis_query"]
    
    # get data for data quality
    data_quality_df = get_superset_data(data_quality_sql)  

    # Simple logging function
    def log_func(msg):
        print(msg)

    #Class to pass default parameters
    class ParamsFrame:
        def __init__(self):
            # Set your desired parameters here
            self.batch_size_var = "300"
            self.min_size_var = "100"
            self.gender_var = True
            self.muac_var = True
            self.age_var = True
            self.flw_ages_var = True
            self.opps_ages_var = True
            self.flw_muac_var = True
            self.opps_muac_var = True
            self.include_tampering_var = False
            self.correlation_var = False
            self.export_csv_var = True

    #create params_frame object to default the settings
    params_frame = ParamsFrame()

    # Create the report object
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    report = FLWDataQualityReport(data_quality_df, downloads_dir, log_func, params_frame)

    # Call the generate method
    output_files = report.generate()
    quality_issues_df = report.excel_data.get('Quality Issues Found')

    if quality_issues_df is not None and not quality_issues_df.empty:
        quality_issues_df['flw_id'] = quality_issues_df['flw_id'].astype(str)
    else:
        print("No 'Quality Issues Found  Try running coverage first")

    summary_df = load_pickel_data_for_summary()
    if summary_df is not None and not summary_df.empty :
        #making sure that flw_id in both dataframes are of type string
        summary_df['flw_id'] = summary_df['flw_id'].astype(str)  
    else:
        print("No 'Summary Data Found'. Try running coverage first")
    
    #After loading/creating summary_df and quality_issues_df, join the two DF's based on flw_id
    final_df = pd.merge(summary_df, quality_issues_df, on="flw_id",how="outer")
    

    #merge user_id_df with final_df to get cchq_user_id
    user_id_df = opps_username_df[['flw_id','cchq_user_id']]
    #making sure that flw_id in both user_id_df is of type string
    user_id_df['flw_id'] = user_id_df['flw_id'].astype(str)
    final_df = pd.merge(final_df,user_id_df,on="flw_id", how="left")

    #Read pickel file for each of the domain on .env file 
    #and append the datasource with each project specific dataframe
    opportunity_to_domain_mapping = load_opportunity_domain_mapping()
    valid_opportunities = list(opportunity_to_domain_mapping.values())
    
    overall_domain_df = pd.DataFrame()
    for domain in valid_opportunities:
        domain_df = pd.DataFrame()
         # Get the directory of the current script
        _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
        # Load coverage data from pickle file for each domain
        pickle_file_name = domain + ".pkl"
        pickle_path = os.path.join(_dir, pickle_file_name)
        if not os.path.exists(pickle_path):
            print("Error: coverage_data.pkl not found. Please run run_coverage.py first.")
            return
        
        try:
            with open(pickle_path, 'rb') as f:
                service_df = pickle.load(f)
        except Exception as e:
            print(f"Error : {str(e)}")

        if service_df is not None and not service_df.empty :
            
            #attach flw_id based on owner_id for ease on joining on flw_id later
            service_df.rename(columns={'owner_id': 'cchq_user_id'}, inplace=True)
            
            #forced closure merging
            forced_du_closure_df = set_forced_du_closure (service_df)
            domain_df = forced_du_closure_df

            #DU's with Camping
            camping_df = set_camping(service_df)
            domain_df = pd.merge(domain_df, camping_df, how='outer')
            
            #DU's with No Children merging
            dus_with_no_children_df = dus_with_no_children(service_df)
            domain_df = pd.merge(domain_df, dus_with_no_children_df, how='outer')

            #Singleton assigned
            singleton_df = set_singleton(service_df)
            domain_df = pd.merge(domain_df, singleton_df, how='outer')

            #DU's to be watched
            set_dus_tobe_watched_df = set_dus_tobe_watched(service_df)
            domain_df = pd.merge(domain_df, set_dus_tobe_watched_df, how='outer')

            #Last 7 days average Form Submission Time
            average_form_submission_sql = SQL_QUERIES["sql_fetch_average_time_form_submission_last_7_days"]
            average_form_submission_last_7_days_df = get_superset_data(average_form_submission_sql)
            domain_df = pd.merge(domain_df, average_form_submission_last_7_days_df, how='outer')

        else:
            print("Error: Coverage data not found. Please run run_coverage.py first.")

    overall_domain_df = pd.concat([overall_domain_df, domain_df], ignore_index=True)
    #join the two dataframes : overall_domain_df from case-data and 
    #final_df: from summary and data quality utility
    ultimate_df=pd.merge(overall_domain_df, final_df,on="cchq_user_id",how="outer" )

    #adding required additional fields
    ultimate_df['date']=date.today().strftime('%Y-%m-%d')
    ultimate_df['flagged_suspended'] = ''
    ultimate_df['action_last_weeks_fb'] = ''
    ultimate_df['email_preview'] = ''
    
    #rename appropriate columns
    ultimate_df.rename(columns={
        'total_visits_x': 'total_forms_submitted',
        'total_visits_y': 'total_forms_submitted_last7days',
    }, inplace=True)


    ultimate_df['score']=0

    print("The Final output will be downloaded into your Downloads folder with name report_flw_data_quality_analysis.xlsx...")
    output_as_excel_in_downloads(ultimate_df, "report_flw_data_quality_analysis")

def remove_empty_flws(df):
    df = df[df["flw_id"].notnull() & (df["flw_id"].astype(str).str.strip() != "")]
    df = df.dropna(subset=['flw_id'])
    return df

def set_dus_tobe_watched(df):
    df = df.copy()
    df['last_modified'] = pd.to_datetime(df['last_modified'], utc=True)
    today_utc = datetime.now(pytz.UTC)
    seven_days_ago = today_utc - timedelta(days=7)
    filtered_df = df[df['last_modified'] >= seven_days_ago].copy()
    filtered_df['ratio'] = filtered_df['delivery_count'] / filtered_df['buildings']

    # Keep only those with ratio > 10 (i.e. more than 10 deliveries per building)
    watched = filtered_df[filtered_df['ratio'] > 10]

    def format_ratio(r):
        return f"1:{int(round(r))}" if r > 0 else '1:0'

    def format_date(dt):
        return dt.strftime('%d %B %Y')

    def concat_watched(group):
        entries = [f"{format_date(row['last_modified'])}: {row['case_name']} ({format_ratio(row['ratio'])})" for _, row in group.iterrows()]
        return ", ".join(entries) if entries else None

    result = watched.groupby('cchq_user_id').apply(concat_watched).reset_index()
    result.columns = ['cchq_user_id', 'dus_to_be_watched']
    # Remove rows where dus_to_be_watched is None or empty
    result = result[result['dus_to_be_watched'].notnull() & (result['dus_to_be_watched'] != '')]
    return result


def set_singleton(df):
    # For each cchq_user_id, check if any row has buildings == 1
    def singleton_status(group):
        return 'yes' if (group['buildings'] == 1).any() else 'no'

    singleton_df = df.groupby('cchq_user_id').apply(singleton_status).reset_index()
    singleton_df.columns = ['cchq_user_id', 'has_singleton']

    # Add count of buildings for each cchq_user_id
    filtered_df = df[df['buildings'] == 1]
    buildings_count_df = filtered_df.groupby('cchq_user_id')['buildings'].count().reset_index()
    buildings_count_df.columns = ['cchq_user_id', 'singleton_count']

    # Merge the two DataFrames
    result = pd.merge(singleton_df, buildings_count_df, on='cchq_user_id', how='left')
    # Fill NaN singleton_count with 0 for users with no singleton buildings
    result['singleton_count'] = result['singleton_count'].fillna(0).astype(int)

    return result[['cchq_user_id', 'has_singleton', 'singleton_count']]


def set_camping(df):
    df = df.copy()
    df['last_modified'] = pd.to_datetime(df['last_modified'], utc=True)

    # Define analysis window
    today_utc = datetime.now(pytz.UTC)
    seven_days_ago = today_utc - timedelta(days=7)

    # Filter rows modified in the last 7 days
    filtered_df = df[df['last_modified'] >= seven_days_ago].copy()

    # Calculate delivery-to-building ratio
    filtered_df['ratio'] = filtered_df['delivery_count'] / filtered_df['buildings']

    # Assign camping status for each row
    def camping_status(r):
        if r > 20:
            return 'Camping ++'
        elif 10 < r <= 20:
            return 'Camping +'
        else:
            return ''

    filtered_df['camping'] = filtered_df['ratio'].apply(camping_status)

    # Aggregate per FLW: prioritize Camping ++ > Camping + > none
    def camping_priority(group):
        if (group['camping'] == 'Camping ++').any():
            return 'Camping ++'
        elif (group['camping'] == 'Camping +').any():
            return 'Camping +'
        else:
            return ''

    camping_status = (
    filtered_df.groupby('cchq_user_id')
    .apply(camping_priority)
    .rename('camping')
    .reset_index()
    )


    # Clean any timezone fields just in case
    for col in camping_status.select_dtypes(include=['datetimetz']).columns:
        camping_status[col] = camping_status[col].dt.tz_localize(None)

    return camping_status[['cchq_user_id', 'camping']]


# def set_camping(df):
#     df['last_modified'] = pd.to_datetime(df['last_modified'], utc=True)

#     # Current UTC time and 7-day window
#     today_utc = datetime.now(pytz.UTC)
#     seven_days_ago = today_utc - timedelta(days=7)

#     # Filter rows modified in the last 7 days
#     filtered_df = df[df['last_modified'] >= seven_days_ago].copy()

#     # Calculate the ratio as buildings / delivery_count
#     filtered_df['ratio'] = filtered_df['buildings'] / filtered_df['delivery_count']

#     # Assign camping status for each row
#     def camping_status(r):
#         if r > 0.1:
#             return 'Camping ++'
#         elif 0.05 < r <= 0.1:
#             return 'Camping +'
#         else:
#             return ''
#     filtered_df['camping'] = filtered_df['ratio'].apply(camping_status)

#     # For each FLW, set 'Camping ++' if any row is 'Camping ++', else 'Camping +' if any row is 'Camping +', else ''
#     def camping_priority(group):
#         if (group['camping'] == 'Camping ++').any():
#             return 'Camping ++'
#         elif (group['camping'] == 'Camping +').any():
#             return 'Camping +'
#         else:
#             return ''

#     camping_status = filtered_df.groupby('cchq_user_id').apply(camping_priority).reset_index(name='camping_status')
#     camping_status.columns = ['cchq_user_id', 'camping_status']

#     # Remove timezone from all datetime columns (shouldn't be any, but for safety)
#     for col in camping_status.select_dtypes(include=['datetimetz']).columns:
#         camping_status[col] = camping_status[col].dt.tz_localize(None)

#     return camping_status[['cchq_user_id', 'camping']]


def dus_with_no_children(df):
    # Convert 'modified_date' to datetime with UTC
    df['checked_out_date'] = pd.to_datetime(df['checked_out_date'], utc=True)

    # Filter for last 7 days
    today_utc = datetime.now(pytz.UTC)
    seven_days_ago = today_utc - timedelta(days=7)
    filtered_df = df[df['checked_out_date'] >= seven_days_ago].copy()

    # Calculate 'dus_with_no_children' column
    filtered_df['dus_with_no_children'] = (
        (filtered_df['delivery_count'] == 0) &
        (filtered_df['du_checkout_remark'] != 'Completed - Found Children') &
        (filtered_df['du_status'].str.lower() == 'completed'))

        #Remove timezone specific metadata on last_modified from filtered_df 
    filtered_df['checked_out_date'] = filtered_df['checked_out_date'].dt.tz_localize(None)
    
    #Grouped together by owner_id/user_id and sum of forced_du_closure to be true
    filtered_df = filtered_df.groupby('cchq_user_id')['dus_with_no_children'].sum().reset_index()
    #Rename the column for clarity
    filtered_df.rename(columns={'dus_with_no_children': 'dus_with_no_children_count'}, inplace=True)

    return filtered_df


def set_forced_du_closure(df):
    df['last_modified'] = pd.to_datetime(df['last_modified'], utc=True)

    # Current UTC time and 7-day window
    today_utc = datetime.now(pytz.UTC)
    seven_days_ago = today_utc - timedelta(days=7)

    # Filter rows modified in the last 7 days
    filtered_df = df[df['last_modified'] >= seven_days_ago]

    # Check for the correct column name
    if 'force_close_status' not in filtered_df.columns:
        # Return a DataFrame with cchq_user_id and forced_du_closure_count = 0
        if 'cchq_user_id' in filtered_df.columns:
            result = filtered_df[['cchq_user_id']].drop_duplicates().copy()
            result['forced_du_closure_count'] = 0
            return result[['cchq_user_id', 'forced_du_closure_count']]
        else:
            return pd.DataFrame(columns=['cchq_user_id', 'forced_du_closure_count'])

    # Set 'forced_du_closure' column based on 'force_close_status'
    filtered_df['forced_du_closure'] = filtered_df['force_close_status'].apply(
        lambda x: True if str(x).lower() == 'yes' else False
    )
    #Remove timezone specific metadata on last_modified from filtered_df 
    filtered_df['last_modified'] = filtered_df['last_modified'].dt.tz_localize(None)

    #Grouped together by owner_id/user_id and sum of forced_du_closure to be true
    filtered_df = filtered_df.groupby('cchq_user_id')['forced_du_closure'].sum().reset_index()
    #Rename the column for clarity
    filtered_df.rename(columns={'forced_du_closure': 'forced_du_closure_count'}, inplace=True)
    return filtered_df


if __name__ == "__main__":
    main()
    