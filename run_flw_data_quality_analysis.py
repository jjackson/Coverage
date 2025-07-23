from dotenv import load_dotenv, find_dotenv
import os, constants
import pickle
from src.utils import data_loader
from src.sqlqueries.sql_queries import SQL_QUERIES
import pandas as pd
from src.reports.flw_data_quality_report import FLWDataQualityReport
from src.org_summary import generate_summary
from src.coverage_master import load_opportunity_domain_mapping

def merging_df(df1,df2,on_str):
    if df1 is not None and not df1.empty and df2 is not None and not df2.empty :
        #making sure that flw_id in both dataframes are of type string
        df1[on_str] = df1[on_str].astype(str)
        df2[on_str] = df2[on_str].astype(str)
        merged_df = pd.merge(df1, df2, on=on_str, how='inner')
    else:
        print("Either DF1 or DF2 is empty. Cannot generate the report")
    return merged_df

def output_as_excel_in_downloads(df, file_name):
    # Output directory (string path)
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    output_path = os.path.join(downloads_dir, file_name +".xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="Merged", index=False)
    print(f"Generated file: {output_path}")

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
    
    data_quality_sql = SQL_QUERIES["flw_data_quality_analysis_query"]
    
    # Example DataFrame (replace with your actual data)
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
    summary_df = load_pickel_data_for_summary()

    #After loading/creating summary_df and quality_issues_df, join the two DF's based on flw_id
    if quality_issues_df is not None and not quality_issues_df.empty and summary_df is not None and not summary_df.empty :
        #making sure that flw_id in both dataframes are of type string
        summary_df['flw_id'] = summary_df['flw_id'].astype(str)
        quality_issues_df['flw_id'] = quality_issues_df['flw_id'].astype(str)
        merged_df = pd.merge(summary_df, quality_issues_df, on='flw_id', how='inner')
    else:
        print("No 'Quality Issues Found OR Summary Data Found'. Try running coverage first")
    
    #Last 7 days average Form Submission Time
    average_form_sbmission_sql = SQL_QUERIES["sql_fetch_average_time_form_submission_last_7_days"]
    average_form_submission_last_7_days_df = get_superset_data(average_form_sbmission_sql)
    #output_as_excel_in_downloads(average_form_submission_last_7_days_df, "average_form_submission_last_7_days_df")
    merged_df=merging_df(merged_df, average_form_submission_last_7_days_df,"flw_id")
    
    #Mapping flw_id on CCC with user_id on cchq
    flwid_hquser_id_mapping_sql = SQL_QUERIES["sql_fetch_userid_flwid_mapping"]
    flwid_hquser_id_mapping_df = get_superset_data(flwid_hquser_id_mapping_sql)
    #output_as_excel_in_downloads(flwid_hquser_id_mapping_df,"flwid_hquser_id_mapping_df")
    merged_df=merging_df(merged_df, flwid_hquser_id_mapping_df,"flw_id")
    output_as_excel_in_downloads(merged_df,"final_output_debug")


if __name__ == "__main__":
    main()
    