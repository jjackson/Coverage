from dotenv import load_dotenv, find_dotenv
import os, constants
from src.utils import data_loader
from src.sqlqueries.sql_queries import SQL_QUERIES
import pandas as pd
from src.reports.flw_data_quality_report import FLWDataQualityReport


def main():
    dotenv_path=find_dotenv("./src/.env")
    load_dotenv(dotenv_path,override=True)
    # Cast to str to satisfy type checker
    sql = SQL_QUERIES["flw_data_quality_analysis_query"]
    superset_url = os.environ.get('SUPERSET_URL')
    superset_username = os.environ.get('SUPERSET_USERNAME')
    superset_password = os.environ.get('SUPERSET_PASSWORD')
    if superset_username is None or superset_password is None or superset_url is None:
        raise ValueError("USERNAME, PASSWORD or SUPERSET_URL must be set in the environment or .env file")
    
    chunk_size: int = constants.SQL_CHUNK_SIZE
    timeout: int = constants.API_TIMEOUT_LIMIT
    verbose: bool = True
    
    output_df = data_loader.superset_query_with_pagination_as_df(
        superset_url, sql, superset_username, superset_password, chunk_size, timeout, verbose
    )
    # Example DataFrame (replace with your actual data)
    df = output_df  


    # Output directory (string path)
    output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    # Simple logging function
    def log_func(msg):
        print(msg)

    #Class to passing default parameters
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
    report = FLWDataQualityReport(df, output_dir, log_func, params_frame)

    # Call the generate method
    output_files = report.generate()

    quality_issues_df = None
    quality_issues_df = report.excel_data.get('Quality Issues Found')

    if quality_issues_df is not None and not quality_issues_df.empty:
        output_path = output_path = os.path.join(os.path.expanduser("~"), "Downloads", "quality_issues_found_only.xlsx")
        with pd.ExcelWriter(output_path) as writer:
            quality_issues_df.to_excel(writer, sheet_name="Quality Issues Found", index=False)
            print(f"Generated file with only 'Quality Issues Found' tab: {output_path}")
    else:
        print("No 'Quality Issues Found' data to export.")



if __name__ == "__main__":
    main()
    