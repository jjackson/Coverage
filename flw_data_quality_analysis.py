from dotenv import load_dotenv
import os, constants
from src.utils import data_loader
from src.sqlqueries.sql_queries import SQL_QUERIES

if __name__ == "__main__":
    # Load environment variables from .env file in the root folder

 
    # Cast to str to satisfy type checker
    sql = SQL_QUERIES["flw_data_quality_analysis_query"]
    superset_url: str = "https://connect-superset.dimagi.com"
    username: str = "ayogi"
    password: str = "Aky_9164"
    chunk_size: int = constants.SQL_CHUNK_SIZE
    timeout: int = constants.API_TIMEOUT_LIMIT
    verbose: bool = True
    
    output_df = data_loader.superset_query_with_pagination_as_df(
        superset_url, sql, username, password, chunk_size, timeout, verbose
    )
    print(output_df)