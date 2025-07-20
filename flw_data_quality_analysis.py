from dotenv import load_dotenv, find_dotenv
import os, constants
from src.utils import data_loader
from src.sqlqueries.sql_queries import SQL_QUERIES


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
    print(output_df)

if __name__ == "__main__":
    main()
    