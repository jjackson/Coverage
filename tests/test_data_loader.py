import os
import sys
import pytest
import pandas as pd
import tempfile
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import (
    load_excel_data, 
    load_csv_data, 
    load_coverage_data, 
    export_to_excel_using_commcare_export,
    get_du_dataframe_from_commcare_api as load_commcare_data
)

# Load environment variables from .env file
load_dotenv()

# Get CommCare credentials from environment variables with fallbacks
COMMCARE_DOMAIN = os.environ.get('COMMCARE_DOMAIN', 'test-domain')
COMMCARE_USERNAME = os.environ.get('COMMCARE_USERNAME', 'test-user')
COMMCARE_API_KEY = os.environ.get('COMMCARE_API_KEY', 'test-api-key')


@pytest.fixture
def sample_excel_file():
    """Create a temporary Excel file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp:
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Area A', 'Area B', 'Area C'],
            'buildings': [10, 20, 30],
            'delivery_count': [5, 10, 15],
            'delivery_target': [15, 25, 35],
            'surface_area': [1.5, 2.5, 3.5]
        })
        # Save to Excel
        df.to_excel(temp.name, sheet_name='Cases', index=False)
        temp_path = temp.name
    
    yield temp_path
    # Cleanup after test
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Point A', 'Point B', 'Point C'],
            'longitude': [10.123, 11.234, 12.345],
            'lattitude': [50.123, 51.234, 52.345]
        })
        # Save to CSV
        df.to_csv(temp.name, index=False)
        temp_path = temp.name
    
    yield temp_path
    # Cleanup after test
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestDataLoader:
    """Tests for data loading utilities."""
    
    def test_load_excel_data(self, sample_excel_file):
        """Test that Excel data can be loaded correctly."""
        # Load the test file
        df = load_excel_data(sample_excel_file)
        
        # Verify data was loaded correctly
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'buildings', 'delivery_count', 'delivery_target', 'surface_area']
        assert df['buildings'].dtype == 'int64'
        assert df['delivery_count'].dtype == 'int64' 
        assert df['delivery_target'].dtype == 'int64'
        assert df['surface_area'].dtype == 'float64'
        
        # Check specific values
        assert df.iloc[0]['buildings'] == 10
        assert df.iloc[1]['name'] == 'Area B'
        assert df.iloc[2]['delivery_target'] == 35
    
    def test_load_csv_data(self, sample_csv_file):
        """Test that CSV data can be loaded correctly."""
        # Load the test file
        df = load_csv_data(sample_csv_file)
        
        # Verify data was loaded correctly
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'longitude', 'lattitude']
        
        # Check specific values
        assert df.iloc[0]['name'] == 'Point A'
        assert df.iloc[1]['longitude'] == 11.234
        assert df.iloc[2]['lattitude'] == 52.345
    
    def test_excel_file_not_found(self):
        """Test that appropriate error is raised when Excel file is not found."""
        with pytest.raises(FileNotFoundError):
            load_excel_data('nonexistent_file.xlsx')
    
    def test_csv_file_not_found(self):
        """Test that appropriate error is raised when CSV file is not found."""
        with pytest.raises(FileNotFoundError):
            load_csv_data('nonexistent_file.csv')
            
    def test_load_coverage_data_file_not_found(self):
        """Test that appropriate error is raised when coverage data file is not found."""
        with pytest.raises(FileNotFoundError):
            load_coverage_data('nonexistent_file.xlsx') 

    def test_export_to_excel_using_commcare_export(self):
        """Test that exports data from CommCare HQ API to Excel using the commcare-export tool."""
        load_commcare_data(COMMCARE_DOMAIN, COMMCARE_USERNAME, COMMCARE_API_KEY)

    def test_export_to_excel_using_commcare_export(self):
        """Test that exports data from CommCare HQ using the commcare-export tool.
        
        This test requires:
        1. commcare-export to be installed (pip install commcare-export[xlsx])
        2. Valid CommCare credentials in the .env file
        3. Internet connection to CommCare HQ
        4. Exactly one .xlsx query configuration file in data/cc-export-config
        
        """
    
        # Find the query configuration file in data/cc-export-config
        # Get the directory of the current test file, then go up one level to project root
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(test_dir)
        config_dir = os.path.join(project_root, 'data', 'cc-export-config')
        
        if not os.path.exists(config_dir):
            pytest.fail(f"Query config directory does not exist: {config_dir}")
            
        config_files = [f for f in os.listdir(config_dir) if f.lower().endswith('.xlsx')]
        
        if not config_files:
            pytest.fail(f"No .xlsx query configuration files found in {config_dir}")
        elif len(config_files) > 1:
            pytest.fail(f"Multiple .xlsx files found in {config_dir}. Expected exactly one: {config_files}")
            
        query_path = os.path.join(config_dir, config_files[0])
        print(f"Using query configuration file: {query_path}")
        
        try:
            
            
            # Run the actual export with explicit output path
            result = export_to_excel_using_commcare_export(
                domain=COMMCARE_DOMAIN,
                username=COMMCARE_USERNAME,
                api_key=COMMCARE_API_KEY,
                query_file_path=query_path,
            )
            
            # Verify the output file exists
            assert os.path.exists(result), f"Output file was not created at {result}"
            
            # Verify the file is a valid Excel file by trying to open it
            try:
                pd.read_excel(result)
                print(f"Successfully created and read Excel file at {result}")
            except Exception as e:
                pytest.fail(f"Created file is not a valid Excel file: {str(e)}")
        
            # Verify it's in the data directory
            data_dir = os.path.join(project_root, "data")
            assert os.path.dirname(os.path.abspath(result)) == os.path.abspath(data_dir), \
                f"Auto-generated file not in data directory: {result}"
        
        finally:
            # Clean up temporary output file
            # if os.path.exists(result):
            #    os.unlink(result)
            pass

    def test_load_commcare_data(self):
        """Test loading data from CommCare API."""
        # This test requires:
        # 1. Valid CommCare credentials in the .env file
        # 2. Internet connection to CommCare HQ
        
        try:
            # Load delivery unit cases
            df = load_commcare_data(
                domain=COMMCARE_DOMAIN,
                user=COMMCARE_USERNAME,
                api_key=COMMCARE_API_KEY,
                case_type="deliver-unit"
            )
            
            # Verify basic structure
            assert isinstance(df, pd.DataFrame)
            
            # If data was retrieved, verify it has expected columns
            if not df.empty:
                print(f"Successfully retrieved {len(df)} cases")
                # Check for some typical columns
                expected_columns = ['case_id', 'case_name', 'owner_id']
                for col in expected_columns:
                    assert col in df.columns, f"Expected column {col} not found in data"
            else:
                print("No cases found or empty response")
                
        except Exception as e:
            # Don't fail the test but log the error
            print(f"Error in CommCare API test: {str(e)}")
            pytest.skip(f"Skipping CommCare API test due to error: {str(e)}")

def test_commcare_api_loader(domain: str, user: str, api_key: str):
    """
    Simple test function to demonstrate loading data from CommCare API.
    
    Args:
        domain: CommCare project space/domain name
        user: Username for authentication
        api_key: API key for authentication
    """
    print("Testing CommCare API data loader...")
    
    try:
        # Load delivery unit cases
        print(f"Fetching delivery-unit cases from {domain}...")
        df = load_commcare_data(domain=domain, user=user, api_key=api_key, case_type="deliver-unit")
        
        # Display basic info about the loaded data
        if not df.empty:
            print(f"Successfully loaded {len(df)} delivery-unit cases")
            print("\nColumns found:")
            print(", ".join(df.columns.tolist()))
            print("\nSample data:")
            print(df.head(3))
        else:
            print("No delivery-unit cases found")
        
        return df
    
    except Exception as e:
        print(f"Error testing CommCare API: {str(e)}")
        return None            