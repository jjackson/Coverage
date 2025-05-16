import os
import sys
import pytest
import pandas as pd
import tempfile
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils_data_loader import load_excel_data, load_csv_data, load_coverage_data, fetch_delivery_units


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
    
    def test_fetch_delivery_units_real_api(self):
        """
        Test the fetch_delivery_units function with real API calls.
        
        This test is skipped by default as it requires:
        1. Real CommCare credentials
        2. Internet connection
        3. API access to the CommCare server
        
        To run this test:
        1. Set the environment variables:
           - COMMCARE_PROJECT_SPACE
           - COMMCARE_USERNAME
           - COMMCARE_API_KEY
        2. Run pytest with the --run-skipped flag:
           pytest tests/test_data_loader.py::TestDataLoader::test_fetch_delivery_units_real_api -v --run-skipped
        """

        """Test the fetch_delivery_units function with real API calls."""
        try:
            from dotenv import load_dotenv
            load_dotenv()  # Load environment variables from .env file
            print("Loaded .env file")
        except ImportError:
            print("python-dotenv not installed. Using environment variables directly.")
    
        # Check if environment variables are set
        project_space = os.environ.get('COMMCARE_PROJECT_SPACE')
        username = os.environ.get('COMMCARE_USERNAME')
        api_key = os.environ.get('COMMCARE_API_KEY')
        
        print(f"DEBUG: project_space={project_space}, username={username}, api_key={api_key}")


        if not all([project_space, username, api_key]):
            pytest.skip("CommCare credentials not set in environment variables")
        
        # Call the function with real credentials
        result_df = fetch_delivery_units(
            project_space=project_space,
            username=username,
            api_key=api_key,
            days_ago=30  # Fetch cases from the last 30 days
        )
        
        # Basic validation of results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= 0  # Could be zero if no cases
        
        # If we got results, verify the structure
        if len(result_df) > 0:
            assert 'caseid' in result_df.columns
            assert 'name' in result_df.columns
            assert 'service_area' in result_df.columns
            assert 'owner_name' in result_df.columns
            
            # Print some stats for manual verification
            print(f"\nRetrieved {len(result_df)} delivery units")
            print(f"Service areas: {result_df['service_area'].nunique()}")
            print(f"FLWs: {result_df['owner_name'].nunique()}")
            print(f"Status counts: {result_df['du_status'].value_counts().to_dict()}") 