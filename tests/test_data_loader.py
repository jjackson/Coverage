import os
import sys
import pytest
import pandas as pd
import tempfile
from pandas.testing import assert_frame_equal

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils_data_loader import load_excel_data, load_csv_data, load_coverage_data


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
            
    # We can add a simplified test for load_coverage_data that uses our fixtures
    # A full test would require mocking the CoverageData class which is beyond the scope
    def test_load_coverage_data_file_not_found(self):
        """Test that appropriate error is raised when coverage data file is not found."""
        with pytest.raises(FileNotFoundError):
            load_coverage_data('nonexistent_file.xlsx') 