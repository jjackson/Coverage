#!/usr/bin/env python3
"""
Test script to demonstrate JSON flattening functionality.
"""

import pandas as pd
import json
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import flatten_json_column, load_csv_data

def test_json_flattening():
    """
    Test the JSON flattening functionality with sample data.
    """
    print("Testing JSON Flattening Functionality")
    print("=" * 50)
    
    # Sample JSON data based on the user's example
    sample_json_data = {
        "id": "646e33cd-e019-4f65-a73f-ffe1c1535b04",
        "form": {
            "case": {
                "@xmlns": "http://commcarehq.org/case/transaction/v2",
                "update": {
                    "reg_gps": "0.3946825 32.593467 1183.5 52.4",
                    "reg_date": "2025-06-10",
                    "child_DOB": "2025-06-01",
                    "child_age": "9",
                    "child_hiv": "no",
                    "child_name": "male",
                    "kmc_status": "enrolled",
                    "mother_age": "25",
                    "reg_status": "Registered",
                    "child_alive": "yes",
                    "child_gender": "Male",
                    "child_referred": "no",
                    "mother_consent": "yes",
                    "child_disability": "no",
                    "child_heart_rate": "102",
                    "child_weight_reg": "900",
                    "danger_sign_list": "High Breath Count,",
                    "child_breath_count": "90",
                    "child_immunization": "",
                    "eligibility_status": "eligible",
                    "gestational_age_lmp": "30",
                    "danger_sign_positive": "yes",
                    "medical_history_child": "no",
                    "gestational_age_preemie": "30",
                    "cause_of_premature_birth": "infections",
                    "mother_able_to_participate": "yes",
                    "time_taken_to_fill_reg_form": "00:7:46",
                    "successful_feeds_in_last_24_hours": "16"
                }
            }
        }
    }
    
    # Create a sample DataFrame with the JSON data
    sample_data = [
        {
            'id': '646e33cd-e019-4f65-a73f-ffe1c1535b04',
            'visit_date': '2025-06-10',
            'latitude': 0.3946825,
            'longitude': 32.593467,
            'form_json': json.dumps(sample_json_data)
        },
        {
            'id': 'test-id-2',
            'visit_date': '2025-06-11',
            'latitude': 0.3946826,
            'longitude': 32.593468,
            'form_json': '{"form": {"case": {"update": {"reg_gps": "0.3946826 32.593468 1184.0 53.0", "reg_date": "2025-06-11"}}}}}'
        },
        {
            'id': 'test-id-3',
            'visit_date': '2025-06-12',
            'latitude': 0.3946827,
            'longitude': 32.593469,
            'form_json': 'invalid json data'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    # Test the flatten_json_column function
    print("Testing flatten_json_column function...")
    df_flattened = flatten_json_column(df, json_column='form_json', json_path='form.case.update', prefix='update')
    
    print("\nFlattened DataFrame:")
    print(df_flattened)
    print("\n" + "="*50)
    
    # Show the new columns that were added
    original_columns = set(df.columns)
    new_columns = set(df_flattened.columns) - original_columns
    
    print("New columns added from JSON flattening:")
    for col in sorted(new_columns):
        print(f"  - {col}")
    
    print("\n" + "="*50)
    
    # Test with a CSV file (if we create one)
    print("Testing with CSV file...")
    
    # Create a temporary CSV file
    csv_file = "test_json_data.csv"
    df.to_csv(csv_file, index=False)
    
    try:
        # Test the load_csv_data function first, then flatten
        df_from_csv = load_csv_data(csv_file)
        print("DataFrame loaded from CSV (before flattening):")
        print(df_from_csv)
        
        # Now flatten the JSON
        df_flattened_from_csv = flatten_json_column(df_from_csv, json_column='form_json', json_path='form.case.update', prefix='update')
        
        print("\nDataFrame after JSON flattening:")
        print(df_flattened_from_csv)
        
        # Clean up
        os.remove(csv_file)
        
    except Exception as e:
        print(f"Error testing CSV loading: {e}")
        if os.path.exists(csv_file):
            os.remove(csv_file)
    
    print("\n" + "="*50)
    print("Test completed!")

if __name__ == "__main__":
    test_json_flattening() 