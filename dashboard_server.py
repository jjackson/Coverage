import os
import sys
import webbrowser
import time
import pickle
from src.flw_summary_dashboard import create_flw_dashboard

def run_server(port=8080):
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
        app = create_flw_dashboard(coverage_data_objects)
        
        # Open the browser automatically
        url = f'http://localhost:{port}'
        print(f"Opening {url} in your browser...")
        webbrowser.open(url)
        
        # Give the browser a moment to open before starting the server
        time.sleep(1)
        
        print("Server is running. Press Ctrl+C to stop.")
        app.run_server(debug=False, port=port)
        
    except Exception as e:
        print(f"Error loading dashboard: {str(e)}")

if __name__ == '__main__':
    run_server() 