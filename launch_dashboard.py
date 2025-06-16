
import pickle
from src.flw_summary_dashboard import create_flw_dashboard
import webbrowser
import time

# Load coverage data from pickle file
with open('coverage_data.pkl', 'rb') as f:
    coverage_data_objects = pickle.load(f)

if coverage_data_objects:
    print(f"Launching dashboard for {len(coverage_data_objects)} opportunities...")
    # Create the dashboard app
    app = create_flw_dashboard(coverage_data_objects)
    
    # Open the browser to the dashboard URL
    webbrowser.open('http://127.0.0.1:8080')
    
    # Run the server
    app.run_server(debug=False)
else:
    print("No coverage data found. Please run coverage analysis first.")
