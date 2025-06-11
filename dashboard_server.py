from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess
import os
import sys
import webbrowser
import time

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/launch_org_dashboard.py':
            try:
                # Get the directory of the current script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Get the Python executable from the virtual environment
                venv_python = os.path.join(script_dir, '.venv', 'Scripts', 'python.exe')
                
                # Create a simple script to launch the dashboard
                launch_script = os.path.join(script_dir, 'launch_dashboard.py')
                with open(launch_script, 'w') as f:
                    f.write("""
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
""")
                
                # Execute the launch script
                if sys.platform == 'win32':
                    subprocess.Popen(
                        [venv_python, launch_script],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    subprocess.Popen(
                        [venv_python, launch_script],
                        start_new_session=True
                    )
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Dashboard launched successfully')
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            # Serve files from the current directory
            return SimpleHTTPRequestHandler.do_GET(self)

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, DashboardHandler)
    print("Starting dashboard server on port 8000...")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 