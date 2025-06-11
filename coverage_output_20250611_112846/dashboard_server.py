from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import os
import sys

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/launch_org_dashboard.py':
            try:
                # Get the directory of the current script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                dashboard_script = os.path.join(script_dir, 'launch_org_dashboard.py')
                
                # Get the Python executable from the virtual environment
                venv_python = os.path.join(script_dir, '.venv', 'Scripts', 'python.exe')
                
                # Execute the dashboard script using the virtual environment's Python
                subprocess.Popen([venv_python, dashboard_script])
                
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
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        self.do_GET()  # Handle POST the same way as GET

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, DashboardHandler)
    print("Starting dashboard server on port 8000...")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 