import subprocess
import sys


def install_pipreqs():
    try:
        import pipreqs
        print("pipreqs is already installed.")
    except ImportError:
        print("pipreqs not found. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pipreqs"])


required_version = (3, 12, 0)

if sys.version_info > required_version:
    print(f"Python version is greater than {required_version[0]}.{required_version[1]}.{required_version[2]}")
else:
    print(
        f"Python version is {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}, which is not greater than {required_version[0]}.{required_version[1]}.{required_version[2]}. This may lead to build issues")

# check if package pipreqs is already installed. If not install, then install it
install_pipreqs()

# Define the project path
project_path = "."

# Run pipreqs to check and install project dependencies and generate requirements.txt for reference
subprocess.run(["pipreqs", project_path, "--force"])
# Run pip install command
subprocess.run(["pip", "install", "-r", "requirements.txt"])