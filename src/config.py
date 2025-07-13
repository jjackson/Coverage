import json
import os
from pathlib import Path

def get_config_path():
    """Get path to config file in data directory"""
    script_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "config.json"

def save_settings(download_file, report_type):
    """Save the last used settings"""
    config = {
        "last_download_file": download_file,
        "last_report_type": report_type
    }
    try:
        with open(get_config_path(), 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass  # Fail silently

def load_settings():
    """Load the last used settings"""
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("last_download_file", ""), config.get("last_report_type", "")
    except Exception:
        pass  # Fail silently
    
    return "", ""  # Return empty defaults