import os
import logging

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Get project root directory (adjust as needed)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            os.makedirs(project_root, exist_ok=True)
            log_file_path = os.path.join(project_root, 'app.log')

            # Create logger
            logger = logging.getLogger("CoverageLogger")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                file_handler = logging.FileHandler(log_file_path, mode='a')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            cls._instance = logger
        return cls._instance

# Usage:
#logger = LoggerSingleton()
#logger.info("This is a singleton logger message.")