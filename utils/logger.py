import logging
import os
import json
from datetime import datetime
import sys

class Logger:
    """
    Dual-channel logger: prints to console and saves to file.
    Creates a unique directory for each run to save logs, configs, and plots.
    """
    def __init__(self, base_dir="experiments"):
        # Create unique run directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Setup Logging
        self.log_file = os.path.join(self.run_dir, "execution.log")
        
        # Configure Python Logger
        self.logger = logging.getLogger("SBTS")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers if re-initialized
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # File Handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s') # Keep console clean
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.info(f"Initialized run directory: {self.run_dir}")

    def info(self, msg):
        self.logger.info(msg)
        
    def save_config(self, config_dict):
        """Saves hyperparameters to a JSON file."""
        config_path = os.path.join(self.run_dir, "hyperparameters.json")
        # Convert non-serializable objects (like classes) to str if necessary
        clean_config = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                        for k, v in config_dict.items()}
        
        with open(config_path, 'w') as f:
            json.dump(clean_config, f, indent=4)
        self.info(f"Hyperparameters saved to {config_path}")

    def get_save_path(self, filename):
        """Returns the full path to save a file in the current run directory."""
        return os.path.join(self.run_dir, filename)