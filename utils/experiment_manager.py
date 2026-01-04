"""
Experiment Manager for JD-SBTS-F

Provides robust experiment tracking, auto-saving of logs, configs, 
artifacts, and visualizations for every run.

Author: Manus AI
"""

import os
import json
import logging
import datetime
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np


class ExperimentManager:
    """
    Manages experiment runs with automatic directory creation,
    logging, config saving, and artifact management.
    
    Usage:
        exp = ExperimentManager(model_name="jd_static_lstm_sbts")
        exp.save_config(config_dict)
        exp.log("Training started...")
        exp.save_artifact("generated_paths.npy", paths)
        exp.save_model(model)
    """
    
    def __init__(
        self,
        model_name: str,
        base_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize experiment manager.
        
        Args:
            model_name: Name of the model being trained
            base_dir: Base directory for all experiments
            config: Optional config dict to save immediately
            verbose: Whether to print log messages to console
        """
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        
        # Create timestamped run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"EXP_{timestamp}_{model_name}"
        self.run_dir = self.base_dir / self.run_name
        
        # Create subdirectories
        self.artifacts_dir = self.run_dir / "artifacts"
        self.models_dir = self.run_dir / "models"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"
        
        for d in [self.artifacts_dir, self.models_dir, self.plots_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Save config if provided
        if config is not None:
            self.save_config(config)
        
        self.log(f"Initialized experiment: {self.run_name}")
        self.log(f"Run directory: {self.run_dir}")
    
    def _setup_logging(self):
        """Setup logging to both file and console."""
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        log_file = self.logs_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def log(self, message: str, level: str = "info"):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level (debug, info, warning, error, critical)
        """
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """
        Save configuration dictionary to JSON file.
        
        Args:
            config: Configuration dictionary
            filename: Output filename
        """
        config_path = self.run_dir / filename
        
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        serializable_config = json.loads(
            json.dumps(config, default=convert)
        )
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=4)
        
        self.log(f"Config saved to {config_path}")
    
    def save_artifact(
        self,
        name: str,
        data: Union[np.ndarray, Dict, Any],
        subdir: Optional[str] = None
    ):
        """
        Save an artifact (numpy array, dict, etc.).
        
        Args:
            name: Artifact name (with extension)
            data: Data to save
            subdir: Optional subdirectory within artifacts
        """
        if subdir:
            save_dir = self.artifacts_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.artifacts_dir
        
        filepath = save_dir / name
        
        if name.endswith('.npy'):
            np.save(filepath, data)
        elif name.endswith('.npz'):
            np.savez(filepath, **data) if isinstance(data, dict) else np.savez(filepath, data=data)
        elif name.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4, default=str)
        elif name.endswith('.csv'):
            if hasattr(data, 'to_csv'):
                data.to_csv(filepath)
            else:
                np.savetxt(filepath, data, delimiter=',')
        else:
            # Try pickle for unknown types
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        self.log(f"Artifact saved: {filepath}")
    
    def save_model(self, model, name: str = "model_state.pth"):
        """
        Save PyTorch model state dict.
        
        Args:
            model: PyTorch model or state dict
            name: Output filename
        """
        try:
            import torch
            filepath = self.models_dir / name
            
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), filepath)
            else:
                torch.save(model, filepath)
            
            self.log(f"Model saved: {filepath}")
        except ImportError:
            self.log("PyTorch not available, skipping model save", level="warning")
    
    def save_plot(self, fig, name: str, close: bool = True):
        """
        Save matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            name: Output filename (with extension)
            close: Whether to close figure after saving
        """
        import matplotlib.pyplot as plt
        
        filepath = self.plots_dir / name
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        self.log(f"Plot saved: {filepath}")
        
        if close:
            plt.close(fig)
    
    def get_path(self, subdir: str = "") -> Path:
        """Get path to a subdirectory in the run directory."""
        if subdir:
            path = self.run_dir / subdir
            path.mkdir(parents=True, exist_ok=True)
            return path
        return self.run_dir
    
    def save_metrics(self, metrics: Dict[str, float], filename: str = "metrics.json"):
        """
        Save metrics dictionary.
        
        Args:
            metrics: Dictionary of metric names to values
            filename: Output filename
        """
        filepath = self.run_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.log(f"Metrics saved: {filepath}")
    
    def finalize(self, summary: Optional[Dict] = None):
        """
        Finalize experiment run.
        
        Args:
            summary: Optional summary dict to save
        """
        if summary:
            self.save_artifact("summary.json", summary)
        
        self.log("=" * 50)
        self.log(f"Experiment {self.run_name} completed")
        self.log(f"Results saved to: {self.run_dir}")
        self.log("=" * 50)


class ConfigManager:
    """
    Manages configuration loading and validation.
    """
    
    DEFAULT_CONFIG = {
        # Data settings
        "use_real_data": True,
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "seq_len": 60,
        "n_assets": 5,
        
        # Model selection
        "model_name": "jd_static_lstm_sbts",
        
        # Jump detection
        "jump_threshold_std": 4.0,
        "use_neural_jumps": False,
        "neural_jump_hidden_dim": 64,
        "neural_jump_epochs": 30,
        "neural_jump_lr": 0.001,
        
        # Volatility calibration
        "vol_bandwidth": 0.5,
        
        # Feedback mechanism (JD-SBTS-F)
        "use_feedback": True,
        "feedback_kappa": 5.0,      # Mean reversion speed
        "feedback_gamma": 0.5,      # Jump impact multiplier
        
        # Drift estimation
        "drift_estimator": "lstm",  # "lstm", "transformer", "kernel"
        "lstm_hidden": 128,
        "lstm_epochs": 50,
        "lstm_lr": 0.005,
        "lstm_dropout": 0.3,
        "lstm_use_huber": True,
        
        # Kernel regression
        "kernel_bandwidth": 0.1,
        "kernel_n_pi": 10,
        
        # LightSB settings
        "lightsb_components": 20,
        "lightsb_lr": 0.005,
        "lightsb_steps": 500,
        "lightsb_min_cov_start": 0.1,
        "lightsb_min_cov_end": 0.001,
        "lightsb_anneal_epochs": 100,
        
        # Generation
        "n_gen_paths": 500,
        "n_time_steps": 60,
        
        # Evaluation
        "eval_discriminative": True,
        "eval_predictive": True,
        "eval_n_runs": 5,
        
        # Experiment
        "seed": 42,
        "device": "auto",
    }
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load config from file or return defaults.
        
        Args:
            path: Path to config file (JSON or YAML)
            
        Returns:
            Configuration dictionary
        """
        config = cls.DEFAULT_CONFIG.copy()
        
        if path is not None:
            path = Path(path)
            if path.exists():
                if path.suffix == '.json':
                    with open(path, 'r') as f:
                        loaded = json.load(f)
                elif path.suffix in ['.yaml', '.yml']:
                    try:
                        import yaml
                        with open(path, 'r') as f:
                            loaded = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError("PyYAML required for YAML config files")
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")
                
                config.update(loaded)
        
        return config
    
    @classmethod
    def save(cls, config: Dict[str, Any], path: str):
        """Save config to file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
