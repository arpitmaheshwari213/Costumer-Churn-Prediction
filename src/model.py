import pickle
from typing import Any
import logging
from log_generator import make_logger
from utils import create_timestamped_filename

class ModelUtility:
    """Utility class to handle saving and loading of models."""
    def __init__(self, logger=None):
        if(logger):
            self.logger = logger
        else:
            log_file = create_timestamped_filename(base_name="logs/model_utility",extension="log")
            self.logger = make_logger("ModelPersistence", log_file)

    @staticmethod
    def save_model(self, model: Any, file_path: str):
        """Saves a machine learning model object using pickle."""
        self.logger.info(f"--- Attempting to save model to {file_path} ---")
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            self.logger.info(f"[ERROR] Failed to save model at {file_path}: {e}")
            return False

    @staticmethod
    def load_model(self, file_path: str):
        """Loads a saved machine learning model object from pickle."""
        self.logger.info(f"--- Attempting to load model from {file_path} ---")
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            self.logger.info(f"[ERROR] Model file not found at: {file_path}")
            return None
        except Exception as e:
            self.logger.info(f"[ERROR] Failed to load model from {file_path}: {e}")
            return None
