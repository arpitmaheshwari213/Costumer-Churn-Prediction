import pickle
from typing import Any

def save_model(model: Any, file_path: str):
    """Saves a machine learning model object using pickle."""
    print(f"--- Attempting to save model to {file_path} ---")
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save model at {file_path}: {e}")
        return False

@staticmethod
def load_model(file_path: str):
    """Loads a saved machine learning model object from pickle."""
    print(f"--- Attempting to load model from {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model from {file_path}: {e}")
        return None
