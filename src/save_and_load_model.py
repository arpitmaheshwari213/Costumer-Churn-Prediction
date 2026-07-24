import pickle
import json
from typing import Any, Dict, List


def save_model(model: Any, file_path: str):
    """Saves a machine learning model object using pickle."""
    print(f"Attempting to save model to {file_path}")
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save model at {file_path}: {e}")
        return False


def load_model(file_path: str):
    """Loads a saved machine learning model object from pickle."""
    print(f"Attempting to load model from {file_path}")
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model from {file_path}: {e}")
        return None


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Saves a Python dictionary or list to a specified file in JSON format.
    """
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\nSuccessfully saved data to {filename}")
    except TypeError as e:
        print(f"[ERROR] Data must be serializable - {e}")
    except IOError as e:
        print(f"[ERROR] File I/O Error, Could not write to file '{filename}' - {e}")


def load_json(filename: str) -> Dict[str, Any]:
    """
    Loads data from a JSON file and returns it as a Python dictionary.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            print(f"Successfully loaded data from {filename}")
            return data
    except FileNotFoundError:
        print(f"[ERROR] The file '{filename}' was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] The file '{filename}' contains invalid JSON format.")
        return {}
    except IOError as e:
        print(f"[ERROR] File I/O Error, Could not read from file '{filename}' - {e}")
        return {}
