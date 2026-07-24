from datetime import datetime, timezone
import yaml
from typing import Dict, Any
import optuna


def create_timestamped_filename(base_name, extension="txt", timestamp=""):
    # Get the current date and time object
    now = datetime.now(timezone.utc)

    # Format the datetime object into a clean string
    if timestamp == "":
        timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Combine the base name and the timestamp
    full_filename = f"{base_name}_{timestamp}.{extension}"

    return full_filename


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Loads and returns configuration values from a YAML file."""
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"Successfully loaded config from {file_path}")
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error loading YAML from {file_path}: {e}")
        return {}


def build_optuna_distribution(config: Dict[str, Any]) -> Any:
    """Reads the blueprint from the YAML dictionary and constructs the correct Optuna distribution object."""

    param_type = config["type"]

    if param_type == "categorical":
        # For categorical parameters (strings)
        return optuna.distributions.CategoricalDistribution(config["values"])

    elif param_type == "float":
        # For float parameters
        return optuna.distributions.FloatDistribution(
            config["min"], config["max"], log=config.get("log_scale", False)
        )

    elif param_type == "int":
        # For integer parameters
        return optuna.distributions.IntDistribution(config["min"], config["max"])

    else:
        raise ValueError(f"Unknown parameter type found in config: {param_type}")


def load_search_space(yaml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Iterates through the yaml and created the actual Optuna distribution map."""
    final_search_space = {}
    for param_name, config in yaml_data.items():
        try:
            final_search_space[param_name] = build_optuna_distribution(config)
        except Exception as e:
            print(f"WARNING: Could not load distribution for {param_name}. Error: {e}")
    return final_search_space
