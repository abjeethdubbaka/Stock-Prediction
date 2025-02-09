import json
import os


def load_config(config_file="config/config.json"):
    """
    Load configuration settings from a JSON file.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    with open(config_file, "r") as file:
        return json.load(file)


# Load config on import
CONFIG = load_config()
