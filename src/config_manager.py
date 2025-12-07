"""
Configuration manager for storing and loading user preferences.
Stores folder paths and model preferences.
"""

import json
import os
from pathlib import Path

try:
    from . import paths
except ImportError:
    import paths


CONFIG_FILE = paths.MODEL_DIR / "config.json"


def load_config():
    """Load configuration from file"""
    config = {
        'drone_folders': [],
        'bird_folders': [],
        'gps_train_folder': None,
        'gps_test_folder': None,
        'last_model_paths': {}
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Update with loaded values, keeping defaults for missing keys
                config.update(loaded_config)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    return config


def save_config(config):
    """Save configuration to file"""
    try:
        # Ensure models directory exists
        paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def update_config(**kwargs):
    """Update specific config values"""
    config = load_config()
    config.update(kwargs)
    save_config(config)
    return config


def get_drone_folders():
    """Get saved drone folders"""
    config = load_config()
    return config.get('drone_folders', [])


def get_bird_folders():
    """Get saved bird folders"""
    config = load_config()
    return config.get('bird_folders', [])


def get_gps_train_folder():
    """Get saved GPS train folder"""
    config = load_config()
    return config.get('gps_train_folder')


def get_gps_test_folder():
    """Get saved GPS test folder"""
    config = load_config()
    return config.get('gps_test_folder')


def set_drone_folders(folders):
    """Save drone folders"""
    update_config(drone_folders=folders)


def set_bird_folders(folders):
    """Save bird folders"""
    update_config(bird_folders=folders)


def set_gps_train_folder(folder):
    """Save GPS train folder"""
    update_config(gps_train_folder=folder)


def set_gps_test_folder(folder):
    """Save GPS test folder"""
    update_config(gps_test_folder=folder)

