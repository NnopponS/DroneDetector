"""Centralized paths used across the project."""

from pathlib import Path

# Base folders
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
GPS_DATA_DIR = DATA_DIR / "gps"
VIDEOS_DIR = DATA_DIR / "videos"
MODEL_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

# Image datasets
BIRD_DIR = IMAGES_DIR / "Birds"
DRONE_DIR = IMAGES_DIR / "Drones"
TEST_PATTERN_DIR = IMAGES_DIR / "Test_Patter"

# GPS datasets
GPS_TRAIN_DIR = GPS_DATA_DIR / "P2_DATA_TRAIN"
GPS_TEST_DIR = GPS_DATA_DIR / "P2_DATA_TEST"

# Video samples
VIDEO_SAMPLE_DIR = VIDEOS_DIR / "vids"

# Model artifacts
BIRD_V_DRONE_MODEL = MODEL_DIR / "bird_v_drone_classifier.pkl"
DRONE_CLASSIFIER_MODEL = MODEL_DIR / "drone_classifier.pkl"
DRONE_CLASSIFIER_PYTORCH_MODEL = MODEL_DIR / "drone_classifier_pytorch.pkl"
GPS_MODEL_FILE = MODEL_DIR / "gps_model.pkl"
GPS_MODEL_PYTORCH_FILE = MODEL_DIR / "gps_model_pytorch.pkl"
SHAPE_CODE_MODEL_FILE = MODEL_DIR / "shape_code_model.pkl"
TRAIN_CACHE_FILE = MODEL_DIR / "train.pkl"


def ensure_structure() -> None:
    """Make sure expected folders exist (safe to call at startup)."""
    for path in [
        DATA_DIR,
        IMAGES_DIR,
        GPS_DATA_DIR,
        VIDEOS_DIR,
        MODEL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
