# DroneDetector

A comprehensive drone detection application with machine learning capabilities, featuring both GUI and command-line interfaces for detecting drones in images and videos, distinguishing them from birds, and estimating GPS coordinates.

## Features

- **Image Detection**: Detect drones in still images using multiple detection modes
- **Video Processing**: Real-time drone tracking and trajectory visualization in video files
- **Machine Learning Models**: 
  - Drone vs Bird classification using ML models
  - Shape-code based classification using Hu moments and contour features
  - GPS/Geolocation regression for estimating drone coordinates
- **Multiple Detection Modes**:
  - Logic-based detection (rule-based filtering)
  - ML-based detection (machine learning classifiers)
- **Video Tracking**: Track multiple drones across frames with speed estimation
- **GUI Interface**: User-friendly graphical interface built with Tkinter
- **Model Training**: Tools for training custom classification models

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

### Key Dependencies

- `opencv-python>=4.8.0` - Image and video processing
- `numpy>=1.24.0` - Numerical operations
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `Pillow>=10.0.0` - Image handling
- `matplotlib>=3.7.0` - Plotting and visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DroneDetector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Tesseract OCR if using OCR features:
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

### GUI Application

Launch the graphical user interface:

```bash
python run_gui.py
```

The GUI provides:
- Image and video file selection
- Real-time detection visualization
- Parameter adjustment controls
- Model selection and training options
- GPS coordinate display
- Video playback with tracking visualization

### Command Line Interface

The application also supports command-line usage through `src/drone_detector.py`:

```bash
# Build shape codes from training data
python src/drone_detector.py build-shape

# Train GPS regression model
python src/drone_detector.py train-geo

# Detect drones in an image
python src/drone_detector.py detect-image <image_path>

# Detect and track drones in a video
python src/drone_detector.py detect-video <video_path>
```

## Project Structure

```
DroneDetector/
├── src/
│   ├── drone_detector_gui.py      # Main GUI application
│   ├── drone_detector.py          # Core detection logic
│   ├── drone_tracker.py           # Video tracking functionality
│   ├── gps_model.py               # GPS regression model
│   ├── image_processing.py        # Image processing utilities
│   ├── video_processing.py        # Video processing utilities
│   ├── bird_v_drone_ml.py        # Bird vs Drone ML classifier
│   ├── drone_only_ml.py          # Drone-only ML classifier
│   ├── shape_code_generator.py    # Shape code generation
│   ├── config_manager.py         # Configuration management
│   └── paths.py                  # Path management
├── run_gui.py                    # GUI launcher
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Detection Modes

### Logic-Based Detection
Uses rule-based filtering with:
- Threshold-based segmentation
- Contour analysis
- Aspect ratio filtering
- Area constraints
- Horizon detection

### ML-Based Detection
Uses trained machine learning models:
- **Drone-Only ML**: Classifier using image processing features, HOG, and Hough transforms
- **Bird vs Drone ML**: Binary classifier to distinguish drones from birds
- **Shape Code Classifier**: Hu moments and contour feature-based classification

## Training Models

### Train Bird vs Drone Classifier
```bash
python src/train_test_drone_bird.py
```

### Train GPS Regression Model
```bash
python src/train_and_test_gps_model.py
```

### Train Shape Code Model
```bash
python src/train_shape_code_model.py
```

## Configuration

The application uses a configuration manager to store:
- Training data folder paths (Drones, Birds, GPS training data)
- Detection parameters
- Model preferences
- User interface settings

## Notes

- Ensure training data is organized in appropriate folders (Drones/, Birds/, P2_DATA_TRAIN/)
- Models are loaded automatically if available in the expected locations
- The application supports both Windows and Linux platforms
- Video processing may require significant computational resources for high-resolution videos

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Authors

[Add author information]
