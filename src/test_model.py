#!/usr/bin/env python3
"""Test the trained GPS model"""

import sys
import cv2

from . import paths
from .gps_model_pytorch import GPSModelPyTorch
from .drone_detector import DroneDetector, ShapeCodeLibrary

print("Testing GPS Model")
print("=" * 60)

# Setup paths
test_dir = paths.GPS_TEST_DIR
model_path = paths.GPS_MODEL_PYTORCH_FILE

# Check if model exists
if not model_path.exists():
    print(f"ERROR: Model not found: {model_path}")
    print("Please train the model first using train_gps_simple.py")
    sys.exit(1)

# Load model
print(f"Loading model from: {model_path}")
model = GPSModelPyTorch()
if not model.load(str(model_path)):
    print("ERROR: Failed to load model")
    sys.exit(1)

print("Model loaded successfully!")

# Load shape code library
shape_cache = paths.MODEL_DIR / "shape_codes.json"
if not shape_cache.exists():
    print("ERROR: Shape code library not found. Please train first.")
    sys.exit(1)

shape_lib = ShapeCodeLibrary(cache_path=shape_cache)
shape_lib.ensure_ready()
detector = DroneDetector(shape_lib)

# Test on images
if not test_dir.exists():
    print(f"ERROR: Test directory not found: {test_dir}")
    sys.exit(1)

print(f"\nTesting on images from: {test_dir}")
print("-" * 60)

test_count = 0
detection_count = 0
results = []

for img_path in sorted(test_dir.glob("*.jpg")):
    csv_path = img_path.with_suffix(".csv")
    
    # Read true GPS if available
    true_lat = true_lon = true_alt = None
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as fp:
                lines = fp.read().strip().splitlines()
            if len(lines) >= 2:
                true_lat, true_lon, true_alt = map(float, lines[1].split(","))
        except:
            pass
    
    # Read and detect
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    
    test_count += 1
    detections, _ = detector.detect(image, debug=False, min_conf=0.12)
    
    if not detections:
        continue
    
    detection_count += 1
    det = max(detections, key=lambda d: d.area)
    
    # Predict GPS
    pred_lat, pred_lon, pred_alt = model.predict(det.center[0], det.center[1])
    
    if pred_lat is None:
        continue
    
    result = {
        'image': img_path.name,
        'pixel': (det.center[0], det.center[1]),
        'predicted': (pred_lat, pred_lon, pred_alt),
        'true': (true_lat, true_lon, true_alt) if true_lat else None
    }
    results.append(result)
    
    # Print result
    print(f"\n{img_path.name}:")
    print(f"  Pixel: ({det.center[0]:.1f}, {det.center[1]:.1f})")
    print(f"  Predicted GPS: Lat={pred_lat:.6f}, Lon={pred_lon:.6f}, Alt={pred_alt:.2f}")
    
    if true_lat is not None:
        lat_err = abs(pred_lat - true_lat)
        lon_err = abs(pred_lon - true_lon)
        alt_err = abs(pred_alt - true_alt)
        print(f"  True GPS: Lat={true_lat:.6f}, Lon={true_lon:.6f}, Alt={true_alt:.2f}")
        print(f"  Error: Lat={lat_err:.6f}, Lon={lon_err:.6f}, Alt={alt_err:.2f}")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print(f"Total images tested: {test_count}")
print(f"Detections found: {detection_count}")
print(f"Successful predictions: {len(results)}")

if results and any(r['true'] for r in results):
    import numpy as np
    lat_errors = []
    lon_errors = []
    alt_errors = []
    
    for r in results:
        if r['true']:
            lat_errors.append(abs(r['predicted'][0] - r['true'][0]))
            lon_errors.append(abs(r['predicted'][1] - r['true'][1]))
            alt_errors.append(abs(r['predicted'][2] - r['true'][2]))
    
    if lat_errors:
        print(f"\nError Statistics:")
        print(f"  Latitude - Mean: {np.mean(lat_errors):.6f}, Std: {np.std(lat_errors):.6f}, Max: {np.max(lat_errors):.6f}")
        print(f"  Longitude - Mean: {np.mean(lon_errors):.6f}, Std: {np.std(lon_errors):.6f}, Max: {np.max(lon_errors):.6f}")
        print(f"  Altitude - Mean: {np.mean(alt_errors):.2f}, Std: {np.std(alt_errors):.2f}, Max: {np.max(alt_errors):.2f}")

print("\nTest complete!")
