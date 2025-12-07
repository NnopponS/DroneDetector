"""
Drone-Only ML Classifier
ใช้เฉพาะ Drone ในการ train (One-Class Classification)
ใช้ image_process_drone pipeline และ HOG + Hough Transform features
"""

import cv2
import numpy as np
import os
import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import hog
from skimage import exposure

try:
    from . import paths
    PATHS_AVAILABLE = True
except ImportError:
    try:
        import paths
        PATHS_AVAILABLE = True
    except ImportError:
        PATHS_AVAILABLE = False
        print("Warning: paths module not available")

# Try to import PyTorch for GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False


# ============================================================
# image_process_drone Pipeline
# ============================================================

def image_process_drone(image, threshold=40):
    """
    Process image using the drone detection pipeline.
    Returns contours of detected objects.
    
    Args:
        image: Input BGR image
        threshold: Threshold value for binarization (default 40)
    
    Returns:
        List of contours that pass the filtering
    """
    if image is None:
        return []
    
    # Convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    
    # Threshold
    ret1, th1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    th1_inv = cv2.bitwise_not(th1)
    
    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated_inv = cv2.dilate(th1_inv, kernel, iterations=3)
    dilated = dilated_inv
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    y_threshold = int(0.35 * h)
    
    for c in contours:
        corner_x, corner_y, cw, ch = cv2.boundingRect(c)
        c_area = cw * ch
        
        # Apply filtering (same as original image_process_drone)
        if c_area > 50 and corner_y < h - y_threshold:
            if corner_y > 500 and ch > 150:
                continue
            if corner_x > 1300 and corner_y < 90:  # Skip time label
                continue
            
            filtered_contours.append(c)
    
    return filtered_contours


# ============================================================
# Noise Filtering Pipeline
# ============================================================

def apply_noise_filtering(image, filter_type='median', kernel_size=3):
    """Apply noise filtering to reduce salt & pepper noise."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if filter_type == 'median':
        filtered = cv2.medianBlur(gray, kernel_size)
    elif filter_type == 'gaussian':
        filtered = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    elif filter_type == 'bilateral':
        filtered = cv2.bilateralFilter(gray, kernel_size, 80, 80)
    else:
        filtered = gray
    
    return filtered


def quantize_image(image, levels=8):
    """Quantize image to reduce sample space for better histogram analysis."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    quantized = np.floor(gray / (256.0 / levels)) * (256.0 / levels)
    quantized = quantized.astype(np.uint8)
    
    return quantized


# ============================================================
# Feature Extraction: HOG
# ============================================================

def extract_hog_features(image, contour=None, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                        orientations=9, fixed_size=(64, 64), **kwargs):
    """
    Extract HOG features from image or contour region.
    Uses fixed size to ensure consistent feature vector length.
    
    Args:
        image: Full image
        contour: Contour to extract region from (if None, use full image)
        pixels_per_cell: Size of cells for HOG
        cells_per_block: Number of cells per block
        orientations: Number of orientation bins
        fixed_size: Fixed size (width, height) for ROI to ensure consistent HOG features
    
    Returns:
        HOG feature vector (fixed size)
    """
    try:
        if contour is not None:
            # Extract ROI from contour
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                return None
            
            # Resize to fixed size to ensure consistent feature vector length
            fixed_w, fixed_h = fixed_size
            roi_resized = cv2.resize(roi, (fixed_w, fixed_h))
            
            if len(roi_resized.shape) == 3:
                gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_resized
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize to fixed size if using full image
            fixed_w, fixed_h = fixed_size
            gray = cv2.resize(gray, (fixed_w, fixed_h))
        
        # Apply noise filtering and quantization
        filtered = apply_noise_filtering(gray, filter_type='median', kernel_size=3)
        quantized = quantize_image(filtered, levels=8)
        
        # Normalize for HOG
        normalized = exposure.rescale_intensity(quantized)
        
        # Extract HOG features (will be consistent size due to fixed input size)
        hog_features = hog(
            normalized,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True,
            visualize=False
        )
        
        return hog_features
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        return None


# ============================================================
# Feature Extraction: Hough Transform
# ============================================================

def extract_hough_features(image, contour=None, rho_resolution=1, theta_resolution=np.pi/180,
                          threshold=50, min_line_length=30, max_line_gap=10, fixed_size=(64, 64), **kwargs):
    """
    Extract Hough Transform features with emphasis on straight edges.
    Returns quantized edge slopes and straight edge ratio.
    Uses fixed size to ensure consistent feature vector length.
    
    Args:
        image: Full image
        contour: Contour to extract region from
        rho_resolution: Distance resolution in pixels
        theta_resolution: Angular resolution in radians
        threshold: Accumulator threshold
        min_line_length: Minimum line length
        max_line_gap: Maximum gap between line segments
        fixed_size: Fixed size (width, height) for ROI
    
    Returns:
        Feature vector with quantized slopes and straight edge ratio (fixed size: 11)
    """
    try:
        if contour is not None:
            # Extract ROI
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                return np.zeros(11)  # Return fixed-size zero vector
            
            # Resize to fixed size
            fixed_w, fixed_h = fixed_size
            roi_resized = cv2.resize(roi, (fixed_w, fixed_h))
            
            if len(roi_resized.shape) == 3:
                gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_resized
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize to fixed size if using full image
            fixed_w, fixed_h = fixed_size
            gray = cv2.resize(gray, (fixed_w, fixed_h))
        
        # Apply noise filtering
        filtered = apply_noise_filtering(gray, filter_type='median', kernel_size=3)
        
        # Edge detection (Canny)
        edges = cv2.Canny(filtered, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, rho_resolution, theta_resolution, threshold,
                                min_line_length, max_line_gap)
        
        if lines is None or len(lines) == 0:
            return np.zeros(11)  # Return fixed-size zero vector: 10 bins + 1 ratio
        
        # Extract line slopes
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)
        
        if len(slopes) == 0:
            return np.zeros(11)  # Return fixed-size zero vector
        
        # Quantize slopes into bins
        num_bins = 10
        slope_array = np.array(slopes)
        quantized_slopes = np.histogram(slope_array, bins=num_bins, range=(-5, 5))[0]
        quantized_slopes = quantized_slopes.astype(float) / len(slopes)  # Normalize
        
        # Calculate straight edge ratio (edges that form lines)
        total_edge_pixels = np.sum(edges > 0)
        straight_edge_pixels = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Approximate pixels on line
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            straight_edge_pixels += line_length
        
        straight_edge_ratio = straight_edge_pixels / total_edge_pixels if total_edge_pixels > 0 else 0.0
        
        # Combine features (always 11 elements: 10 bins + 1 ratio)
        features = np.concatenate([quantized_slopes, [straight_edge_ratio]])
        
        return features
    except Exception as e:
        print(f"Error extracting Hough features: {e}")
        return np.zeros(11)  # Return fixed-size zero vector on error


# ============================================================
# Combined Feature Extraction
# ============================================================

def extract_combined_features(image, contour, fixed_size=(64, 64)):
    """
    Extract combined features: HOG + Hough Transform
    Uses fixed size to ensure consistent feature vector length.
    
    Args:
        image: Full BGR image
        contour: Contour to extract features from
        fixed_size: Fixed size (width, height) for ROI
    
    Returns:
        Combined feature vector (fixed size)
    """
    features_list = []
    
    # Extract HOG features (with fixed size)
    hog_feat = extract_hog_features(image, contour, fixed_size=fixed_size)
    if hog_feat is not None:
        features_list.append(hog_feat)
    else:
        # If HOG fails, return None
        return None
    
    # Extract Hough features (with fixed size)
    hough_feat = extract_hough_features(image, contour, fixed_size=fixed_size)
    if hough_feat is not None:
        features_list.append(hough_feat)
    else:
        # If Hough fails, still use HOG only
        pass
    
    if len(features_list) == 0:
        return None
    
    # Concatenate all features
    combined = np.concatenate(features_list)
    
    return combined


# ============================================================
# Drone-Only Classifier
# ============================================================

class DroneOnlyClassifier:
    """
    Binary classifier that learns from both Drone and Bird images.
    Can work in one-class mode (drone only) or binary mode (drone vs bird).
    Uses Random Forest for classification.
    """
    
    def __init__(self, model_path=None, use_gpu=False):
        # Use models folder for saving
        if model_path is None:
            if PATHS_AVAILABLE:
                # Ensure models directory exists
                paths.ensure_structure()
                model_path = paths.BIRD_V_DRONE_MODEL
            else:
                model_path = os.path.join(os.getcwd(), "models", "bird_v_drone_classifier.pkl")
        # Convert Path object to string if needed
        self.model_path = str(model_path) if model_path else None
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.is_binary = False  # True if trained with both drone and bird
        self.pytorch_model = None
        
    def load_image(self, img_path):
        """Load image using secure method"""
        try:
            stream = open(img_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def process_training_image(self, img_path):
        """
        Process single training image using image_process_drone pipeline.
        Returns list of feature vectors from detected contours.
        """
        image = self.load_image(img_path)
        if image is None:
            return []
        
        # Use image_process_drone to get contours
        contours = image_process_drone(image, threshold=40)
        
        if len(contours) == 0:
            return []
        
        # Extract features from each contour (use fixed size for consistency)
        feature_vectors = []
        fixed_size = (64, 64)  # Fixed size for all ROIs
        for contour in contours:
            features = extract_combined_features(image, contour, fixed_size=fixed_size)
            if features is not None:
                feature_vectors.append(features)
        
        return feature_vectors
    
    def train(self, drone_dir, progress_callback=None):
        """
        Train classifier using only Drone images.
        
        Args:
            drone_dir: Directory containing drone images
            progress_callback: Optional callback function(current, total, message)
        
        Returns:
            True if training successful
        """
        print(f"Training Drone-Only Classifier from: {drone_dir}")
        
        # Collect all image files
        image_files = []
        for fname in os.listdir(drone_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append(os.path.join(drone_dir, fname))
        
        if len(image_files) == 0:
            print("No image files found!")
            return False
        
        print(f"Found {len(image_files)} images")
        
        # Process all images and extract features
        all_features = []
        total_contours = 0
        
        for i, img_path in enumerate(image_files):
            if progress_callback:
                progress_callback(i, len(image_files), f"Processing {os.path.basename(img_path)}")
            
            feature_vectors = self.process_training_image(img_path)
            all_features.extend(feature_vectors)
            total_contours += len(feature_vectors)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_files)} images, {total_contours} contours extracted")
        
        if len(all_features) == 0:
            print("No features extracted from images!")
            return False
        
        print(f"Total features extracted: {len(all_features)}")
        
        # Validate and filter feature vectors to ensure consistent size
        # Check feature vector sizes
        feature_sizes = [len(f) if f is not None else 0 for f in all_features]
        if len(set(feature_sizes)) > 1:
            print(f"Warning: Feature vectors have different sizes: {set(feature_sizes)}")
            print("Filtering out inconsistent feature vectors...")
            # Find the most common size
            size_counts = Counter(feature_sizes)
            expected_size = size_counts.most_common(1)[0][0]
            print(f"Expected feature size: {expected_size}")
            
            # Filter to keep only features with expected size
            filtered_features = []
            for i, feat in enumerate(all_features):
                if feat is not None and len(feat) == expected_size:
                    filtered_features.append(feat)
                else:
                    print(f"  Skipping feature {i} with size {len(feat) if feat is not None else 0}")
            
            all_features = filtered_features
            print(f"After filtering: {len(all_features)} feature vectors")
        
        if len(all_features) == 0:
            print("No valid features after filtering!")
            return False
        
        # Verify all features have the same size
        feature_size = len(all_features[0])
        for i, feat in enumerate(all_features):
            if feat is None or len(feat) != feature_size:
                print(f"Error: Feature {i} has inconsistent size: {len(feat) if feat is not None else 0} (expected {feature_size})")
                return False
        
        print(f"All feature vectors have consistent size: {feature_size}")
        
        # Convert to numpy array (now safe - all have same size)
        try:
            X = np.array(all_features, dtype=np.float64)
        except ValueError as e:
            print(f"Error converting to numpy array: {e}")
            print(f"Feature sizes: {[len(f) if f is not None else 0 for f in all_features[:10]]}")
            return False
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest (one-class: all are positive examples)
        # Use Isolation Forest for anomaly detection, or Random Forest with all positive labels
        print("Training classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # For one-class, we create labels (all are 1 = drone)
        y = np.ones(len(X_scaled))
        
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"[OK] Training complete! Trained on {len(X_scaled)} feature vectors")
        return True
    
    def train_multiple(self, drone_dirs, progress_callback=None):
        """
        Train classifier using multiple directories of Drone images.
        
        Args:
            drone_dirs: List of directories containing drone images
            progress_callback: Optional callback function(current, total, message)
        
        Returns:
            True if training successful
        """
        print(f"Training Drone-Only Classifier from {len(drone_dirs)} directories")
        
        # Collect all image files from all directories
        all_image_files = []
        for drone_dir in drone_dirs:
            if not os.path.exists(drone_dir):
                print(f"Warning: Directory not found: {drone_dir}")
                continue
            
            image_files = []
            for fname in os.listdir(drone_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_files.append(os.path.join(drone_dir, fname))
            
            all_image_files.extend(image_files)
            print(f"Found {len(image_files)} images in {drone_dir}")
        
        if len(all_image_files) == 0:
            print("No image files found in any directory!")
            return False
        
        print(f"Total images to process: {len(all_image_files)}")
        
        # Process all images and extract features
        all_features = []
        total_contours = 0
        
        for i, img_path in enumerate(all_image_files):
            if progress_callback:
                progress_callback(i, len(all_image_files), f"Processing {os.path.basename(img_path)}")
            
            feature_vectors = self.process_training_image(img_path)
            all_features.extend(feature_vectors)
            total_contours += len(feature_vectors)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(all_image_files)} images, {total_contours} contours extracted")
        
        if len(all_features) == 0:
            print("No features extracted from images!")
            return False
        
        print(f"Total features extracted: {len(all_features)}")
        
        # Validate and filter feature vectors to ensure consistent size
        # Check feature vector sizes
        feature_sizes = [len(f) if f is not None else 0 for f in all_features]
        if len(set(feature_sizes)) > 1:
            print(f"Warning: Feature vectors have different sizes: {set(feature_sizes)}")
            print("Filtering out inconsistent feature vectors...")
            # Find the most common size
            size_counts = Counter(feature_sizes)
            expected_size = size_counts.most_common(1)[0][0]
            print(f"Expected feature size: {expected_size}")
            
            # Filter to keep only features with expected size
            filtered_features = []
            for i, feat in enumerate(all_features):
                if feat is not None and len(feat) == expected_size:
                    filtered_features.append(feat)
                else:
                    print(f"  Skipping feature {i} with size {len(feat) if feat is not None else 0}")
            
            all_features = filtered_features
            print(f"After filtering: {len(all_features)} feature vectors")
        
        if len(all_features) == 0:
            print("No valid features after filtering!")
            return False
        
        # Verify all features have the same size
        feature_size = len(all_features[0])
        for i, feat in enumerate(all_features):
            if feat is None or len(feat) != feature_size:
                print(f"Error: Feature {i} has inconsistent size: {len(feat) if feat is not None else 0} (expected {feature_size})")
                return False
        
        print(f"All feature vectors have consistent size: {feature_size}")
        
        # Convert to numpy array (now safe - all have same size)
        try:
            X = np.array(all_features, dtype=np.float64)
        except ValueError as e:
            print(f"Error converting to numpy array: {e}")
            print(f"Feature sizes: {[len(f) if f is not None else 0 for f in all_features[:10]]}")
            return False
        
        # Normalize features
        if progress_callback:
            progress_callback(len(all_image_files), len(all_image_files), "Training classifier...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest (one-class: all are positive examples)
        # Use Isolation Forest for anomaly detection, or Random Forest with all positive labels
        print("Training classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # For one-class, we create labels (all are 1 = drone)
        y = np.ones(len(X_scaled))
        
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"[OK] Training complete! Trained on {len(X_scaled)} feature vectors from {len(drone_dirs)} directories")
        self.is_binary = False  # One-class mode
        return True
    
    def train_binary(self, drone_dirs, bird_dirs, progress_callback=None):
        """
        Train binary classifier using both Drone and Bird images.
        
        Args:
            drone_dirs: List of directories containing drone images
            bird_dirs: List of directories containing bird images
            progress_callback: Optional callback function(current, total, message)
        
        Returns:
            True if training successful
        """
        print(f"Training Binary Classifier (Drone vs Bird)")
        print(f"  Drone directories: {len(drone_dirs)}")
        print(f"  Bird directories: {len(bird_dirs)}")
        
        # Collect all image files from all directories
        all_drone_files = []
        for drone_dir in drone_dirs:
            if not os.path.exists(drone_dir):
                print(f"Warning: Drone directory not found: {drone_dir}")
                continue
            image_files = []
            for fname in os.listdir(drone_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_files.append(os.path.join(drone_dir, fname))
            all_drone_files.extend(image_files)
            print(f"Found {len(image_files)} drone images in {drone_dir}")
        
        all_bird_files = []
        for bird_dir in bird_dirs:
            if not os.path.exists(bird_dir):
                print(f"Warning: Bird directory not found: {bird_dir}")
                continue
            image_files = []
            for fname in os.listdir(bird_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_files.append(os.path.join(bird_dir, fname))
            all_bird_files.extend(image_files)
            print(f"Found {len(image_files)} bird images in {bird_dir}")
        
        if len(all_drone_files) == 0:
            print("No drone image files found!")
            return False
        
        if len(all_bird_files) == 0:
            print("No bird image files found!")
            return False
        
        print(f"Total: {len(all_drone_files)} drone images, {len(all_bird_files)} bird images")
        
        # Process drone images
        drone_features = []
        drone_labels = []
        total_images = len(all_drone_files) + len(all_bird_files)
        current = 0
        
        for i, img_path in enumerate(all_drone_files):
            if progress_callback:
                progress_callback(current, total_images, f"Processing drone: {os.path.basename(img_path)}")
            current += 1
            
            feature_vectors = self.process_training_image(img_path)
            drone_features.extend(feature_vectors)
            drone_labels.extend([1] * len(feature_vectors))  # 1 = Drone
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(all_drone_files)} drone images, {len(drone_features)} features")
        
        # Process bird images
        bird_features = []
        bird_labels = []
        
        for i, img_path in enumerate(all_bird_files):
            if progress_callback:
                progress_callback(current, total_images, f"Processing bird: {os.path.basename(img_path)}")
            current += 1
            
            feature_vectors = self.process_training_image(img_path)
            bird_features.extend(feature_vectors)
            bird_labels.extend([0] * len(feature_vectors))  # 0 = Bird
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(all_bird_files)} bird images, {len(bird_features)} features")
        
        # Combine features and labels
        all_features = drone_features + bird_features
        all_labels = drone_labels + bird_labels
        
        if len(all_features) == 0:
            print("No features extracted from images!")
            return False
        
        print(f"Total features: {len(all_features)} (Drones: {len(drone_features)}, Birds: {len(bird_features)})")
        
        # Validate feature sizes
        feature_sizes = [len(f) if f is not None else 0 for f in all_features]
        if len(set(feature_sizes)) > 1:
            print(f"Warning: Feature vectors have different sizes: {set(feature_sizes)}")
            expected_size = Counter(feature_sizes).most_common(1)[0][0]
            print(f"Filtering to expected size: {expected_size}")
            filtered_features = []
            filtered_labels = []
            for feat, label in zip(all_features, all_labels):
                if feat is not None and len(feat) == expected_size:
                    filtered_features.append(feat)
                    filtered_labels.append(label)
            all_features = filtered_features
            all_labels = filtered_labels
            print(f"After filtering: {len(all_features)} feature vectors")
        
        if len(all_features) == 0:
            print("No valid features after filtering!")
            return False
        
        # Convert to numpy arrays
        try:
            X = np.array(all_features, dtype=np.float64)
            y = np.array(all_labels, dtype=np.int32)
        except ValueError as e:
            print(f"Error converting to numpy array: {e}")
            return False
        
        # Normalize features
        if progress_callback:
            progress_callback(total_images, total_images, "Training classifier...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest binary classifier
        print("Training binary classifier (Drone vs Bird)...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        self.is_binary = True
        
        # Print class distribution
        drone_count = np.sum(y == 1)
        bird_count = np.sum(y == 0)
        print(f"[OK] Binary training complete!")
        print(f"  Trained on {len(X_scaled)} feature vectors")
        print(f"  Drones: {drone_count}, Birds: {bird_count}")
        return True
    
    def predict(self, contour, image):
        """
        Predict if contour is a drone.
        
        Args:
            contour: Contour to classify
            image: Full BGR image
        
        Returns:
            (is_drone: bool, confidence: float)
        """
        if not self.is_trained or self.classifier is None:
            return False, 0.0
        
        try:
            # Extract features (use fixed size for consistency)
            fixed_size = (64, 64)  # Same fixed size as training
            features = extract_combined_features(image, contour, fixed_size=fixed_size)
            if features is None:
                return False, 0.0
            
            # Normalize
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.classifier.predict(features_scaled)[0]
            proba = self.classifier.predict_proba(features_scaled)[0]
            
            if self.is_binary:
                # Binary classification: class 1 = Drone, class 0 = Bird
                is_drone = (prediction == 1)
                # Confidence is probability of being drone (class 1)
                confidence = float(proba[1] if len(proba) > 1 else 0.0)
            else:
                # One-class: prediction=1 means drone
                is_drone = (prediction == 1)
                confidence = float(proba[1] if len(proba) > 1 else proba[0])
            
            return is_drone, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0
    
    def save(self, filepath=None):
        """Save model to file"""
        if not self.is_trained:
            return False
        
        filepath = filepath or self.model_path
        
        try:
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'is_trained': True,
                'is_binary': getattr(self, 'is_binary', False)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"[OK] Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load(self, filepath=None):
        """Load model from file"""
        filepath = filepath or self.model_path
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data.get('classifier')
            self.scaler = model_data.get('scaler')
            self.is_trained = model_data.get('is_trained', False)
            self.is_binary = model_data.get('is_binary', False)
            
            if self.is_trained:
                mode = "Binary (Drone vs Bird)" if self.is_binary else "One-Class (Drone only)"
                print(f"[OK] Model loaded from {filepath} (Mode: {mode})")
            
            return self.is_trained
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def get_drone_only_classifier(use_gpu=False):
    """Factory function to create DroneOnlyClassifier"""
    return DroneOnlyClassifier(use_gpu=use_gpu)

