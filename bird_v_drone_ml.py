"""
Bird vs Drone ML Classifier with Advanced Feature Extraction
Based on Bird-v-Drone repository approach:
- HOG (Histogram of Oriented Gradients)
- Hough Transform (with quantized edge slopes)
- Enhanced shape features
- Noise filtering pipeline
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import hog
from skimage import exposure
from multiprocessing import Pool, cpu_count
from functools import partial

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

# Check for OpenCV CUDA support
try:
    cv2_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
except:
    cv2_cuda_available = False


# ============================================================
# Noise Filtering Pipeline
# ============================================================

def apply_noise_filtering(image, filter_type='median', kernel_size=3):
    """
    Apply noise filtering to reduce salt & pepper noise.
    
    Args:
        image: Input grayscale image
        filter_type: 'median', 'gaussian', or 'bilateral'
        kernel_size: Size of filter kernel
    
    Returns:
        Filtered image
    """
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
    """
    Quantize image to reduce sample space for better histogram analysis.
    
    Args:
        image: Input image
        levels: Number of quantization levels (default 8)
    
    Returns:
        Quantized image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Quantize to specified levels
    quantized = np.floor(gray / (256.0 / levels)) * (256.0 / levels)
    quantized = quantized.astype(np.uint8)
    
    return quantized


# ============================================================
# GPU-Accelerated Feature Extraction
# ============================================================

if TORCH_AVAILABLE:
    def extract_hog_features_gpu(image_tensor, device, orientations=9, pixels_per_cell=(8, 8), 
                                  cells_per_block=(2, 2)):
        """
        Extract HOG features on GPU using PyTorch.
        
        Args:
            image_tensor: Grayscale image tensor on GPU (H, W)
            device: torch device
            orientations: Number of orientation bins
            pixels_per_cell: Size (width, height) of a cell
            cells_per_block: Number of cells in each block
        
        Returns:
            HOG feature vector
        """
        try:
            # Ensure image is 2D
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.squeeze(0)
            
            H, W = image_tensor.shape
            
            # Resize to minimum size
            min_size = 64
            if H < min_size or W < min_size:
                scale = max(min_size / H, min_size / W)
                new_h, new_w = int(H * scale), int(W * scale)
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0).unsqueeze(0), 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                H, W = image_tensor.shape
            
            # Compute gradients using Sobel filters
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
            
            img = image_tensor.unsqueeze(0).unsqueeze(0)
            gx = F.conv2d(img, sobel_x, padding=1).squeeze()
            gy = F.conv2d(img, sobel_y, padding=1).squeeze()
            
            # Compute magnitude and orientation
            magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
            orientation = torch.atan2(gy, gx) + np.pi  # 0 to 2π
            
            # Quantize orientations
            orientation_bins = torch.floor(orientation / (2 * np.pi / orientations)).long()
            orientation_bins = torch.clamp(orientation_bins, 0, orientations - 1)
            
            # Compute HOG features
            cell_h, cell_w = pixels_per_cell
            cells_y = H // cell_h
            cells_x = W // cell_w
            
            # Reshape into cells
            magnitude_cells = magnitude[:cells_y*cell_h, :cells_x*cell_w].view(
                cells_y, cell_h, cells_x, cell_w
            )
            orientation_cells = orientation_bins[:cells_y*cell_h, :cells_x*cell_w].view(
                cells_y, cell_h, cells_x, cell_w
            )
            
            # Compute histogram for each cell
            hog_features = []
            for cy in range(cells_y):
                for cx in range(cells_x):
                    cell_mag = magnitude_cells[cy, :, cx, :]
                    cell_ori = orientation_cells[cy, :, cx, :]
                    
                    # Histogram
                    hist = torch.zeros(orientations, device=device)
                    for o in range(orientations):
                        mask = (cell_ori == o)
                        hist[o] = cell_mag[mask].sum()
                    
                    hog_features.append(hist)
            
            # Block normalization
            block_h, block_w = cells_per_block
            blocks_y = cells_y - block_h + 1
            blocks_x = cells_x - block_w + 1
            
            if blocks_y <= 0 or blocks_x <= 0:
                # Fallback: return cell features
                return torch.cat(hog_features).cpu().numpy()
            
            # Group into blocks and normalize
            block_features = []
            for by in range(blocks_y):
                for bx in range(blocks_x):
                    block_hist = []
                    for i in range(block_h):
                        for j in range(block_w):
                            idx = (by + i) * cells_x + (bx + j)
                            block_hist.append(hog_features[idx])
                    
                    block_vec = torch.cat(block_hist)
                    # L2-Hys normalization
                    norm = torch.norm(block_vec)
                    if norm > 0:
                        block_vec = block_vec / (norm + 1e-8)
                        block_vec = torch.clamp(block_vec, 0, 0.2)  # Clip
                        norm2 = torch.norm(block_vec)
                        if norm2 > 0:
                            block_vec = block_vec / (norm2 + 1e-8)
                    block_features.append(block_vec)
            
            result = torch.cat(block_features).cpu().numpy()
            return result
            
        except Exception as e:
            print(f"GPU HOG extraction error: {e}")
            # Return zero vector
            cells_y = max(1, H // pixels_per_cell[1])
            cells_x = max(1, W // pixels_per_cell[0])
            blocks_y = max(1, cells_y - cells_per_block[0] + 1)
            blocks_x = max(1, cells_x - cells_per_block[1] + 1)
            feature_size = orientations * cells_per_block[0] * cells_per_block[1] * blocks_y * blocks_x
            return np.zeros(feature_size)


# ============================================================
# HOG Feature Extraction
# ============================================================

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False, device=None):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    HOG captures shape and texture patterns effectively.
    
    Args:
        image: Input image (BGR or grayscale)
        orientations: Number of orientation bins
        pixels_per_cell: Size (width, height) of a cell
        cells_per_block: Number of cells in each block
        visualize: If True, return HOG image visualization
    
    Returns:
        HOG feature vector (and optionally HOG image)
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to minimum size for HOG to work well
        min_size = 64
        if gray.shape[0] < min_size or gray.shape[1] < min_size:
            scale = max(min_size / gray.shape[0], min_size / gray.shape[1])
            new_h = int(gray.shape[0] * scale)
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, new_h))
        
        # Extract HOG features using skimage
        features, hog_image = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True,
            block_norm='L2-Hys'
        )
        
        if visualize:
            # Rescale HOG image for better visualization
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            return features, hog_image_rescaled
        
        return features
    
    except Exception as e:
        print(f"HOG extraction error: {e}")
        # Return zero vector if extraction fails
        # Estimate size: orientations * cells_per_block[0] * cells_per_block[1] * num_blocks
        num_blocks_h = max(1, (gray.shape[0] // pixels_per_cell[1] - cells_per_block[0] + 1))
        num_blocks_w = max(1, (gray.shape[1] // pixels_per_cell[0] - cells_per_block[1] + 1))
        feature_size = orientations * cells_per_block[0] * cells_per_block[1] * num_blocks_h * num_blocks_w
        return np.zeros(feature_size)


def extract_hog_from_roi(image, contour, padding=10, **hog_params):
    """
    Extract HOG features from a specific ROI defined by contour.
    
    Args:
        image: Full image (BGR, grayscale, or RGBA)
        contour: OpenCV contour
        padding: Padding around bounding box
        **hog_params: Parameters passed to extract_hog_features
    
    Returns:
        HOG feature vector
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add padding
    if len(image.shape) == 2:
        h_img, w_img = image.shape
    else:
        h_img, w_img = image.shape[:2]
    
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w_img - x, w + 2 * padding)
    h = min(h_img - y, h + 2 * padding)
    
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0 or w < 8 or h < 8:
        # Return minimal HOG features (zeros)
        return extract_hog_features(np.zeros((64, 64), dtype=np.uint8), **hog_params)
    
    return extract_hog_features(roi, **hog_params)


# ============================================================
# Hough Transform Feature Extraction
# ============================================================

def extract_hough_features(image, contour=None, rho_resolution=1, theta_resolution=np.pi/180,
                          threshold=50, min_line_length=10, max_line_gap=5):
    """
    Extract Hough Transform features with quantized edge slopes.
    Drones typically have more straight edges (arms, body) than birds.
    
    Args:
        image: Input image or ROI (BGR, grayscale, or RGBA)
        contour: Optional contour to extract ROI from image
        rho_resolution: Distance resolution in pixels
        theta_resolution: Angular resolution in radians
        threshold: Minimum votes for a line
        min_line_length: Minimum line length
        max_line_gap: Maximum gap between line segments
    
    Returns:
        Dictionary with Hough features:
        - num_lines: Number of detected lines
        - avg_line_length: Average line length
        - straight_edge_ratio: Ratio of straight to curved edges
        - quantized_slopes: Histogram of quantized edge slopes (8 bins)
    """
    try:
        # Extract ROI if contour provided
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            padding = 5
            if len(image.shape) == 2:
                h_img, w_img = image.shape
            else:
                h_img, w_img = image.shape[:2]
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w_img - x, w + 2 * padding)
            h = min(h_img - y, h + 2 * padding)
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                roi = image
        else:
            roi = image
        
        # Convert to grayscale - handle all formats
        if len(roi.shape) == 2:
            gray = roi.copy()  # Already grayscale
        elif roi.shape[2] == 4:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)  # RGBA
        elif roi.shape[2] == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # BGR
        else:
            gray = roi.copy()  # Fallback
        
        # Edge detection (Canny)
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough Line Transform (Standard)
        lines = cv2.HoughLinesP(
            edges,
            rho=rho_resolution,
            theta=theta_resolution,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        # Calculate features
        num_lines = 0
        line_lengths = []
        slopes = []
        
        if lines is not None:
            num_lines = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                line_lengths.append(length)
                
                # Calculate slope (quantized)
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    slopes.append(slope)
                else:
                    slopes.append(np.inf)  # Vertical line
        
        avg_line_length = np.mean(line_lengths) if line_lengths else 0.0
        
        # Quantize slopes into 8 bins (more weight on straight edges)
        # Bins: vertical, near-vertical, horizontal, near-horizontal, diagonal, etc.
        quantized_slopes = np.zeros(8)
        for slope in slopes:
            if np.isinf(slope):
                quantized_slopes[0] += 1  # Vertical
            elif abs(slope) < 0.2:
                quantized_slopes[1] += 1  # Near-horizontal
            elif abs(slope) > 5:
                quantized_slopes[2] += 1  # Near-vertical
            elif 0.7 < abs(slope) < 1.5:
                quantized_slopes[3] += 1  # Diagonal ±45°
            elif 0.2 <= abs(slope) < 0.7:
                quantized_slopes[4] += 1  # Shallow diagonal
            elif 1.5 <= abs(slope) <= 5:
                quantized_slopes[5] += 1  # Steep diagonal
            else:
                quantized_slopes[6] += 1  # Other
        
        # Normalize quantized slopes
        if num_lines > 0:
            quantized_slopes = quantized_slopes / num_lines
        
        # Calculate straight edge ratio (more lines = more straight edges)
        total_edge_pixels = np.sum(edges > 0)
        straight_edge_pixels = sum(line_lengths)
        straight_edge_ratio = straight_edge_pixels / total_edge_pixels if total_edge_pixels > 0 else 0.0
        
        return {
            'num_lines': num_lines,
            'avg_line_length': avg_line_length,
            'straight_edge_ratio': straight_edge_ratio,
            'quantized_slopes': quantized_slopes.tolist()
        }
    
    except Exception as e:
        print(f"Hough feature extraction error: {e}")
        return {
            'num_lines': 0,
            'avg_line_length': 0.0,
            'straight_edge_ratio': 0.0,
            'quantized_slopes': [0.0] * 8
        }


# ============================================================
# Multiprocessing Helper Functions (Module-level for pickling)
# ============================================================

def _process_single_image_helper(path, feature_config_dict):
    """Process a single image file - used for multiprocessing (module-level for pickling)"""
    try:
        # Load image
        stream = open(path, "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        stream.close()
        
        if img is None:
            return None
        
        # Convert to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Apply noise filtering if configured
        if feature_config_dict.get('apply_noise_filter', True):
            filtered_img = apply_noise_filtering(img, filter_type='median', kernel_size=3)
        else:
            filtered_img = img
        
        # Convert to grayscale
        if len(filtered_img.shape) == 3:
            gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = filtered_img
        
        # Edge detection and thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Use largest contour
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 50:
            return None
        
        # Extract features
        features = extract_comprehensive_features(
            c,
            img,
            use_hog=feature_config_dict.get('use_hog', True),
            use_hough=feature_config_dict.get('use_hough', True),
            use_shape=feature_config_dict.get('use_shape', True),
            apply_noise_filter=False  # Already filtered
        )
        
        return features
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def _process_single_image_wrapper(args):
    """Wrapper for multiprocessing (module-level for pickling)"""
    path, label, feature_config_dict = args
    features = _process_single_image_helper(path, feature_config_dict)
    if features is not None:
        return (features, label)
    return None


# ============================================================
# Enhanced Feature Extraction (Combining All Features)
# ============================================================

def extract_comprehensive_features(contour, image=None, use_hog=True, use_hough=True, 
                                  use_shape=True, apply_noise_filter=True):
    """
    Extract comprehensive features combining:
    - HOG features (texture and shape patterns)
    - Hough Transform features (straight edges)
    - Shape features (Hu moments, geometric properties)
    
    Args:
        contour: OpenCV contour
        image: Original image (required for HOG and Hough)
        use_hog: Include HOG features
        use_hough: Include Hough Transform features
        use_shape: Include traditional shape features
        apply_noise_filter: Apply noise filtering before feature extraction
    
    Returns:
        Combined feature vector as numpy array
    """
    features_list = []
    
    if image is None:
        # Can't extract HOG/Hough without image, use shape only
        use_hog = False
        use_hough = False
    
    # Apply noise filtering if requested
    # Ensure image is in correct format (BGR or grayscale)
    processed_image = image
    if image is not None:
        # Convert to BGR if needed for consistency
        if len(image.shape) == 2:  # Grayscale
            processed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] == 1:  # Single channel
            processed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if apply_noise_filter:
            filtered_image = apply_noise_filtering(processed_image, filter_type='median', kernel_size=3)
            quantized_image = quantize_image(filtered_image, levels=8)
        else:
            quantized_image = processed_image
    else:
        quantized_image = None
    
    # Extract shape features (from image_processing.py)
    if use_shape:
        try:
            from image_processing import extract_features as extract_shape_features
            shape_features = extract_shape_features(contour)
            if shape_features is not None:
                features_list.append(shape_features)
        except Exception as e:
            print(f"Shape feature extraction error: {e}")
    
    # Extract HOG features
    if use_hog and processed_image is not None:
        try:
            hog_features = extract_hog_from_roi(
                quantized_image if (apply_noise_filter and quantized_image is not None) else processed_image,
                contour,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )
            if hog_features is not None:
                features_list.append(hog_features)
        except Exception as e:
            print(f"HOG extraction error: {e}")
    
    # Extract Hough Transform features
    if use_hough and processed_image is not None:
        try:
            hough_features = extract_hough_features(
                quantized_image if (apply_noise_filter and quantized_image is not None) else processed_image,
                contour,
                threshold=50,
                min_line_length=10,
                max_line_gap=5
            )
            # Flatten Hough features into vector
            hough_vector = [
                hough_features['num_lines'],
                hough_features['avg_line_length'],
                hough_features['straight_edge_ratio']
            ] + hough_features['quantized_slopes']
            features_list.append(np.array(hough_vector))
        except Exception as e:
            print(f"Hough extraction error: {e}")
    
    # Combine all features
    if len(features_list) == 0:
        return None
    
    combined_features = np.concatenate(features_list)
    
    # Replace NaNs and Infs
    combined_features = np.nan_to_num(combined_features)
    
    return combined_features


# ============================================================
# PyTorch Neural Network for GPU Support
# ============================================================

if TORCH_AVAILABLE:
    class FeatureClassifierNet(nn.Module):
        """Neural Network for feature-based classification"""
        def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
            super(FeatureClassifierNet, self).__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            # Output layer (binary classification)
            layers.append(nn.Linear(prev_dim, 2))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)


# ============================================================
# Enhanced ML Classifier with HOG + Hough + Shape Features
# ============================================================

class BirdVsDroneClassifier:
    """
    Enhanced classifier using Bird-v-Drone repository approach:
    - HOG features (Histogram of Oriented Gradients)
    - Hough Transform features (quantized edge slopes)
    - Traditional shape features (Hu moments, geometric properties)
    """
    
    def __init__(self, model_path='bird_v_drone_classifier.pkl', use_gpu=True):
        self.model_path = model_path
        
        # Validate GPU request
        if use_gpu:
            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    "GPU mode requested but PyTorch is not available. "
                    "Please install PyTorch with CUDA support or use CPU mode."
                )
            
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU mode requested but CUDA is not available. "
                    "Possible reasons: No NVIDIA GPU, CUDA drivers not installed, "
                    "or PyTorch not compiled with CUDA support. "
                    "Please use CPU mode instead."
                )
            
            self.use_gpu = True
            self.device = torch.device('cuda')
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')
        
        # Initialize classifier (will be set based on use_gpu in train method)
        self.clf = None
        self.pytorch_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.expected_feature_dim = None  # Store expected feature dimension
        self.feature_config = {
            'use_hog': True,
            'use_hough': True,
            'use_shape': True,
            'apply_noise_filter': True
        }
        
        # Default to RandomForest if GPU not available
        if not self.use_gpu:
            self.clf = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
    
    def train(self, drone_dir, bird_dir, use_hog=True, use_hough=True, 
              use_shape=True, apply_noise_filter=True, progress_callback=None):
        """
        Train the enhanced classifier.
        
        Args:
            drone_dir: Directory containing drone images
            bird_dir: Directory containing bird images
            use_hog: Include HOG features
            use_hough: Include Hough Transform features
            use_shape: Include shape features
            apply_noise_filter: Apply noise filtering
            progress_callback: Optional callback function(current, total, message) for progress updates
        """
        self.feature_config = {
            'use_hog': use_hog,
            'use_hough': use_hough,
            'use_shape': use_shape,
            'apply_noise_filter': apply_noise_filter
        }
        
        X = []
        y = []
        
        print(f"Training Bird-v-Drone Enhanced Classifier...")
        print(f"  Drones: {drone_dir}")
        print(f"  Birds: {bird_dir}")
        print(f"  Features: HOG={use_hog}, Hough={use_hough}, Shape={use_shape}")
        
        # Note about GPU performance
        if self.use_gpu:
            print(f"\n⚠ Note: Feature extraction (HOG/Hough/Shape) runs on CPU and may take 70-80% of total time.")
            print(f"   GPU will accelerate only the neural network training phase (20-30% of total time).")
            print(f"   Using multiprocessing to speed up feature extraction on CPU.\n")
        
        # Count total images first for progress calculation
        def count_images(directory):
            count = 0
            if os.path.exists(directory):
                for fname in os.listdir(directory):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        count += 1
            return count
        
        total_drone_images = count_images(drone_dir)
        total_bird_images = count_images(bird_dir)
        total_images = total_drone_images + total_bird_images
        
        if progress_callback:
            progress_callback(0, total_images, "Starting image processing...")
        
        # Use multiprocessing for faster feature extraction
        num_workers = min(cpu_count(), 8)  # Use up to 8 cores
        print(f"Using {num_workers} CPU cores for parallel feature extraction...")
        
        # Process Drones (Label = 0) - SWAPPED: Drones are now class 0
        drone_count = 0
        processed_count = 0
        if os.path.exists(drone_dir):
            drone_files = [os.path.join(drone_dir, f) for f in os.listdir(drone_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            # Use multiprocessing for parallel processing
            with Pool(processes=num_workers) as pool:
                args_list = [(path, 0, self.feature_config.copy()) for path in drone_files]  # Changed: 1 -> 0
                
                # Process in chunks to update progress
                chunk_size = max(10, len(drone_files) // 20)  # Update progress ~20 times
                for i in range(0, len(args_list), chunk_size):
                    chunk = args_list[i:i+chunk_size]
                    chunk_results = pool.map(_process_single_image_wrapper, chunk)
                    
                    for result in chunk_results:
                        if result is not None:
                            features, label = result
                            X.append(features)
                            y.append(label)
                            drone_count += 1
                        processed_count += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(processed_count, total_images, 
                                        f"Processing drones: {drone_count}/{len(drone_files)} (parallel)")
                    
                    if drone_count % 50 == 0:
                        print(f"  Processed {drone_count} drone images...")
        
        # Process Birds (Label = 1) - SWAPPED: Birds are now class 1
        bird_count = 0
        if os.path.exists(bird_dir):
            bird_files = [os.path.join(bird_dir, f) for f in os.listdir(bird_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            # Use multiprocessing for parallel processing
            with Pool(processes=num_workers) as pool:
                args_list = [(path, 1, self.feature_config.copy()) for path in bird_files]  # Changed: 0 -> 1
                
                # Process in chunks to update progress
                chunk_size = max(10, len(bird_files) // 20)
                for i in range(0, len(args_list), chunk_size):
                    chunk = args_list[i:i+chunk_size]
                    chunk_results = pool.map(_process_single_image_wrapper, chunk)
                    
                    for result in chunk_results:
                        if result is not None:
                            features, label = result
                            X.append(features)
                            y.append(label)
                            bird_count += 1
                        processed_count += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(processed_count, total_images, 
                                        f"Processing birds: {bird_count}/{len(bird_files)} (parallel)")
                    
                    if bird_count % 50 == 0:
                        print(f"  Processed {bird_count} bird images...")
        
        print(f"Collected {drone_count} drone samples and {bird_count} bird samples.")
        
        if len(X) == 0:
            print("No training data found!")
            if progress_callback:
                progress_callback(total_images, total_images, "Error: No valid training data found!")
            return False
        
        # Check feature dimensions consistency
        if progress_callback:
            progress_callback(processed_count, total_images, "Validating features...")
        
        feature_dims = [len(f) for f in X]
        if len(set(feature_dims)) > 1:
            print(f"Warning: Inconsistent feature dimensions: {set(feature_dims)}")
            # Use most common dimension
            from collections import Counter
            most_common_dim = Counter(feature_dims).most_common(1)[0][0]
            X = [f for f in X if len(f) == most_common_dim]
            y = y[:len(X)]
            print(f"Using {len(X)} samples with dimension {most_common_dim}")
        
        # Store expected feature dimension
        if len(X) > 0:
            self.expected_feature_dim = len(X[0])
            print(f"Expected feature dimension: {self.expected_feature_dim}")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Fit model - use PyTorch if GPU available, otherwise RandomForest
        if self.use_gpu:
            # Double-check GPU availability before training
            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    "GPU mode requested but PyTorch is not available. "
                    "Please install PyTorch with CUDA support or use CPU mode."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU mode requested but CUDA is not available. "
                    "Please check CUDA installation or use CPU mode."
                )
            
            # Proceed with GPU training
            if progress_callback:
                progress_callback(processed_count, total_images, f"Training PyTorch classifier on {self.device.type.upper()}...")
            
            # Print GPU info
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            print(f"Training PyTorch classifier on {self.device.type.upper()}...")
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory:.2f} GB")
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            input_dim = X_scaled.shape[1]
            
            # Initialize PyTorch model - Larger model for better GPU utilization
            # Increase model size slightly to better utilize GPU
            hidden_dims = [512, 256, 128, 64] if input_dim > 500 else [256, 128, 64]
            self.pytorch_model = FeatureClassifierNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3).to(self.device)
            print(f"Model initialized with {sum(p.numel() for p in self.pytorch_model.parameters())} parameters")
            
            # Convert to tensors - Use pin_memory for faster CPU-GPU transfer
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.LongTensor(y)
            
            # Move to GPU (pin_memory helps with faster transfer)
            X_tensor = X_tensor.to(self.device, non_blocking=True)
            y_tensor = y_tensor.to(self.device, non_blocking=True)
            
            print(f"Data transferred to {self.device.type.upper()}: {len(X_tensor)} samples")
            
            # Training parameters - Optimized for GPU
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.pytorch_model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            # Training loop - Optimized batch size for GPU
            epochs = 100
            # Use larger batch size for GPU to maximize utilization
            # Adjust based on available GPU memory
            batch_size = min(256, len(X_tensor))  # Larger batch for GPU, but not larger than dataset
            if batch_size < 32:
                batch_size = len(X_tensor)  # Use full batch if dataset is small
            
            print(f"Using batch size: {batch_size} for GPU training")
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            # Enable cuDNN benchmarking for faster training (if available)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                print("cuDNN benchmarking enabled for faster training")
            
            for epoch in range(epochs):
                self.pytorch_model.train()
                total_loss = 0.0
                num_batches = 0
                
                # Shuffle data each epoch for better training
                indices = torch.randperm(len(X_tensor), device=self.device)
                X_shuffled = X_tensor[indices]
                y_shuffled = y_tensor[indices]
                
                # Mini-batch training with optimized batch size
                for i in range(0, len(X_shuffled), batch_size):
                    batch_X = X_shuffled[i:i+batch_size]
                    batch_y = y_shuffled[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.pytorch_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.pytorch_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else total_loss
                scheduler.step(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    # Show GPU memory usage if available
                    if torch.cuda.is_available():
                        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, GPU Memory: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB")
                    else:
                        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                    if progress_callback:
                        progress_callback(processed_count, total_images, f"Training... Epoch {epoch+1}/{epochs}")
            
            self.is_trained = True
            print(f"✓ PyTorch model trained on {self.device.type.upper()}")
        else:
            # Use RandomForest (CPU)
            if progress_callback:
                progress_callback(processed_count, total_images, "Training RandomForest classifier...")
            print("Training RandomForest classifier...")
            
            if self.clf is None:
                self.clf = make_pipeline(
                    StandardScaler(),
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            
            self.clf.fit(X, y)
            self.is_trained = True
            print("✓ RandomForest model trained")
        
        # Save model
        if progress_callback:
            progress_callback(processed_count, total_images, "Saving model...")
        self.save()
        
        if progress_callback:
            progress_callback(total_images, total_images, "Training Complete!")
        
        print("✓ Model trained and saved.")
        return True
    
    def _process_image_file(self, path):
        """Helper to load image and extract comprehensive features from largest contour"""
        try:
            # Handle unicode paths
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
            
            if img is None:
                return None
            
            # Convert to BGR if grayscale (1 channel) or RGBA (4 channels)
            if len(img.shape) == 2:  # Grayscale (1 channel)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA (4 channels)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:  # Single channel
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Apply noise filtering if configured
            if self.feature_config['apply_noise_filter']:
                filtered_img = apply_noise_filtering(img, filter_type='median', kernel_size=3)
            else:
                filtered_img = img
            
            # Convert to grayscale for processing
            if len(filtered_img.shape) == 3:
                gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = filtered_img
            
            # Edge detection and thresholding to find object
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphology to close gaps
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Use largest contour as the object
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 50:  # Too small
                return None
            
            # Extract comprehensive features
            features = extract_comprehensive_features(
                c,
                img,  # Use original image for better HOG/Hough
                use_hog=self.feature_config['use_hog'],
                use_hough=self.feature_config['use_hough'],
                use_shape=self.feature_config['use_shape'],
                apply_noise_filter=False  # Already filtered above
            )
            
            return features
        
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None
    
    def predict(self, contour, image=None):
        """
        Predict if a contour is a drone (True) or Bird (False).
        
        Args:
            contour: OpenCV contour
            image: Original image (required for HOG/Hough features)
        
        Returns:
            (is_drone: bool, drone_probability: float)
        """
        if not self.is_trained:
            print("Warning: Model not trained, cannot predict")
            return False, 0.0
        
        # Check if image is needed but not provided
        if image is None and (self.feature_config['use_hog'] or self.feature_config['use_hough']):
            print("Warning: Image required for HOG/Hough features, using shape features only")
            # Fallback to shape-only if image not available
            temp_config = self.feature_config.copy()
            temp_config['use_hog'] = False
            temp_config['use_hough'] = False
            
            features = extract_comprehensive_features(
                contour,
                image,
                use_hog=False,
                use_hough=False,
                use_shape=self.feature_config['use_shape'],
                apply_noise_filter=self.feature_config['apply_noise_filter']
            )
        else:
            # Use configured features
            features = extract_comprehensive_features(
                contour,
                image,
                use_hog=self.feature_config['use_hog'],
                use_hough=self.feature_config['use_hough'],
                use_shape=self.feature_config['use_shape'],
                apply_noise_filter=self.feature_config['apply_noise_filter']
            )
        
        if features is None:
            return False, 0.0
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        feature_dim = features.shape[1]
        
        # Check dimension match with expected dimension BEFORE scaling
        # Also check scaler dimension if available
        scaler_expected_dim = None
        if hasattr(self.scaler, 'n_features_in_'):
            scaler_expected_dim = self.scaler.n_features_in_
        
        # Use scaler dimension as primary source of truth if available
        target_dim = scaler_expected_dim if scaler_expected_dim is not None else self.expected_feature_dim
        
        if target_dim is not None and feature_dim != target_dim:
            print(f"⚠ Feature dimension mismatch: got {feature_dim}, expected {target_dim}")
            if scaler_expected_dim:
                print(f"   (from scaler.n_features_in_)")
            elif self.expected_feature_dim:
                print(f"   (from expected_feature_dim)")
            print(f"   This usually means feature extraction config differs from training.")
            print(f"   Training config: {self.feature_config}")
            print(f"   Current feature extraction may be using different settings.")
            
            # Try to handle dimension mismatch
            if feature_dim < target_dim:
                # Pad with zeros (mean value padding might be better, but zeros is safer)
                padding = np.zeros((1, target_dim - feature_dim))
                features = np.hstack([features, padding])
                print(f"   ✓ Padded features from {feature_dim} to {features.shape[1]} dimensions")
            elif feature_dim > target_dim:
                # Truncate (take first N features)
                features = features[:, :target_dim]
                print(f"   ✓ Truncated features from {feature_dim} to {features.shape[1]} dimensions")
            else:
                # Should not happen, but return error
                print(f"   ✗ Cannot fix dimension mismatch, returning False")
                return False, 0.0
        
        # Verify dimension is correct now
        if target_dim is not None and features.shape[1] != target_dim:
            print(f"✗ Feature dimension still incorrect after fix: {features.shape[1]} != {target_dim}")
            return False, 0.0
        
        try:
            if self.use_gpu and self.pytorch_model is not None:
                # PyTorch prediction
                self.pytorch_model.eval()
                with torch.no_grad():
                    # Scale features - this should work now after dimension fix
                    try:
                        features_scaled = self.scaler.transform(features)
                    except Exception as scaler_error:
                        print(f"✗ Scaler error: {scaler_error}")
                        print(f"   Feature shape: {features.shape}, Scaler expects: {self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 'unknown'}")
                        return False, 0.0
                    
                    features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                    outputs = self.pytorch_model(features_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    # SWAPPED: Class 0 = Drone, Class 1 = Bird (as per training: y.append(0) for drones, y.append(1) for birds)
                    drone_prob = probs[0] if len(probs) > 0 else 0.0  # Class 0 = Drone
                    bird_prob = probs[1] if len(probs) > 1 else 0.0    # Class 1 = Bird
                    print(f"✓ PyTorch prediction: Drone={drone_prob:.3f}, Bird={bird_prob:.3f}")
            else:
                # RandomForest prediction
                try:
                    # For RandomForest in pipeline, features go directly (scaler is inside pipeline)
                    probs = self.clf.predict_proba(features)[0]
                except Exception as rf_error:
                    print(f"✗ RandomForest prediction error: {rf_error}")
                    # Try with scaler if pipeline doesn't have it
                    if hasattr(self, 'scaler') and self.scaler is not None:
                        try:
                            features_scaled = self.scaler.transform(features)
                            # If clf is not a pipeline, we need to scale manually
                            if hasattr(self.clf, 'named_steps'):
                                probs = self.clf.predict_proba(features_scaled)[0]
                            else:
                                probs = self.clf.predict_proba(features_scaled)[0]
                        except:
                            print(f"✗ Cannot fix RandomForest prediction, returning False")
                            return False, 0.0
                    else:
                        return False, 0.0
                
                # SWAPPED: Class 0 = Drone, Class 1 = Bird (as per training)
                drone_prob = probs[0] if len(probs) > 0 else 0.0  # Class 0 = Drone
                bird_prob = probs[1] if len(probs) > 1 else 0.0  # Class 1 = Bird
                print(f"✓ RandomForest prediction: Drone={drone_prob:.3f}, Bird={bird_prob:.3f}")
            
            # Return: (is_drone: bool, drone_probability: float)
            # If drone_prob > 0.5, it's more likely a drone than a bird
            is_drone = drone_prob > 0.5
            return is_drone, drone_prob
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def save(self, filepath=None):
        """Save model and configuration"""
        filepath = filepath or self.model_path
        
        try:
            model_data = {
                'use_gpu': self.use_gpu,
                'is_trained': self.is_trained,
                'feature_config': self.feature_config,
                'scaler': self.scaler,
                'expected_feature_dim': self.expected_feature_dim
            }
            
            # Save PyTorch model if available
            if self.use_gpu and self.pytorch_model is not None:
                model_data['pytorch_model_state'] = self.pytorch_model.state_dict()
                model_data['pytorch_model_input_dim'] = self.pytorch_model.network[0].in_features
            else:
                # Save RandomForest model
                model_data['classifier'] = self.clf
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load(self, filepath=None):
        """Load model and configuration"""
        filepath = filepath or self.model_path
        
        try:
            if not os.path.exists(filepath):
                print(f"Error: File not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.is_trained = model_data.get('is_trained', False)
            self.feature_config = model_data.get('feature_config', {
                'use_hog': True,
                'use_hough': True,
                'use_shape': True,
                'apply_noise_filter': True
            })
            self.expected_feature_dim = model_data.get('expected_feature_dim', None)
            
            # Load scaler (for PyTorch) or extract from pipeline (for RandomForest)
            if 'scaler' in model_data:
                self.scaler = model_data['scaler']
                if hasattr(self.scaler, 'n_features_in_'):
                    scaler_dim = self.scaler.n_features_in_
                    print(f"Scaler expects {scaler_dim} features")
                    # Use scaler dimension if expected_feature_dim not set
                    if self.expected_feature_dim is None:
                        self.expected_feature_dim = scaler_dim
                        print(f"   Using scaler dimension as expected_feature_dim: {self.expected_feature_dim}")
                    elif self.expected_feature_dim != scaler_dim:
                        print(f"⚠ Warning: expected_feature_dim ({self.expected_feature_dim}) != scaler.n_features_in_ ({scaler_dim})")
                        # Use scaler's dimension as source of truth
                        self.expected_feature_dim = scaler_dim
                        print(f"   Using scaler dimension: {self.expected_feature_dim}")
            elif 'classifier' in model_data and hasattr(model_data['classifier'], 'named_steps'):
                # Extract scaler from sklearn pipeline
                if 'standardscaler' in model_data['classifier'].named_steps:
                    self.scaler = model_data['classifier'].named_steps['standardscaler']
                    if hasattr(self.scaler, 'n_features_in_'):
                        scaler_dim = self.scaler.n_features_in_
                        print(f"Scaler from pipeline expects {scaler_dim} features")
                        # Use scaler dimension if expected_feature_dim not set
                        if self.expected_feature_dim is None:
                            self.expected_feature_dim = scaler_dim
                            print(f"   Using scaler dimension as expected_feature_dim: {self.expected_feature_dim}")
                        elif self.expected_feature_dim != scaler_dim:
                            print(f"⚠ Warning: expected_feature_dim ({self.expected_feature_dim}) != scaler.n_features_in_ ({scaler_dim})")
                            self.expected_feature_dim = scaler_dim
                            print(f"   Using scaler dimension: {self.expected_feature_dim}")
                else:
                    self.scaler = StandardScaler()
            else:
                self.scaler = StandardScaler()
            
            if self.expected_feature_dim:
                print(f"✓ Expected feature dimension: {self.expected_feature_dim}")
            else:
                print("⚠ Warning: expected_feature_dim not set - dimension checking will be disabled")
            
            # Load PyTorch model if available
            if 'pytorch_model_state' in model_data and TORCH_AVAILABLE:
                self.use_gpu = model_data.get('use_gpu', False)
                if self.use_gpu:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    self.device = torch.device('cpu')
                
                input_dim = model_data['pytorch_model_input_dim']
                # Verify input_dim matches expected_feature_dim
                if self.expected_feature_dim and input_dim != self.expected_feature_dim:
                    print(f"⚠ Warning: PyTorch input_dim ({input_dim}) != expected_feature_dim ({self.expected_feature_dim})")
                    # Use PyTorch model's dimension as source of truth
                    self.expected_feature_dim = input_dim
                    print(f"   Using PyTorch model dimension: {self.expected_feature_dim}")
                
                self.pytorch_model = FeatureClassifierNet(input_dim=input_dim).to(self.device)
                self.pytorch_model.load_state_dict(model_data['pytorch_model_state'])
                self.pytorch_model.eval()
                print(f"✓ PyTorch model loaded (device: {self.device.type}, input_dim={input_dim})")
            else:
                # Load RandomForest model
                self.use_gpu = False
                self.clf = model_data.get('classifier')
                if self.clf is None:
                    # Fallback to new RandomForest if classifier not found
                    self.clf = make_pipeline(
                        StandardScaler(),
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=20,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                    print("⚠ Warning: Classifier not found in model file, created new RandomForest")
                else:
                    # Extract expected dimension from RandomForest pipeline if available
                    if hasattr(self.clf, 'named_steps') and 'standardscaler' in self.clf.named_steps:
                        scaler_in_pipeline = self.clf.named_steps['standardscaler']
                        if hasattr(scaler_in_pipeline, 'n_features_in_'):
                            pipeline_dim = scaler_in_pipeline.n_features_in_
                            if self.expected_feature_dim and pipeline_dim != self.expected_feature_dim:
                                print(f"⚠ Warning: Pipeline scaler dimension ({pipeline_dim}) != expected_feature_dim ({self.expected_feature_dim})")
                                self.expected_feature_dim = pipeline_dim
                            elif not self.expected_feature_dim:
                                self.expected_feature_dim = pipeline_dim
                print(f"✓ RandomForest model loaded (expected_dim={self.expected_feature_dim})")
            
            print(f"✓ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False


# Global instance
_bird_v_drone_classifier = None

def get_bird_v_drone_classifier(use_gpu=True):
    """Get global instance of Bird-v-Drone classifier"""
    global _bird_v_drone_classifier
    if _bird_v_drone_classifier is None:
        _bird_v_drone_classifier = BirdVsDroneClassifier(use_gpu=use_gpu)
    return _bird_v_drone_classifier
