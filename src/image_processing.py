import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from . import paths
except ImportError:
    # Fallback for when run as script
    import paths

# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

def extract_features(contour):
    """
    Extract robust shape features from a contour for ML classification.
    Features:
    - Hu Moments (7) - Log transformed
    - Solidity
    - Circularity
    - Aspect Ratio
    - Extent
    - Inertia Ratio (from moments)
    """
    try:
        # 0. Basic Moments
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None
        
        # 1. Hu Moments (Invariant to scale, rotation, translation)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log transform to make them comparable (handling small values)
        hu_logs = []
        for i in range(7):
            val = hu_moments[i]
            if val != 0:
                hu_logs.append(-1 * np.sign(val) * np.log10(abs(val)))
            else:
                hu_logs.append(0)
        
        # 2. Geometric Properties
        area = moments['m00']
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return None
        
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity: Ratio of contour area to its convex hull area.
        # Drones with arms have lower solidity than blobby birds.
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Circularity: 4*pi*Area / Perimeter^2. Perfect circle = 1.
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Aspect Ratio of bounding rect
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Extent: Ratio of contour area to bounding rect area
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # 3. Inertia Ratio (Elongation) from moments
        # Calculate eigenvalues of the covariance matrix of second central moments
        # (This is more accurate "elongation" than bounding box aspect ratio)
        denominator = np.sqrt((2 * moments['mu11'])**2 + (moments['mu20'] - moments['mu02'])**2)
        eps = 1e-7
        ratio_1 = (moments['mu20'] + moments['mu02'] + denominator) / (moments['mu20'] + moments['mu02'] - denominator + eps)
        inertia_ratio = np.sqrt(ratio_1) if ratio_1 > 0 else 0
        
        # Combine all features
        features = np.array(hu_logs + [solidity, circularity, aspect_ratio, extent, inertia_ratio])
        
        # Replace NaNs or Infs
        features = np.nan_to_num(features)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ---------------------------------------------------------
# ML Classifier
# ---------------------------------------------------------

class DroneClassifier:
    def __init__(self, model_path=None):
        self.model_path = str(model_path or paths.DRONE_CLASSIFIER_MODEL)
        # Random Forest is robust and requires less tuning than SVM
        self.clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
        self.is_trained = False
        
    def train(self, drone_dir, bird_dir):
        """Train the classifier using images from drone and bird directories."""
        X = []
        y = []
        
        print(f"Training from:\n Drones: {drone_dir}\n Birds: {bird_dir}")
        
        # Process Drones (Label = 1)
        drone_count = 0
        if os.path.exists(drone_dir):
            for fname in os.listdir(drone_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    path = os.path.join(drone_dir, fname)
                    feats = self._process_image_file(path)
                    if feats is not None:
                        X.append(feats)
                        y.append(1) # 1 = Drone
                        drone_count += 1
                        
        # Process Birds (Label = 0)
        bird_count = 0
        if os.path.exists(bird_dir):
            for fname in os.listdir(bird_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    path = os.path.join(bird_dir, fname)
                    feats = self._process_image_file(path)
                    if feats is not None:
                        X.append(feats)
                        y.append(0) # 0 = Bird
                        bird_count += 1
        
        print(f"Collected {drone_count} drone samples and {bird_count} bird samples.")
        
        if len(X) == 0:
            print("No training data found!")
            return False
            
        # Fit model
        self.clf.fit(X, y)
        self.is_trained = True
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)
            
        print("Model trained and saved.")
        return True

    def _process_image_file(self, path):
        """Helper to load image and extract features from largest contour"""
        try:
            # Handle unicode paths
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
            
            if img is None: return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Simple thresholding to find object
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours: return None
            
            # Assume largest contour is the object
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 50: return None # Too small
            
            return extract_features(c)
        except Exception as e:
            # print(f"Error processing {path}: {e}")
            return None

    def predict(self, contour):
        """Predict if a contour is a drone (True) or Bird (False)"""
        if not self.is_trained:
            # Fallback if not trained: Assume everything nice is a drone? 
            # Or return False to be safe.
            return False, 0.0
            
        features = extract_features(contour)
        if features is None:
            return False, 0.0
            
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Get probability
        probs = self.clf.predict_proba(features)[0]
        drone_prob = probs[1] # Probability of class 1 (Drone)
        
        return drone_prob > 0.5, drone_prob

    def load(self):
        """Load pretrained model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.clf = pickle.load(f)
                self.is_trained = True
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        return False

# ---------------------------------------------------------
# Legacy / Fallback (Minimal wrapper to prevent breakages)
# ---------------------------------------------------------

def extract_shape_code(contour, DEBUG=False):
    # Backward compatibility stub
    return extract_features(contour)

def is_drone_by_shape_code(shape_code_info, DEBUG=False):
    # Backward compatibility stub - logic handled by classifier now
    return False

# Global instance
_classifier = DroneClassifier()

def get_classifier():
    return _classifier
