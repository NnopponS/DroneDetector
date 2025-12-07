"""
Shape Code Generator with OCR support
สร้าง shape codes จากรูป Drone และ Bird โดยใช้ contour features และ OCR
"""

import cv2
import numpy as np
import os
import json
import pickle

try:
    from . import paths
    from .image_processing import extract_features
except ImportError:
    # Fallback for when run as script
    import paths
    from image_processing import extract_features

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR features will be disabled.")


class ShapeCodeGenerator:
    def __init__(self, invert_classification=False):
        """
        Args:
            invert_classification: If True, swap drone/bird classification results (default False - use correct folder mapping)
        """
        self.drone_shape_codes = []
        self.bird_shape_codes = []
        self.ocr_enabled = OCR_AVAILABLE
        self.invert_classification = invert_classification  # Default False - assume folders are correctly labeled
    
    def extract_shape_features(self, contour):
        """Extract shape features from contour"""
        return extract_features(contour)
    
    def extract_ocr_text(self, image, contour):
        """Extract text from ROI using OCR"""
        if not self.ocr_enabled:
            return ""
        
        try:
            x, y, w, h = cv2.boundingRect(contour)
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                return ""
            
            # Convert to grayscale
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi
            
            # Enhance image for OCR
            roi_gray = cv2.resize(roi_gray, (max(100, w*2), max(100, h*2)))
            _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            text = pytesseract.image_to_string(roi_binary, config='--psm 8')
            return text.strip()
        except Exception as e:
            return ""
    
    def process_image(self, img_path):
        """Process single image and extract shape codes"""
        try:
            # Load image
            stream = open(img_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
            
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < 50:
                return None
            
            # Extract features
            shape_features = self.extract_shape_features(largest_contour)
            if shape_features is None:
                return None
            
            # Extract OCR text
            ocr_text = self.extract_ocr_text(image, largest_contour)
            
            return {
                'features': shape_features.tolist(),
                'ocr_text': ocr_text,
                'area': cv2.contourArea(largest_contour),
                'perimeter': cv2.arcLength(largest_contour, True)
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def train(self, drone_dir, bird_dir, output_path=None):
        """
        Train shape code model from directories
        
        Args:
            drone_dir: Directory containing DRONE images (must be actual drones)
            bird_dir: Directory containing BIRD images (must be actual birds)
            output_path: Path to save the trained model
        """
        output_path = str(output_path or paths.SHAPE_CODE_MODEL_FILE)
        print("Generating shape codes from images...")
        print(f"Drone directory: {drone_dir}")
        print(f"Bird directory: {bird_dir}")
        
        # Process drone images from drone_dir
        drone_count = 0
        for fname in os.listdir(drone_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                img_path = os.path.join(drone_dir, fname)
                shape_code = self.process_image(img_path)
                if shape_code:
                    self.drone_shape_codes.append(shape_code)
                    drone_count += 1
        
        # Process bird images from bird_dir
        bird_count = 0
        for fname in os.listdir(bird_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                img_path = os.path.join(bird_dir, fname)
                shape_code = self.process_image(img_path)
                if shape_code:
                    self.bird_shape_codes.append(shape_code)
                    bird_count += 1
        
        print(f"Generated {drone_count} drone shape codes from '{drone_dir}'")
        print(f"Generated {bird_count} bird shape codes from '{bird_dir}'")
        
        # Save model
        model_data = {
            'drone_shape_codes': self.drone_shape_codes,
            'bird_shape_codes': self.bird_shape_codes,
            'ocr_enabled': self.ocr_enabled
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[OK] Shape code model saved to {output_path}")
        return True
    
    def load(self, model_path=None):
        """Load shape code model"""
        try:
            model_path = str(model_path or paths.SHAPE_CODE_MODEL_FILE)
            
            if not os.path.exists(model_path):
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            drone_codes = model_data.get('drone_shape_codes', [])
            bird_codes = model_data.get('bird_shape_codes', [])
            
            # Load codes with correct mapping:
            # drone_shape_codes = codes from Drones folder (should be actual drones)
            # bird_shape_codes = codes from Birds folder (should be actual birds)
            if self.invert_classification:
                # Only swap if explicitly requested (for backward compatibility with old models)
                self.drone_shape_codes = bird_codes  # Swapped
                self.bird_shape_codes = drone_codes  # Swapped
                print(f"⚠ Loaded and SWAPPED: {len(self.drone_shape_codes)} drone codes (from bird folder) and {len(self.bird_shape_codes)} bird codes (from drone folder)")
                print("⚠ WARNING: Using inverted classification. Please re-train model with correct folder mapping.")
            else:
                # Normal mapping: Drones folder -> drone codes, Birds folder -> bird codes
                self.drone_shape_codes = drone_codes
                self.bird_shape_codes = bird_codes
                print(f"[OK] Loaded {len(self.drone_shape_codes)} drone codes (from Drones folder) and {len(self.bird_shape_codes)} bird codes (from Birds folder)")
            
            self.ocr_enabled = model_data.get('ocr_enabled', False)
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def classify(self, contour, image=None):
        """Classify contour as drone or bird using shape codes"""
        try:
            if len(self.drone_shape_codes) == 0 and len(self.bird_shape_codes) == 0:
                return False, 0.0
            
            # Extract features
            features = self.extract_shape_features(contour)
            if features is None:
                return False, 0.0
            
            features = np.array(features)
            if features.size == 0:
                return False, 0.0
            
            # Calculate distances to drone codes
            drone_distances = []
            for code in self.drone_shape_codes:
                try:
                    if 'features' not in code:
                        continue
                    code_features = np.array(code['features'])
                    if code_features.size != features.size:
                        continue
                    distance = np.linalg.norm(features - code_features)
                    drone_distances.append(distance)
                except Exception as e:
                    print(f"Error calculating drone distance: {e}")
                    continue
            
            # Calculate distances to bird codes
            bird_distances = []
            for code in self.bird_shape_codes:
                try:
                    if 'features' not in code:
                        continue
                    code_features = np.array(code['features'])
                    if code_features.size != features.size:
                        continue
                    distance = np.linalg.norm(features - code_features)
                    bird_distances.append(distance)
                except Exception as e:
                    print(f"Error calculating bird distance: {e}")
                    continue
            
            if len(drone_distances) == 0 and len(bird_distances) == 0:
                return False, 0.0
            
            # Find minimum distances
            min_drone_dist = min(drone_distances) if drone_distances else float('inf')
            min_bird_dist = min(bird_distances) if bird_distances else float('inf')
            
            # Classify based on minimum distance
            # Return format: (is_drone: bool, drone_probability: float)
            # True = Drone, False = Bird
            # Note: If invert_classification is True, the codes are already swapped in load()
            
            if min_drone_dist < min_bird_dist:
                # Closer to drone codes = DRONE
                total_dist = min_drone_dist + min_bird_dist
                if total_dist > 0:
                    prob = 1.0 - (min_drone_dist / total_dist)
                else:
                    prob = 0.5
                is_drone = True
            else:
                # Closer to bird codes = BIRD
                total_dist = min_drone_dist + min_bird_dist
                if total_dist > 0:
                    prob = min_drone_dist / total_dist  # Lower prob = more likely bird
                else:
                    prob = 0.5
                is_drone = False
            
            return is_drone, max(0.0, min(1.0, prob))
        except Exception as e:
            print(f"Shape Code Classifier error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
