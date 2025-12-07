"""
GPS Model using Gradient Boosting Regressor
ใช้ Gradient Boosting สำหรับความแม่นยำสูง
"""

import numpy as np
import json
import os
import pickle

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Gradient Boosting requires sklearn.")


class GPSModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0):
        """
        Args:
            n_estimators: จำนวน trees (default: 100, เพิ่มเพื่อความแม่นยำมากขึ้น)
            learning_rate: learning rate (default: 0.1, ลดเพื่อความเสถียร)
            max_depth: ความลึกของ tree (default: 3, เพิ่มเพื่อความซับซ้อน)
            subsample: สัดส่วนของ samples ที่ใช้ในแต่ละ tree (default: 1.0 = ใช้ทั้งหมด)
        """
        self.use_sklearn = SKLEARN_AVAILABLE
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Gradient Boosting requires scikit-learn. Please install it.")
        
        # ใช้ Gradient Boosting Regressor
        # Gradient Boosting ไม่ต้องการ feature scaling เพราะจัดการเอง
        self.model_lat = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42
        )
        self.model_lon = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42
        )
        self.model_alt = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42
        )
        
        self.is_trained = False
        self.training_stats = {}  # เก็บสถิติการเทรน
        
    def _create_features(self, x, y):
        """
        สร้าง feature vector จาก pixel coordinates
        Gradient Boosting ใช้แค่ x, y เป็น features (จัดการ non-linearity เอง)
        """
        return np.array([[x, y]])  # 2D array สำหรับ sklearn
    
    def train(self, data_points=None, X=None, y_lat=None, y_lon=None, y_alt=None):
        """
        Train the model using data points.
        
        Args:
            data_points: List of tuples (center_x, center_y, lat, lon, alt) - OR
            X: List of [x, y] coordinates
            y_lat: List of latitudes
            y_lon: List of longitudes
            y_alt: List of altitudes
            
        Returns:
            bool: Success status
        """
        # Support both formats
        if data_points is not None:
            # Prepare data from tuples
            X = []
            Y_lat = []
            Y_lon = []
            Y_alt = []
            
            for x, y, lat, lon, alt in data_points:
                X.append([x, y])
                Y_lat.append(lat)
                Y_lon.append(lon)
                Y_alt.append(alt)
        elif X is not None and y_lat is not None and y_lon is not None and y_alt is not None:
            # Use provided arrays
            Y_lat = y_lat
            Y_lon = y_lon
            Y_alt = y_alt
        else:
            print("Error: Need either data_points or (X, y_lat, y_lon, y_alt)")
            return False
        
        if not X or len(X) < 10:
            print(f"Error: Need at least 10 data points (got {len(X)})")
            return False
            
        X = np.array(X)
        Y_lat = np.array(Y_lat)
        Y_lon = np.array(Y_lon)
        Y_alt = np.array(Y_alt)
        
        try:
            # Gradient Boosting ไม่ต้องการ feature scaling
            # Train models
            print("Training Gradient Boosting models...")
            self.model_lat.fit(X, Y_lat)
            self.model_lon.fit(X, Y_lon)
            self.model_alt.fit(X, Y_alt)
            
            # Calculate training scores (R² score)
            score_lat = self.model_lat.score(X, Y_lat)
            score_lon = self.model_lon.score(X, Y_lon)
            score_alt = self.model_alt.score(X, Y_alt)
            
            # Cross-validation (ถ้ามีข้อมูลเพียงพอ)
            if len(X) >= 20:
                print("Running cross-validation...")
                cv_scores_lat = cross_val_score(self.model_lat, X, Y_lat, cv=5, scoring='r2')
                cv_scores_lon = cross_val_score(self.model_lon, X, Y_lon, cv=5, scoring='r2')
                cv_scores_alt = cross_val_score(self.model_alt, X, Y_alt, cv=5, scoring='r2')
                
                self.training_stats = {
                    'train_score_lat': score_lat,
                    'train_score_lon': score_lon,
                    'train_score_alt': score_alt,
                    'cv_score_lat': cv_scores_lat.mean(),
                    'cv_score_lon': cv_scores_lon.mean(),
                    'cv_score_alt': cv_scores_alt.mean(),
                    'n_samples': len(X),
                    'model_type': 'GradientBoostingRegressor'
                }
            else:
                self.training_stats = {
                    'train_score_lat': score_lat,
                    'train_score_lon': score_lon,
                    'train_score_alt': score_alt,
                    'n_samples': len(X),
                    'model_type': 'GradientBoostingRegressor'
                }
            
            self.is_trained = True
            print(f"[OK] Training complete with {len(X)} samples")
            if self.use_sklearn and 'cv_score_lat' in self.training_stats:
                print(f"  - Lat R²: {self.training_stats['cv_score_lat']:.4f}")
                print(f"  - Lon R²: {self.training_stats['cv_score_lon']:.4f}")
                print(f"  - Alt R²: {self.training_stats['cv_score_alt']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def predict(self, center_x, center_y):
        """
        Predict GPS coordinates from pixel position.
        
        Args:
            center_x: X coordinate in pixels
            center_y: Y coordinate in pixels
            
        Returns:
            tuple: (lat, lon, alt) or None if not trained
        """
        if not self.is_trained:
            return None
            
        try:
            features = np.array([[center_x, center_y]])
            
            # Gradient Boosting prediction
            lat = self.model_lat.predict(features)[0]
            lon = self.model_lon.predict(features)[0]
            alt = self.model_alt.predict(features)[0]
            
            return lat, lon, alt
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_training_stats(self):
        """Get training statistics"""
        return self.training_stats.copy() if self.training_stats else {}
        
    def save(self, filepath):
        """Save model using pickle (Gradient Boosting ไม่สามารถ save เป็น JSON ได้)"""
        if not self.is_trained:
            print("Error: Model is not trained yet")
            return False
            
        try:
            # Convert Path object to string if needed
            filepath_str = str(filepath) if filepath else None
            if not filepath_str:
                print("Error: No filepath provided")
                return False
            
            # ใช้ pickle สำหรับ Gradient Boosting
            model_data = {
                'model_lat': self.model_lat,
                'model_lon': self.model_lon,
                'model_alt': self.model_alt,
                'training_stats': self.training_stats,
                'model_type': 'GradientBoostingRegressor'
            }
            
            # Save as pickle file
            if not filepath_str.endswith('.pkl'):
                filepath_str = filepath_str.replace('.json', '.pkl')
            
            with open(filepath_str, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"[OK] Model saved to {filepath_str}")
            return True
            
        except Exception as e:
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load(self, filepath):
        """Load model from pickle file"""
        try:
            # Convert Path object to string if needed
            filepath_str = str(filepath) if filepath else None
            if not filepath_str:
                print("Error: No filepath provided")
                return False
            
            # ตรวจสอบว่าไฟล์มีอยู่จริง
            if not os.path.exists(filepath_str):
                print(f"Error: File not found: {filepath_str}")
                return False
            
            # ถ้าเป็น JSON เก่า ให้บอกว่าไม่รองรับ
            if filepath_str.endswith('.json'):
                print("Error: Old JSON format not supported. Please retrain the model with Gradient Boosting.")
                return False
            
            # ถ้าไม่มี extension หรือไม่ใช่ .pkl ให้เพิ่ม .pkl
            if not filepath_str.endswith('.pkl'):
                # ลองหา .pkl file
                pkl_path = filepath_str.replace('.json', '.pkl')
                if os.path.exists(pkl_path):
                    filepath_str = pkl_path
                elif os.path.exists(filepath_str + '.pkl'):
                    filepath_str = filepath_str + '.pkl'
                else:
                    # ถ้าไม่มี extension ให้เพิ่ม .pkl
                    if '.' not in os.path.basename(filepath_str):
                        filepath_str = filepath_str + '.pkl'
                    else:
                        print(f"Error: File must be .pkl format. Got: {filepath_str}")
                        return False
            
            # ตรวจสอบอีกครั้งว่าไฟล์มีอยู่
            if not os.path.exists(filepath_str):
                print(f"Error: File not found: {filepath_str}")
                return False
            
            # Load model จาก pickle file
            with open(filepath_str, 'rb') as f:
                model_data = pickle.load(f)
            
            # ตรวจสอบว่าเป็น dictionary และมีข้อมูลที่จำเป็น
            if not isinstance(model_data, dict):
                print("Error: Invalid model file format")
                return False
            
            if 'model_lat' not in model_data or 'model_lon' not in model_data or 'model_alt' not in model_data:
                print("Error: Model file missing required components")
                return False
            
            # โหลดโมเดล
            self.model_lat = model_data['model_lat']
            self.model_lon = model_data['model_lon']
            self.model_alt = model_data['model_alt']
            self.training_stats = model_data.get('training_stats', {})
            
            self.is_trained = True
            print(f"[OK] Model loaded from {filepath_str}")
            
            if self.training_stats:
                print(f"  - Trained with {self.training_stats.get('n_samples', 'unknown')} samples")
                print(f"  - Model type: {self.training_stats.get('model_type', 'GradientBoostingRegressor')}")
            
            return True
            
        except pickle.UnpicklingError as e:
            print(f"Error: Cannot unpickle file. File may be corrupted or in wrong format: {e}")
            return False
        except Exception as e:
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()
            return False
