#!/usr/bin/env python3
"""
Script สำหรับเทรนและทดสอบโมเดล Drone vs Bird Classifier
รองรับทั้ง PyTorch CNN และ Enhanced ML Classifier
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import paths

# Import classifiers
try:
    from .drone_classifier_pytorch import DroneClassifierPyTorch
    PYTORCH_CNN_AVAILABLE = True
except ImportError:
    PYTORCH_CNN_AVAILABLE = False
    print("Warning: DroneClassifierPyTorch not available")

try:
    from .bird_v_drone_ml import BirdVsDroneClassifier
    ENHANCED_ML_AVAILABLE = True
except ImportError:
    ENHANCED_ML_AVAILABLE = False
    print("Warning: BirdVsDroneClassifier not available")

def print_confusion_matrix(y_true, y_pred, labels=['Bird', 'Drone']):
    """Print confusion matrix in a readable format"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(" " * 12 + "Predicted")
    print(" " * 12 + " ".join(f"{label:>8}" for label in labels))
    for i, label in enumerate(labels):
        print(f"{label:>10} " + " ".join(f"{cm[i][j]:>8}" for j in range(len(labels))))

def predict_from_image_path(model, img_path):
    """Helper function to predict from image path using contour detection"""
    try:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            return None, None
        
        # Find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 50:
            return None, None
        
        # Predict
        is_drone, prob = model.predict(contour, image)
        return is_drone, prob
    except Exception as e:
        return None, None

def train_and_test_pytorch_cnn(drone_dir, bird_dir, test_drone_dir=None, test_bird_dir=None, epochs=50):
    """Train and test PyTorch CNN classifier"""
    if not PYTORCH_CNN_AVAILABLE:
        print("❌ PyTorch CNN classifier not available")
        return None
    
    print("\n" + "="*60)
    print("Training PyTorch CNN Classifier")
    print("="*60)
    
    # Create model
    model = DroneClassifierPyTorch(model_path=paths.DRONE_CLASSIFIER_PYTORCH_MODEL)
    
    # Train
    print(f"\nTraining with:")
    print(f"  Drone directory: {drone_dir}")
    print(f"  Bird directory: {bird_dir}")
    print(f"  Epochs: {epochs}")
    
    success = model.train(
        drone_dir=drone_dir,
        bird_dir=bird_dir,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2
    )
    
    if not success:
        print("❌ Training failed")
        return None
    
    # Save model
    if model.save(paths.DRONE_CLASSIFIER_PYTORCH_MODEL):
        print("\n✓ Model saved successfully")
    else:
        print("⚠️  Failed to save model")
    
    # Test if test directories provided
    if test_drone_dir and test_bird_dir:
        print("\n" + "="*60)
        print("Testing PyTorch CNN Classifier")
        print("="*60)
        
        test_images = []
        test_labels = []
        
        # Load test drone images
        for fname in os.listdir(test_drone_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                test_images.append(os.path.join(test_drone_dir, fname))
                test_labels.append(1)  # Drone
        
        # Load test bird images
        for fname in os.listdir(test_bird_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                test_images.append(os.path.join(test_bird_dir, fname))
                test_labels.append(0)  # Bird
        
        if len(test_images) == 0:
            print("⚠️  No test images found")
            return model
        
        print(f"\nTesting on {len(test_images)} images ({sum(test_labels)} drones, {len(test_labels)-sum(test_labels)} birds)")
        
        # Predict
        predictions = []
        correct = 0
        failed = 0
        
        for img_path, true_label in zip(test_images, test_labels):
            is_drone, prob = predict_from_image_path(model, img_path)
            if is_drone is not None:
                pred_label = 1 if is_drone else 0
                predictions.append(pred_label)
                if pred_label == true_label:
                    correct += 1
            else:
                predictions.append(-1)  # Failed prediction
                failed += 1
        
        # Calculate metrics
        valid_predictions = [p for p in predictions if p != -1]
        valid_labels = [test_labels[i] for i, p in enumerate(predictions) if p != -1]
        
        if len(valid_predictions) > 0:
            accuracy = accuracy_score(valid_labels, valid_predictions)
            print(f"\nTest Results:")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Correct: {correct}/{len(valid_predictions)}")
            if failed > 0:
                print(f"  Failed: {failed}")
            
            print_confusion_matrix(valid_labels, valid_predictions)
            
            print("\nClassification Report:")
            print(classification_report(valid_labels, valid_predictions, target_names=['Bird', 'Drone']))
        else:
            print("❌ No valid predictions")
    
    return model

def train_and_test_enhanced_ml(drone_dir, bird_dir, test_drone_dir=None, test_bird_dir=None, use_gpu=False):
    """Train and test Enhanced ML Classifier (HOG + Hough + Shape)"""
    if not ENHANCED_ML_AVAILABLE:
        print("❌ Enhanced ML classifier not available")
        return None
    
    print("\n" + "="*60)
    print("Training Enhanced ML Classifier (HOG + Hough + Shape)")
    print("="*60)
    
    # Create model
    try:
        model = BirdVsDroneClassifier(model_path=paths.BIRD_V_DRONE_MODEL, use_gpu=use_gpu)
    except RuntimeError as e:
        print(f"⚠️  GPU not available, using CPU: {e}")
        model = BirdVsDroneClassifier(model_path=paths.BIRD_V_DRONE_MODEL, use_gpu=False)
    
    # Train
    print(f"\nTraining with:")
    print(f"  Drone directory: {drone_dir}")
    print(f"  Bird directory: {bird_dir}")
    print(f"  Features: HOG + Hough + Shape")
    
    success = model.train(
        drone_dir=drone_dir,
        bird_dir=bird_dir,
        use_hog=True,
        use_hough=True,
        use_shape=True,
        apply_noise_filter=True
    )
    
    if not success:
        print("❌ Training failed")
        return None
    
    # Save model
    if model.save(paths.BIRD_V_DRONE_MODEL):
        print("\n✓ Model saved successfully")
    else:
        print("⚠️  Failed to save model")
    
    # Test if test directories provided
    if test_drone_dir and test_bird_dir:
        print("\n" + "="*60)
        print("Testing Enhanced ML Classifier")
        print("="*60)
        
        test_images = []
        test_labels = []
        
        # Load test drone images
        for fname in os.listdir(test_drone_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                test_images.append(os.path.join(test_drone_dir, fname))
                test_labels.append(1)  # Drone
        
        # Load test bird images
        for fname in os.listdir(test_bird_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                test_images.append(os.path.join(test_bird_dir, fname))
                test_labels.append(0)  # Bird
        
        if len(test_images) == 0:
            print("⚠️  No test images found")
            return model
        
        print(f"\nTesting on {len(test_images)} images ({sum(test_labels)} drones, {len(test_labels)-sum(test_labels)} birds)")
        
        # Predict
        predictions = []
        correct = 0
        failed = 0
        
        for img_path, true_label in zip(test_images, test_labels):
            is_drone, prob = predict_from_image_path(model, img_path)
            if is_drone is not None:
                pred_label = 1 if is_drone else 0
                predictions.append(pred_label)
                if pred_label == true_label:
                    correct += 1
            else:
                predictions.append(-1)  # Failed prediction
                failed += 1
        
        # Calculate metrics
        valid_predictions = [p for p in predictions if p != -1]
        valid_labels = [test_labels[i] for i, p in enumerate(predictions) if p != -1]
        
        if len(valid_predictions) > 0:
            accuracy = accuracy_score(valid_labels, valid_predictions)
            print(f"\nTest Results:")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Correct: {correct}/{len(valid_predictions)}")
            if failed > 0:
                print(f"  Failed: {failed}")
            
            print_confusion_matrix(valid_labels, valid_predictions)
            
            print("\nClassification Report:")
            print(classification_report(valid_labels, valid_predictions, target_names=['Bird', 'Drone']))
        else:
            print("❌ No valid predictions")
    
    return model

def main():
    """Main function"""
    drone_dir = paths.DRONE_DIR
    bird_dir = paths.BIRD_DIR
    
    print("="*60)
    print("Drone vs Bird Classifier Training and Testing")
    print("="*60)
    
    # Check directories
    if not drone_dir.exists():
        print(f"❌ Drone directory not found: {drone_dir}")
        sys.exit(1)
    
    if not bird_dir.exists():
        print(f"❌ Bird directory not found: {bird_dir}")
        sys.exit(1)
    
    print(f"\nTraining directories:")
    print(f"  Drones: {drone_dir}")
    print(f"  Birds: {bird_dir}")
    
    # Count images
    drone_count = len([f for f in os.listdir(drone_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    bird_count = len([f for f in os.listdir(bird_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    print(f"\nTraining data:")
    print(f"  Drone images: {drone_count}")
    print(f"  Bird images: {bird_count}")
    print(f"  Total: {drone_count + bird_count}")
    
    if drone_count == 0 or bird_count == 0:
        print("❌ Need images in both directories")
        sys.exit(1)
    
    # Train both models by default (can be changed)
    print("\n" + "="*60)
    print("Available Models:")
    print("="*60)
    print("1. PyTorch CNN Classifier (Deep Learning)")
    print("2. Enhanced ML Classifier (HOG + Hough + Shape)")
    print("3. Both (default)")
    
    # Default to both models
    choice = '3'
    print(f"\nTraining both models (default)")
    print("Note: Using 20% of training data for validation during training")
    
    models_trained = []
    
    if choice in ['1', '3']:
        if PYTORCH_CNN_AVAILABLE:
            model = train_and_test_pytorch_cnn(
                drone_dir=str(drone_dir),
                bird_dir=str(bird_dir),
                epochs=50
            )
            if model:
                models_trained.append(('PyTorch CNN', model))
        else:
            print("⚠️  PyTorch CNN not available")
    
    if choice in ['2', '3']:
        if ENHANCED_ML_AVAILABLE:
            model = train_and_test_enhanced_ml(
                drone_dir=str(drone_dir),
                bird_dir=str(bird_dir),
                use_gpu=False  # Set to True if you have GPU
            )
            if model:
                models_trained.append(('Enhanced ML', model))
        else:
            print("⚠️  Enhanced ML Classifier not available")
    
    if not models_trained:
        print("\n❌ No models were trained successfully")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nTrained {len(models_trained)} model(s):")
    for name, _ in models_trained:
        print(f"  ✓ {name}")
    
    print("\nModel files saved:")
    if choice in ['1', '3'] and PYTORCH_CNN_AVAILABLE:
        print(f"  - {paths.DRONE_CLASSIFIER_PYTORCH_MODEL}")
    if choice in ['2', '3'] and ENHANCED_ML_AVAILABLE:
        print(f"  - {paths.BIRD_V_DRONE_MODEL}")

if __name__ == "__main__":
    main()
