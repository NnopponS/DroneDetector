"""
Test Bird-v-Drone ML Classifier
ทดสอบความถูกต้องของ ML model
"""

import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from bird_v_drone_ml import get_bird_v_drone_classifier
    BIRD_V_DRONE_AVAILABLE = True
except ImportError as e:
    print(f"Error: Bird-v-Drone ML module not available: {e}")
    BIRD_V_DRONE_AVAILABLE = False
    exit(1)


def test_model(drone_dir, bird_dir, model_path='bird_v_drone_classifier.pkl', use_gpu=True):
    """
    Test the ML model with test images
    
    Args:
        drone_dir: Directory containing drone test images
        bird_dir: Directory containing bird test images
        model_path: Path to saved model
        use_gpu: Whether to use GPU
    """
    print("=" * 60)
    print("Bird-v-Drone ML Classifier Test")
    print("=" * 60)
    
    # Load classifier
    print("\n[1/5] Loading classifier...")
    try:
        classifier = get_bird_v_drone_classifier(use_gpu=use_gpu)
        if not classifier.load(model_path):
            print(f"Error: Could not load model from {model_path}")
            print("Please train the model first.")
            return False
        
        if not classifier.is_trained:
            print("Error: Model is not trained.")
            return False
        
        device_type = "GPU" if (hasattr(classifier, 'pytorch_model') and classifier.pytorch_model is not None) else "CPU"
        print(f"✓ Model loaded successfully (Device: {device_type})")
        print(f"  Feature config: {classifier.feature_config}")
        if hasattr(classifier, 'expected_feature_dim') and classifier.expected_feature_dim:
            print(f"  Expected feature dimension: {classifier.expected_feature_dim}")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Collect test images
    print("\n[2/5] Collecting test images...")
    drone_images = []
    bird_images = []
    
    if os.path.exists(drone_dir):
        for fname in os.listdir(drone_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                drone_images.append(os.path.join(drone_dir, fname))
    
    if os.path.exists(bird_dir):
        for fname in os.listdir(bird_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                bird_images.append(os.path.join(bird_dir, fname))
    
    print(f"  Found {len(drone_images)} drone images")
    print(f"  Found {len(bird_images)} bird images")
    
    if len(drone_images) == 0 and len(bird_images) == 0:
        print("Error: No test images found!")
        return False
    
    # Test predictions
    print("\n[3/5] Running predictions...")
    y_true = []
    y_pred = []
    y_proba = []
    failed_images = []
    
    # Test drone images (should be predicted as drone = True)
    print("  Testing drone images...")
    for i, img_path in enumerate(drone_images):
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                failed_images.append((img_path, "Could not load image"))
                continue
            
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            th1_inv = cv2.bitwise_not(th1)
            kernel = np.ones((3, 3), np.uint8)
            dilated_inv = cv2.dilate(th1_inv, kernel, iterations=3)
            contours, _ = cv2.findContours(dilated_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                failed_images.append((img_path, "No contours found"))
                continue
            
            # Use largest contour
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 50:
                failed_images.append((img_path, "Contour too small"))
                continue
            
            # Predict
            is_drone, prob = classifier.predict(c, image)
            
            y_true.append(1)  # 1 = Drone (true label)
            y_pred.append(1 if is_drone else 0)  # 1 = Drone, 0 = Bird
            y_proba.append(prob)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(drone_images)} drone images...")
        except Exception as e:
            failed_images.append((img_path, str(e)))
            print(f"    Error processing {os.path.basename(img_path)}: {e}")
    
    # Test bird images (should be predicted as bird = False)
    print("  Testing bird images...")
    for i, img_path in enumerate(bird_images):
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                failed_images.append((img_path, "Could not load image"))
                continue
            
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            th1_inv = cv2.bitwise_not(th1)
            kernel = np.ones((3, 3), np.uint8)
            dilated_inv = cv2.dilate(th1_inv, kernel, iterations=3)
            contours, _ = cv2.findContours(dilated_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                failed_images.append((img_path, "No contours found"))
                continue
            
            # Use largest contour
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 50:
                failed_images.append((img_path, "Contour too small"))
                continue
            
            # Predict
            is_drone, prob = classifier.predict(c, image)
            
            y_true.append(0)  # 0 = Bird (true label)
            y_pred.append(1 if is_drone else 0)  # 1 = Drone, 0 = Bird
            y_proba.append(prob)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(bird_images)} bird images...")
        except Exception as e:
            failed_images.append((img_path, str(e)))
            print(f"    Error processing {os.path.basename(img_path)}: {e}")
    
    print(f"\n  Total tested: {len(y_true)} images")
    if failed_images:
        print(f"  Failed: {len(failed_images)} images")
    
    if len(y_true) == 0:
        print("Error: No successful predictions!")
        return False
    
    # Calculate metrics
    print("\n[4/5] Calculating metrics...")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Display results
    print("\n[5/5] Test Results")
    print("=" * 60)
    print(f"Total Test Images: {len(y_true)}")
    print(f"  - Drones: {np.sum(y_true == 1)}")
    print(f"  - Birds: {np.sum(y_true == 0)}")
    print()
    print("Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print()
    print("Confusion Matrix:")
    print("                Predicted")
    print("              Drone   Bird")
    print(f"Actual Drone   {cm[1][1]:4d}   {cm[1][0]:4d}")
    print(f"       Bird    {cm[0][1]:4d}   {cm[0][0]:4d}")
    print()
    
    # Detailed breakdown
    drone_correct = np.sum((y_true == 1) & (y_pred == 1))
    drone_total = np.sum(y_true == 1)
    bird_correct = np.sum((y_true == 0) & (y_pred == 0))
    bird_total = np.sum(y_true == 0)
    
    print("Per-Class Performance:")
    if drone_total > 0:
        drone_acc = drone_correct / drone_total
        print(f"  Drone Detection: {drone_correct}/{drone_total} ({drone_acc*100:.2f}%)")
    if bird_total > 0:
        bird_acc = bird_correct / bird_total
        print(f"  Bird Detection:  {bird_correct}/{bird_total} ({bird_acc*100:.2f}%)")
    print()
    
    # Show probability distribution
    print("Confidence Statistics:")
    drone_probs = y_proba[y_true == 1]
    bird_probs = y_proba[y_true == 0]
    
    if len(drone_probs) > 0:
        print(f"  Drone predictions (should be high):")
        print(f"    Mean: {np.mean(drone_probs):.3f}, Min: {np.min(drone_probs):.3f}, Max: {np.max(drone_probs):.3f}")
    if len(bird_probs) > 0:
        print(f"  Bird predictions (should be low):")
        print(f"    Mean: {np.mean(bird_probs):.3f}, Min: {np.min(bird_probs):.3f}, Max: {np.max(bird_probs):.3f}")
    print()
    
    # Show some example predictions
    print("Sample Predictions (first 10):")
    print("-" * 60)
    for i in range(min(10, len(y_true))):
        true_label = "Drone" if y_true[i] == 1 else "Bird"
        pred_label = "Drone" if y_pred[i] == 1 else "Bird"
        correct = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"  {correct} True: {true_label:5s} | Predicted: {pred_label:5s} | Confidence: {y_proba[i]:.3f}")
    
    if failed_images:
        print(f"\nFailed Images ({len(failed_images)}):")
        for img_path, error in failed_images[:10]:  # Show first 10
            print(f"  - {os.path.basename(img_path)}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    
    # Default paths
    drone_dir = os.path.join(os.getcwd(), "Drones")
    bird_dir = os.path.join(os.getcwd(), "Birds")
    model_path = "bird_v_drone_classifier.pkl"
    use_gpu = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        drone_dir = sys.argv[1]
    if len(sys.argv) > 2:
        bird_dir = sys.argv[2]
    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    if len(sys.argv) > 4:
        use_gpu = sys.argv[4].lower() in ['true', '1', 'yes', 'gpu']
    
    print(f"Test Configuration:")
    print(f"  Drone directory: {drone_dir}")
    print(f"  Bird directory: {bird_dir}")
    print(f"  Model path: {model_path}")
    print(f"  Use GPU: {use_gpu}")
    print()
    
    success = test_model(drone_dir, bird_dir, model_path, use_gpu)
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed!")
        sys.exit(1)
