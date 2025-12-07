
import os
import cv2
import numpy as np
import pickle
from image_processing import train_shape_code_model

def main():
    base_dir = os.getcwd()
    drone_folder = os.path.join(base_dir, "Drones")
    bird_folder = os.path.join(base_dir, "Birds")
    output_path = os.path.join(base_dir, "shape_code_model.pkl")
    
    print(f"Training Shape Code Model...")
    print(f"Drone folder: {drone_folder}")
    print(f"Bird folder: {bird_folder}")
    
    if not os.path.exists(drone_folder) or not os.path.exists(bird_folder):
        print("Error: Training folders not found!")
        return
        
    try:
        model = train_shape_code_model(
            drone_folder=drone_folder,
            bird_folder=bird_folder,
            output_model_path=output_path,
            DEBUG=True
        )
        
        if model.is_trained:
            print(f"\nSuccess! Model saved to: {output_path}")
            print(f"Drone templates: {len(model.drone_shape_codes)}")
            print(f"Bird templates: {len(model.bird_shape_codes)}")
        else:
            print("\nTraining failed: No valid templates found.")
            
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
