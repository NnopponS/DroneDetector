"""
Drone Detection GUI Application
ระบบตรวจจับโดรนพร้อม UI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import platform
import csv
import threading
import time
from datetime import datetime, timedelta
import math
from drone_tracker import DroneTracker
from gps_model import GPSModel
from video_processing import VideoProcessor
from image_processing import get_classifier

try:
    # Shape-code based classifier (Hu moments + contour features)
    from shape_code_generator import ShapeCodeGenerator
    SHAPE_CODE_AVAILABLE = True
except ImportError:
    SHAPE_CODE_AVAILABLE = False
    print("Warning: ShapeCodeGenerator module not available. Shape-code filtering disabled.")
try:
    from drone_only_ml import get_drone_only_classifier
    DRONE_ONLY_AVAILABLE = True
except ImportError as e:
    print(f"Drone-Only ML module not available: {e}")
    DRONE_ONLY_AVAILABLE = False

# Import paths module for centralized path management
try:
    import paths
    PATHS_AVAILABLE = True
except ImportError:
    # Fallback if paths module not available
    PATHS_AVAILABLE = False
    print("Warning: paths module not available, using fallback paths")

# Import config manager for storing preferences
try:
    from config_manager import (
        load_config, save_config, update_config,
        get_drone_folders, get_bird_folders, get_gps_train_folder,
        set_drone_folders, set_bird_folders, set_gps_train_folder
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config_manager not available, preferences will not be saved")


class DroneDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Detection Application - Enhanced (ML)")
        self.root.geometry("1400x900")
        
        # ตัวแปรพื้นฐาน
        self.current_image_path = None
        self.current_video_path = None
        self.video_cap = None
        self.is_playing_video = False
        self.video_thread = None
        self.video_fps = 24.0
        self.video_start_time = None
        
        # Initialize dictionary for steps EARLY to avoid errors
        self.processed_steps = {}
        
        # Training Folders
        self.gps_train_folder = None
        
        # ================================
        # Detection Parameters (Optimized for stability)
        # ================================
        self.threshold_value = 40
        self.min_contour_area = 50
        self.max_contour_area = 15000
        self.min_drone_area = 50  # Reduced for better detection
        self.max_drone_area = 15000
        self.min_aspect_ratio = 0.15
        self.max_aspect_ratio = 6.0
        self.bottom_crop_ratio = 0.35
        self.debug_mode = False
        
        # Detection Mode (Logic vs Model) - Allow selection
        self.detection_mode = tk.StringVar(value="logic")  # Default to logic-based
        self.video_detection_mode = tk.StringVar(value="logic")  # Default to logic-based for video
        self.selected_classifier_model = tk.StringVar(value="Drone-Only ML (image_process_drone + HOG + Hough)")  # For ML mode
        self.video_selected_classifier_model = tk.StringVar(value="Drone-Only ML (image_process_drone + HOG + Hough)")  # For ML mode
        
        # Model paths for tracking latest
        self.model_paths = {}  # Store model paths by type
        self.latest_model_path = None
        
        # Drone Tracker
        self.tracker = DroneTracker(max_drones=2, max_lost_frames=30, distance_threshold=500)
        
        # Video Processor
        self.video_processor = None
        
        # GPS Model - Auto-load if exists
        self.gps_model = GPSModel()
        self.gps_data = None
        self.using_custom_model = False
        
        # Try to auto-load GPS model from models folder
        if PATHS_AVAILABLE:
            paths.ensure_structure()
            gps_model_path = paths.GPS_MODEL_FILE
            if gps_model_path.exists():
                if self.gps_model.load(str(gps_model_path)):
                    self.using_custom_model = True
                    self.gps_model.model_path = str(gps_model_path)
                    print(f"[OK] Auto-loaded GPS Model from {gps_model_path}")
                else:
                    print(f"[WARNING] GPS Model file exists but failed to load: {gps_model_path}")
            else:
                print(f"[INFO] No existing GPS model found at {gps_model_path}")
        
        # GPS Base
        self.base_lat = 14.30492
        self.base_lon = 101.17255
        self.base_alt = 46.91
        
        # Drone-Only ML Classifier (uses image_process_drone pipeline)
        # Will be initialized after UI is created
        self.drone_only_classifier = None
        self.shape_code_classifier = None
        
        # Old models removed - using only Drone-Only ML (image_process_drone + HOG + Hough)
        self.classifier = None  # Old Random Forest - removed
        self.pytorch_classifier = None  # Old PyTorch - removed
        self.pytorch_gps_model = None
        self.pytorch_available = False

        # สร้าง UI - creating widgets sets up status_var and canvas
        self.create_widgets()
        
        # Initialize Drone-Only classifier - Auto-load if exists
        if DRONE_ONLY_AVAILABLE:
            try:
                self.drone_only_classifier = get_drone_only_classifier(use_gpu=False)
                # Try to load from default model path
                if PATHS_AVAILABLE:
                    model_path = paths.BIRD_V_DRONE_MODEL
                    if model_path.exists():
                        if self.drone_only_classifier.load():
                            print(f"[OK] Auto-loaded Drone-Only ML Classifier from {model_path}")
                        else:
                            print(f"[WARNING] Model file exists but failed to load: {model_path}")
                    else:
                        print(f"[INFO] No existing model found at {model_path}, will need to train")
                else:
                    # Try to load anyway
                    if self.drone_only_classifier.load():
                        print("Loaded Drone-Only ML Classifier")
            except Exception as e:
                print(f"Error loading Drone-Only classifier: {e}")
                self.drone_only_classifier = None

        # Initialize shape-code classifier (used as additional filter in ML mode) - Auto-load if exists
        if SHAPE_CODE_AVAILABLE:
            try:
                # ShapeCodeGenerator internally uses paths.SHAPE_CODE_MODEL_FILE
                self.shape_code_classifier = ShapeCodeGenerator()
                if PATHS_AVAILABLE:
                    shape_model_path = paths.SHAPE_CODE_MODEL_FILE
                    if shape_model_path.exists():
                        if self.shape_code_classifier.load():
                            print(f"[OK] Auto-loaded Shape Code model from {shape_model_path}")
                        else:
                            print(f"[WARNING] Shape code model file exists but failed to load: {shape_model_path}")
                            self.shape_code_classifier = None
                    else:
                        print(f"[INFO] No existing shape code model found at {shape_model_path}")
                        self.shape_code_classifier = None
                else:
                    if self.shape_code_classifier.load():
                        print("Loaded Shape Code model for ML filtering")
                    else:
                        self.shape_code_classifier = None
                        print("Shape Code model not found or failed to load.")
            except Exception as e:
                print(f"Error loading Shape Code model: {e}")
                self.shape_code_classifier = None
        
        # Initialize model status
        self.root.after(500, lambda: [self.update_model_list(), self.update_video_model_list()])
        
        # Load saved folder preferences after UI is created
        self.root.after(1000, self.load_saved_preferences)
    
    def create_widgets(self):
        """สร้าง UI"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left Panel
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        
        # Mode Selection
        mode_frame = ttk.LabelFrame(left_panel, text="System Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="image")
        ttk.Radiobutton(mode_frame, text="Image Processing (Single)", variable=self.mode_var, 
                       value="image", command=self.switch_mode).pack(anchor=tk.W)
        
        ttk.Radiobutton(mode_frame, text="Video Processing (Real-time)", variable=self.mode_var, 
                       value="video", command=self.switch_mode).pack(anchor=tk.W)

        ttk.Radiobutton(mode_frame, text="System Training", variable=self.mode_var, 
                       value="training", command=self.switch_mode).pack(anchor=tk.W)

        # --- Shared Settings --- (Removed sensitivity settings)

        # --- Image Mode Panel ---
        self.image_panel = ttk.Frame(left_panel)
        
        # Image File Selection
        img_file_frame = ttk.LabelFrame(self.image_panel, text="Image Source", padding="10")
        img_file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(img_file_frame, text="Select Image File", command=self.select_image).pack(fill=tk.X, pady=2)
        self.img_label = ttk.Label(img_file_frame, text="No file selected", wraplength=250)
        self.img_label.pack(fill=tk.X, pady=5)
        
        # Detection Mode Selection
        detection_mode_frame = ttk.LabelFrame(self.image_panel, text="Detection Mode", padding="10")
        detection_mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(detection_mode_frame, text="Detection Method:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Mode selection radio buttons
        mode_selection_frame = ttk.Frame(detection_mode_frame)
        mode_selection_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Radiobutton(mode_selection_frame, text="Logic-Based (Rule-Based)", 
                       variable=self.detection_mode, value="logic",
                       command=self._on_detection_mode_change).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Radiobutton(mode_selection_frame, text="ML-Based (Drone-Only ML)", 
                       variable=self.detection_mode, value="model",
                       command=self._on_detection_mode_change).pack(side=tk.LEFT)
        
        # Description based on selected mode
        self.detection_mode_desc = ttk.Label(detection_mode_frame, text="", 
                                            foreground="gray", font=("TkDefaultFont", 8))
        self.detection_mode_desc.pack(anchor=tk.W, pady=(0, 5))
        
        # Model status (only shown for ML mode)
        self.model_status_label = ttk.Label(detection_mode_frame, text="", foreground="blue")
        self.model_status_label.pack(anchor=tk.W, pady=2)
        
        # Update initial display
        self._on_detection_mode_change()
        
        # Image Steps - with scrollable canvas
        img_steps_frame = ttk.LabelFrame(self.image_panel, text="Processing Pipeline", padding="10")
        img_steps_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create canvas with scrollbar for steps
        steps_canvas = tk.Canvas(img_steps_frame, highlightthickness=0)
        steps_scrollbar = ttk.Scrollbar(img_steps_frame, orient="vertical", command=steps_canvas.yview)
        steps_scrollable_frame = ttk.Frame(steps_canvas)
        
        def configure_scroll_region(event=None):
            steps_canvas.configure(scrollregion=steps_canvas.bbox("all"))
        
        steps_scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        steps_canvas.create_window((0, 0), window=steps_scrollable_frame, anchor="nw")
        steps_canvas.configure(yscrollcommand=steps_scrollbar.set)
        
        # Enable mouse wheel scrolling (cross-platform)
        def on_mousewheel(event):
            if platform.system() == 'Windows':
                steps_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif platform.system() == 'Darwin':  # macOS
                steps_canvas.yview_scroll(int(-1*event.delta), "units")
            else:  # Linux
                if event.num == 4:
                    steps_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    steps_canvas.yview_scroll(1, "units")
        
        # Bind mouse wheel events based on platform
        if platform.system() == 'Windows':
            steps_canvas.bind_all("<MouseWheel>", lambda e: on_mousewheel(e) if steps_canvas.winfo_containing(e.x_root, e.y_root) else None)
        elif platform.system() == 'Darwin':  # macOS
            steps_canvas.bind_all("<MouseWheel>", lambda e: on_mousewheel(e) if steps_canvas.winfo_containing(e.x_root, e.y_root) else None)
        else:  # Linux
            steps_canvas.bind_all("<Button-4>", lambda e: on_mousewheel(e) if steps_canvas.winfo_containing(e.x_root, e.y_root) else None)
            steps_canvas.bind_all("<Button-5>", lambda e: on_mousewheel(e) if steps_canvas.winfo_containing(e.x_root, e.y_root) else None)
        
        steps = [
            ("Original", "original"),
            ("(1) Grayscale", "grayscale"),
            ("(2) Threshold", "threshold"),
            ("(3) Inversion", "inversion"),
            ("(4) Dilation", "dilation"),
            ("(5) Contours", "contours"),
            ("(6) Shape Classification", "shape_classification"),
            ("Final Detection", "final")
        ]
        self.step_buttons = {}
        for step_name, step_key in steps:
            btn = ttk.Button(steps_scrollable_frame, text=step_name, 
                           command=lambda k=step_key: self.show_step(k))
            btn.pack(fill=tk.X, pady=2)
            self.step_buttons[step_key] = btn
        
        steps_canvas.pack(side="left", fill="both", expand=True)
        steps_scrollbar.pack(side="right", fill="y")
        
        # Update scroll region after buttons are created
        steps_scrollable_frame.update_idletasks()
        configure_scroll_region()

        # Image Model Load
        img_model_frame = ttk.LabelFrame(self.image_panel, text="Active Models", padding="10")
        img_model_frame.pack(fill=tk.X, pady=(0, 10))
        self.img_model_status = tk.StringVar(value="GPS: Default Formula")
        ttk.Label(img_model_frame, textvariable=self.img_model_status, wraplength=250).pack(fill=tk.X, pady=2)
        ttk.Button(img_model_frame, text="Load Custom GPS Model", command=self.load_model).pack(fill=tk.X, pady=2)
        
        self.shape_model_status_img = tk.StringVar(value="Drone-Only ML Classifier")
        if self.drone_only_classifier and self.drone_only_classifier.is_trained:
             self.shape_model_status_img.set("Drone-Only ML: Ready [OK]")
        else:
             self.shape_model_status_img.set("Drone-Only ML: NOT Trained")
        ttk.Label(img_model_frame, textvariable=self.shape_model_status_img, wraplength=250).pack(fill=tk.X, pady=2)

        # --- Video Mode Panel ---
        self.video_panel = ttk.Frame(left_panel)
        
        # Video File Selection
        video_file_frame = ttk.LabelFrame(self.video_panel, text="Video Source", padding="10")
        video_file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(video_file_frame, text="Select Video File", command=self.select_video).pack(fill=tk.X, pady=2)
        self.video_label = ttk.Label(video_file_frame, text="No file selected", wraplength=250)
        self.video_label.pack(fill=tk.X, pady=5)
        
        # Video Controls
        video_controls_frame = ttk.LabelFrame(self.video_panel, text="Playback Controls", padding="10")
        video_controls_frame.pack(fill=tk.X, pady=(0, 10))
        self.play_btn = ttk.Button(video_controls_frame, text="▶ Play Video", command=self.play_video, state=tk.DISABLED)
        self.play_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(video_controls_frame, text="⏹ Stop Video", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Video Detection Mode Selection
        video_detection_frame = ttk.LabelFrame(self.video_panel, text="Detection Mode", padding="10")
        video_detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(video_detection_frame, text="Detection Method:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Mode selection radio buttons for video
        video_mode_selection_frame = ttk.Frame(video_detection_frame)
        video_mode_selection_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Radiobutton(video_mode_selection_frame, text="Logic-Based (Rule-Based)", 
                       variable=self.video_detection_mode, value="logic",
                       command=self._on_video_detection_mode_change).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Radiobutton(video_mode_selection_frame, text="ML-Based (Drone-Only ML)", 
                       variable=self.video_detection_mode, value="model",
                       command=self._on_video_detection_mode_change).pack(side=tk.LEFT)
        
        # Description based on selected mode
        self.video_detection_mode_desc = ttk.Label(video_detection_frame, text="", 
                                                   foreground="gray", font=("TkDefaultFont", 8))
        self.video_detection_mode_desc.pack(anchor=tk.W, pady=(0, 5))
        
        # Video model status (only shown for ML mode)
        self.video_model_status_label = ttk.Label(video_detection_frame, text="", foreground="blue")
        self.video_model_status_label.pack(anchor=tk.W, pady=2)
        
        # Update initial display
        self._on_video_detection_mode_change()
        
        # Video Info
        video_info_frame = ttk.LabelFrame(self.video_panel, text="Video Statistics", padding="10")
        video_info_frame.pack(fill=tk.X, pady=(0, 10))
        self.video_info_label = ttk.Label(video_info_frame, text="FPS: -\nDuration: -", wraplength=250)
        self.video_info_label.pack(fill=tk.X, pady=5)
        
        # --- Training Mode Panel --- (Scrollable)
        # Create scrollable frame for training panel
        self.training_container = ttk.Frame(left_panel)
        training_canvas = tk.Canvas(self.training_container, highlightthickness=0)
        training_scrollbar = ttk.Scrollbar(self.training_container, orient="vertical", command=training_canvas.yview)
        self.training_panel = ttk.Frame(training_canvas)
        
        # Configure scrollable region
        def configure_training_scroll(event=None):
            training_canvas.configure(scrollregion=training_canvas.bbox("all"))
        
        self.training_panel.bind("<Configure>", configure_training_scroll)
        training_canvas.create_window((0, 0), window=self.training_panel, anchor="nw")
        training_canvas.configure(yscrollcommand=training_scrollbar.set)
        
        # Mouse wheel scrolling (cross-platform)
        def on_training_mousewheel(event):
            if platform.system() == 'Windows':
                training_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif platform.system() == 'Darwin':  # macOS
                training_canvas.yview_scroll(int(-1*event.delta), "units")
            else:  # Linux
                if event.num == 4:
                    training_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    training_canvas.yview_scroll(1, "units")
        
        # Bind mouse wheel events based on platform
        if platform.system() == 'Windows':
            training_canvas.bind_all("<MouseWheel>", lambda e: on_training_mousewheel(e) if training_canvas.winfo_containing(e.x_root, e.y_root) else None)
        elif platform.system() == 'Darwin':  # macOS
            training_canvas.bind_all("<MouseWheel>", lambda e: on_training_mousewheel(e) if training_canvas.winfo_containing(e.x_root, e.y_root) else None)
        else:  # Linux
            training_canvas.bind_all("<Button-4>", lambda e: on_training_mousewheel(e) if training_canvas.winfo_containing(e.x_root, e.y_root) else None)
            training_canvas.bind_all("<Button-5>", lambda e: on_training_mousewheel(e) if training_canvas.winfo_containing(e.x_root, e.y_root) else None)
        
        training_scrollbar.pack(side="right", fill="y")
        training_canvas.pack(side="left", fill="both", expand=True)
        
        # 1. GPS Model Training
        gps_train_frame = ttk.LabelFrame(self.training_panel, text="1. GPS Localization Training", padding="10")
        gps_train_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(gps_train_frame, text="Train a model to predict GPS from pixel coordinates.").pack(anchor=tk.W, pady=(0,5))
        
        # Set default GPS folder label based on paths
        if PATHS_AVAILABLE:
            default_gps_text = f"Folder: Default ({paths.GPS_TRAIN_DIR.name})"
        else:
            default_gps_text = "Folder: Default (P2_DATA_TRAIN)"
        self.gps_folder_label = ttk.Label(gps_train_frame, text=default_gps_text, wraplength=250)
        self.gps_folder_label.pack(fill=tk.X, pady=2)
        
        ttk.Button(gps_train_frame, text="Select Training Folder", command=self.select_gps_folder).pack(fill=tk.X, pady=2)
        
        # Model type selection
        self.gps_model_type = tk.StringVar(value="sklearn")
        ttk.Radiobutton(gps_train_frame, text="Gradient Boosting (sklearn)", variable=self.gps_model_type, value="sklearn").pack(anchor=tk.W)
        if self.pytorch_available and self.pytorch_gps_model is not None:
            ttk.Radiobutton(gps_train_frame, text="Neural Network (PyTorch)", variable=self.gps_model_type, value="pytorch").pack(anchor=tk.W)
        
        self.train_btn = ttk.Button(gps_train_frame, text="Start GPS Training", command=self.start_training)
        self.train_btn.pack(fill=tk.X, pady=5)
        self.save_model_btn = ttk.Button(gps_train_frame, text="Save GPS Model", command=self.save_model, state=tk.DISABLED)
        self.save_model_btn.pack(fill=tk.X, pady=2)
        
        # Drone vs Bird ML Training (uses image_process_drone pipeline)
        drone_only_frame = ttk.LabelFrame(self.training_panel, text="Drone vs Bird ML Training (image_process_drone + HOG + Hough)", padding="10")
        drone_only_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(drone_only_frame, text="Train binary classifier using Drone AND Bird images:").pack(anchor=tk.W, pady=(0,5))
        ttk.Label(drone_only_frame, text="Features: HOG + Hough Transform (straight edges)", foreground="gray", font=("TkDefaultFont", 8)).pack(anchor=tk.W, pady=(0,5))
        
        if not DRONE_ONLY_AVAILABLE:
            ttk.Label(drone_only_frame, text="⚠ Warning: Drone-Only ML module not available", 
                     foreground="red", font=("TkDefaultFont", 9)).pack(anchor=tk.W, pady=(0,5))
        
        # Drone dataset selection
        drone_folder_frame = ttk.LabelFrame(drone_only_frame, text="Drone Images Folders", padding="5")
        drone_folder_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        ttk.Label(drone_folder_frame, text="Select Drone Dataset Folders (can select multiple):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        drone_listbox_frame = ttk.Frame(drone_folder_frame)
        drone_listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        drone_scrollbar = ttk.Scrollbar(drone_listbox_frame)
        drone_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.do_drone_folders_listbox = tk.Listbox(drone_listbox_frame, selectmode=tk.EXTENDED, height=3, yscrollcommand=drone_scrollbar.set)
        self.do_drone_folders_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        drone_scrollbar.config(command=self.do_drone_folders_listbox.yview)
        
        # Add default drone folders if available
        if PATHS_AVAILABLE:
            if paths.DRONE_DIR.exists():
                self.do_drone_folders_listbox.insert(tk.END, str(paths.DRONE_DIR))
            default_do_dir = str(paths.GPS_TEST_DIR)
            if os.path.exists(default_do_dir):
                self.do_drone_folders_listbox.insert(tk.END, default_do_dir)
        else:
            drones_dir = os.path.join(os.getcwd(), "data", "images", "Drones")
            if os.path.exists(drones_dir):
                self.do_drone_folders_listbox.insert(tk.END, drones_dir)
        
        drone_buttons_frame = ttk.Frame(drone_folder_frame)
        drone_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(drone_buttons_frame, text="Add Drone Folder", command=self.add_drone_folder).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(drone_buttons_frame, text="Remove Selected", command=self.remove_drone_folder).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(drone_buttons_frame, text="Clear All", command=self.clear_drone_folders).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Bird dataset selection
        bird_folder_frame = ttk.LabelFrame(drone_only_frame, text="Bird Images Folders", padding="5")
        bird_folder_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        ttk.Label(bird_folder_frame, text="Select Bird Dataset Folders (can select multiple):", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        bird_listbox_frame = ttk.Frame(bird_folder_frame)
        bird_listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        bird_scrollbar = ttk.Scrollbar(bird_listbox_frame)
        bird_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.do_bird_folders_listbox = tk.Listbox(bird_listbox_frame, selectmode=tk.EXTENDED, height=3, yscrollcommand=bird_scrollbar.set)
        self.do_bird_folders_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bird_scrollbar.config(command=self.do_bird_folders_listbox.yview)
        
        # Add default bird folders if available
        if PATHS_AVAILABLE:
            if paths.BIRD_DIR.exists():
                self.do_bird_folders_listbox.insert(tk.END, str(paths.BIRD_DIR))
        else:
            birds_dir = os.path.join(os.getcwd(), "data", "images", "Birds")
            if os.path.exists(birds_dir):
                self.do_bird_folders_listbox.insert(tk.END, birds_dir)
        
        bird_buttons_frame = ttk.Frame(bird_folder_frame)
        bird_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(bird_buttons_frame, text="Add Bird Folder", command=self.add_bird_folder).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(bird_buttons_frame, text="Remove Selected", command=self.remove_bird_folder).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(bird_buttons_frame, text="Clear All", command=self.clear_bird_folders).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Training buttons
        train_buttons_frame = ttk.Frame(drone_only_frame)
        train_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(train_buttons_frame, text="Train Binary (Drone vs Bird)", command=self.train_binary_classifier).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(train_buttons_frame, text="Retrain (Clear & Train)", command=self.retrain_binary_classifier).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Status display - Drone-Only ML only
        self.shape_code_status = tk.StringVar(value="Classifier Checked")
        if self.drone_only_classifier and self.drone_only_classifier.is_trained:
            status_text = "Status: Drone-Only ML Ready [OK]"
        else:
            status_text = "Status: Drone-Only ML NOT Trained - Please train the model"
        self.shape_code_status.set(status_text)
        
        # Progress Bar for Training
        progress_frame = ttk.LabelFrame(self.training_panel, text="Training Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', 
                                           variable=self.progress_var, length=300, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready", font=("TkDefaultFont", 9))
        self.progress_label.pack(anchor=tk.W)
        
        # Status display - Drone-Only ML only
        status_frame = ttk.LabelFrame(self.training_panel, text="Training Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, textvariable=self.shape_code_status, wraplength=250).pack(fill=tk.X, pady=2)
        
        # Info Panel (Shared)
        info_frame = ttk.LabelFrame(left_panel, text="System Log", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=10, width=30, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Right Panel - Display (Split into Processing View and Result)
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs (Processing Steps and Result)
        self.result_notebook = ttk.Notebook(self.right_panel)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Processing Steps Tab
        processing_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(processing_frame, text="Processing Steps")
        self.canvas = tk.Canvas(processing_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Result Tab
        result_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(result_frame, text="Result")
        self.result_canvas = tk.Canvas(result_frame, bg="black")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready - Enhanced with sklearn")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Initialize
        self.switch_mode()
        

    def load_saved_preferences(self):
        """Load saved folder preferences from config"""
        if not CONFIG_AVAILABLE:
            return
        
        try:
            config = load_config()
            
            # Load drone folders
            saved_drone_folders = config.get('drone_folders', [])
            if saved_drone_folders and hasattr(self, 'do_drone_folders_listbox'):
                for folder in saved_drone_folders:
                    if os.path.exists(folder):
                        if folder not in self.do_drone_folders_listbox.get(0, tk.END):
                            self.do_drone_folders_listbox.insert(tk.END, folder)
            
            # Load bird folders
            saved_bird_folders = config.get('bird_folders', [])
            if saved_bird_folders and hasattr(self, 'do_bird_folders_listbox'):
                for folder in saved_bird_folders:
                    if os.path.exists(folder):
                        if folder not in self.do_bird_folders_listbox.get(0, tk.END):
                            self.do_bird_folders_listbox.insert(tk.END, folder)
            
            # Load GPS train folder
            saved_gps_folder = config.get('gps_train_folder')
            if saved_gps_folder and os.path.exists(saved_gps_folder):
                self.gps_train_folder = saved_gps_folder
                if hasattr(self, 'gps_folder_label'):
                    self.gps_folder_label.config(text=f"Folder: .../{os.path.basename(saved_gps_folder)}")
        except Exception as e:
            print(f"Error loading preferences: {e}")
    
    def select_folder(self, folder_var, title):
        """Helper function to select a folder"""
        folder = filedialog.askdirectory(title=title)
        if folder:
            folder_var.set(folder)
    
    def add_drone_folder(self):
        """Add a folder to the dataset list"""
        folder = filedialog.askdirectory(title="Select Drone Images Folder")
        if folder:
            # Check if already in list
            items = self.do_drone_folders_listbox.get(0, tk.END)
            if folder not in items:
                self.do_drone_folders_listbox.insert(tk.END, folder)
                # Save preferences
                if CONFIG_AVAILABLE:
                    all_folders = list(self.do_drone_folders_listbox.get(0, tk.END))
                    set_drone_folders(all_folders)
            else:
                messagebox.showinfo("Info", "Folder already in the list")
    
    def remove_drone_folder(self):
        """Remove selected folder(s) from the list"""
        selected_indices = self.do_drone_folders_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select folder(s) to remove")
            return
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.do_drone_folders_listbox.delete(index)
        
        # Save preferences
        if CONFIG_AVAILABLE:
            all_folders = list(self.do_drone_folders_listbox.get(0, tk.END))
            set_drone_folders(all_folders)
    
    def clear_drone_folders(self):
        """Clear all folders from the list"""
        if messagebox.askyesno("Confirm", "Clear all drone folders from the list?"):
            self.do_drone_folders_listbox.delete(0, tk.END)
            # Save preferences
            if CONFIG_AVAILABLE:
                set_drone_folders([])
    
    def add_bird_folder(self):
        """Add a bird folder to the dataset list"""
        folder = filedialog.askdirectory(title="Select Bird Images Folder")
        if folder:
            items = self.do_bird_folders_listbox.get(0, tk.END)
            if folder not in items:
                self.do_bird_folders_listbox.insert(tk.END, folder)
                # Save preferences
                if CONFIG_AVAILABLE:
                    all_folders = list(self.do_bird_folders_listbox.get(0, tk.END))
                    set_bird_folders(all_folders)
            else:
                messagebox.showinfo("Info", "Folder already in the list")
    
    def remove_bird_folder(self):
        """Remove selected bird folder(s) from the list"""
        selected_indices = self.do_bird_folders_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select folder(s) to remove")
            return
        for index in reversed(selected_indices):
            self.do_bird_folders_listbox.delete(index)
        
        # Save preferences
        if CONFIG_AVAILABLE:
            all_folders = list(self.do_bird_folders_listbox.get(0, tk.END))
            set_bird_folders(all_folders)
    
    def clear_bird_folders(self):
        """Clear all bird folders from the list"""
        if messagebox.askyesno("Confirm", "Clear all bird folders from the list?"):
            self.do_bird_folders_listbox.delete(0, tk.END)
            # Save preferences
            if CONFIG_AVAILABLE:
                set_bird_folders([])
    
    def train_binary_classifier(self):
        """Train Binary ML classifier (Drone vs Bird)"""
        if not DRONE_ONLY_AVAILABLE or self.drone_only_classifier is None:
            messagebox.showerror("Error", "Drone-Only ML module not available.")
            return
        
        # Check if model already exists
        if PATHS_AVAILABLE and paths.BIRD_V_DRONE_MODEL.exists():
            if self.drone_only_classifier.is_trained:
                response = messagebox.askyesnocancel(
                    "Model Already Exists",
                    f"Model already exists at:\n{paths.BIRD_V_DRONE_MODEL}\n\n"
                    f"Current model is trained.\n\n"
                    f"Would you like to:\n"
                    f"- Yes: Retrain (overwrite existing model)\n"
                    f"- No: Use existing model (skip training)\n"
                    f"- Cancel: Do nothing"
                )
                if response is None:  # Cancel
                    return
                elif response is False:  # No - use existing
                    messagebox.showinfo("Info", "Using existing model. No training needed.")
                    self.update_model_list()
                    self.update_video_model_list()
                    return
                # If Yes, continue with training
        
        # Get all selected folders
        drone_dirs = list(self.do_drone_folders_listbox.get(0, tk.END))
        bird_dirs = list(self.do_bird_folders_listbox.get(0, tk.END))
        
        if not drone_dirs:
            messagebox.showerror("Error", "Please add at least one Drone dataset folder")
            return
        
        if not bird_dirs:
            messagebox.showerror("Error", "Please add at least one Bird dataset folder")
            return
        
        # Validate all folders exist
        for drone_dir in drone_dirs:
            if not os.path.exists(drone_dir):
                messagebox.showerror("Error", f"Drone folder not found: {drone_dir}")
                return
        
        for bird_dir in bird_dirs:
            if not os.path.exists(bird_dir):
                messagebox.showerror("Error", f"Bird folder not found: {bird_dir}")
                return
        
        # Save preferences
        if CONFIG_AVAILABLE:
            set_drone_folders(drone_dirs)
            set_bird_folders(bird_dirs)
        
        def _train():
            try:
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Initializing..."))
                self.root.after(0, lambda: self.status_var.set("Training Binary ML Classifier (Drone vs Bird)... This may take a while."))
                self.root.update()
                
                def count_images(directories):
                    total = 0
                    for directory in directories:
                        if os.path.exists(directory):
                            for fname in os.listdir(directory):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                    total += 1
                    return total
                
                total_drone_images = count_images(drone_dirs)
                total_bird_images = count_images(bird_dirs)
                total_images = total_drone_images + total_bird_images
                
                if total_drone_images == 0:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No drone images found in the specified folders."))
                    return
                
                if total_bird_images == 0:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No bird images found in the specified folders."))
                    return
                
                self.root.after(0, lambda: self.progress_label.config(text=f"Found {total_drone_images} drone + {total_bird_images} bird images. Starting processing..."))
                self.root.after(0, lambda: self.progress_var.set(5))
                self.root.update()
                
                def update_progress(current, total, message):
                    try:
                        if total > 0:
                            if "Training classifier" in message:
                                progress_pct = 90
                            elif "Saving model" in message:
                                progress_pct = 95
                            elif "Training Complete" in message:
                                progress_pct = 100
                            else:
                                progress_pct = 5 + int((current / total) * 85)
                            self.root.after(0, lambda p=progress_pct: self.progress_var.set(p))
                            self.root.after(0, lambda m=message: self.progress_label.config(text=m))
                    except Exception as e:
                        print(f"Progress update error: {e}")
                
                # Train binary classifier
                success = self.drone_only_classifier.train_binary(
                    drone_dirs,
                    bird_dirs,
                    progress_callback=update_progress
                )
                
                self.root.after(0, lambda: self.progress_var.set(95))
                self.root.after(0, lambda: self.progress_label.config(text="Finalizing..."))
                
                if success:
                    # Save model to models folder automatically
                    if PATHS_AVAILABLE:
                        # Ensure models directory exists
                        paths.ensure_structure()
                        model_path = paths.BIRD_V_DRONE_MODEL
                        if self.drone_only_classifier.save(model_path):
                            print(f"[OK] Model saved to {model_path}")
                        else:
                            print(f"[WARNING] Failed to save model to {model_path}")
                    else:
                        self.drone_only_classifier.save()
                    
                    # Save preferences
                    if CONFIG_AVAILABLE:
                        set_drone_folders(drone_dirs)
                        set_bird_folders(bird_dirs)
                    
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.progress_label.config(text="Training Complete! [OK]"))
                    model_location = str(paths.BIRD_V_DRONE_MODEL) if PATHS_AVAILABLE else "default location"
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        f"Binary ML Classifier trained successfully!\n"
                        f"Features: HOG + Hough Transform\n"
                        f"Pipeline: image_process_drone\n"
                        f"Drone datasets: {len(drone_dirs)} folder(s)\n"
                        f"Bird datasets: {len(bird_dirs)} folder(s)\n\n"
                        f"Model saved to:\n{model_location}"))
                    # Update status
                    self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: self.update_video_model_list())
                    if self.drone_only_classifier.is_trained:
                        status_text = self.shape_code_status.get()
                        if "Drone-Only ML Ready [OK]" not in status_text:
                            new_status = status_text.replace("Drone-Only ML NOT Trained", "Drone-Only ML Ready [OK] (Binary Mode)")
                            self.root.after(0, lambda: self.shape_code_status.set(new_status))
                else:
                    self.root.after(0, lambda: self.progress_var.set(0))
                    self.root.after(0, lambda: self.progress_label.config(text="Training Failed"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Binary ML Classifier training failed."))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Error occurred"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Training error: {msg}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        training_thread = threading.Thread(target=_train, daemon=True)
        training_thread.start()
    
    def retrain_binary_classifier(self):
        """Clear existing model and train binary classifier again"""
        if not DRONE_ONLY_AVAILABLE or self.drone_only_classifier is None:
            messagebox.showerror("Error", "Drone-Only ML module not available.")
            return
        
        drone_dirs = list(self.do_drone_folders_listbox.get(0, tk.END))
        bird_dirs = list(self.do_bird_folders_listbox.get(0, tk.END))
        
        if not drone_dirs or not bird_dirs:
            messagebox.showerror("Error", "Please add at least one Drone and one Bird dataset folder")
            return
        
        for drone_dir in drone_dirs:
            if not os.path.exists(drone_dir):
                messagebox.showerror("Error", f"Drone folder not found: {drone_dir}")
                return
        
        for bird_dir in bird_dirs:
            if not os.path.exists(bird_dir):
                messagebox.showerror("Error", f"Bird folder not found: {bird_dir}")
                return
        
        if not messagebox.askyesno("Confirm Retrain", 
            "This will clear the existing model and train a new binary classifier.\n\nContinue?"):
            return
        
        def _retrain():
            try:
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Clearing existing model..."))
                self.root.after(0, lambda: self.status_var.set("Retraining Binary ML Classifier... This may take a while."))
                self.root.update()
                
                from sklearn.preprocessing import StandardScaler
                self.drone_only_classifier.is_trained = False
                self.drone_only_classifier.is_binary = False
                self.drone_only_classifier.classifier = None
                self.drone_only_classifier.scaler = StandardScaler()
                
                def count_images(directories):
                    total = 0
                    for directory in directories:
                        if os.path.exists(directory):
                            for fname in os.listdir(directory):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                    total += 1
                    return total
                
                total_drone_images = count_images(drone_dirs)
                total_bird_images = count_images(bird_dirs)
                
                if total_drone_images == 0 or total_bird_images == 0:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No images found in the specified folders."))
                    return
                
                self.root.after(0, lambda: self.progress_label.config(text=f"Found {total_drone_images} drone + {total_bird_images} bird images. Starting processing..."))
                self.root.after(0, lambda: self.progress_var.set(5))
                self.root.update()
                
                def update_progress(current, total, message):
                    try:
                        if total > 0:
                            if "Training classifier" in message:
                                progress_pct = 90
                            elif "Saving model" in message:
                                progress_pct = 95
                            elif "Training Complete" in message:
                                progress_pct = 100
                            else:
                                progress_pct = 5 + int((current / total) * 85)
                            self.root.after(0, lambda p=progress_pct: self.progress_var.set(p))
                            self.root.after(0, lambda m=message: self.progress_label.config(text=m))
                    except Exception as e:
                        print(f"Progress update error: {e}")
                
                success = self.drone_only_classifier.train_binary(
                    drone_dirs,
                    bird_dirs,
                    progress_callback=update_progress
                )
                
                self.root.after(0, lambda: self.progress_var.set(95))
                self.root.after(0, lambda: self.progress_label.config(text="Finalizing..."))
                
                if success:
                    # Save model to models folder automatically
                    if PATHS_AVAILABLE:
                        paths.ensure_structure()
                        model_path = paths.BIRD_V_DRONE_MODEL
                        if self.drone_only_classifier.save(model_path):
                            print(f"[OK] Model saved to {model_path}")
                        else:
                            print(f"[WARNING] Failed to save model to {model_path}")
                    else:
                        self.drone_only_classifier.save()
                    
                    # Save preferences
                    if CONFIG_AVAILABLE:
                        set_drone_folders(drone_dirs)
                        set_bird_folders(bird_dirs)
                    
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.progress_label.config(text="Retraining Complete! [OK]"))
                    model_location = str(paths.BIRD_V_DRONE_MODEL) if PATHS_AVAILABLE else "default location"
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        f"Binary ML Classifier retrained successfully!\n"
                        f"Features: HOG + Hough Transform\n"
                        f"Pipeline: image_process_drone\n"
                        f"Drone datasets: {len(drone_dirs)} folder(s)\n"
                        f"Bird datasets: {len(bird_dirs)} folder(s)\n\n"
                        f"Model saved to:\n{model_location}"))
                    self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: self.update_video_model_list())
                    if self.drone_only_classifier.is_trained:
                        status_text = self.shape_code_status.get()
                        if "Drone-Only ML Ready [OK]" not in status_text:
                            new_status = status_text.replace("Drone-Only ML NOT Trained", "Drone-Only ML Ready [OK] (Binary Mode)")
                            self.root.after(0, lambda: self.shape_code_status.set(new_status))
                else:
                    self.root.after(0, lambda: self.progress_var.set(0))
                    self.root.after(0, lambda: self.progress_label.config(text="Retraining Failed"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Binary ML Classifier retraining failed."))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Error occurred"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Retraining error: {msg}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        retraining_thread = threading.Thread(target=_retrain, daemon=True)
        retraining_thread.start()
    
    def train_drone_only_classifier(self):
        """Train Drone-Only ML classifier using image_process_drone pipeline"""
        if not DRONE_ONLY_AVAILABLE or self.drone_only_classifier is None:
            messagebox.showerror("Error", "Drone-Only ML module not available.")
            return
        
        # Get all selected folders
        drone_dirs = list(self.do_drone_folders_listbox.get(0, tk.END))
        
        if not drone_dirs:
            messagebox.showerror("Error", "Please add at least one dataset folder")
            return
        
        # Validate all folders exist
        for drone_dir in drone_dirs:
            if not os.path.exists(drone_dir):
                messagebox.showerror("Error", f"Folder not found: {drone_dir}")
                return
        
        def _train():
            try:
                # Reset progress
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Initializing..."))
                self.root.after(0, lambda: self.status_var.set("Training Drone-Only ML Classifier... This may take a while."))
                self.root.update()
                
                # Count total images for progress calculation
                def count_images(directories):
                    total = 0
                    for directory in directories:
                        if os.path.exists(directory):
                            for fname in os.listdir(directory):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                    total += 1
                    return total
                
                total_images = count_images(drone_dirs)
                
                if total_images == 0:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No images found in the specified folders."))
                    return
                
                # Update progress
                self.root.after(0, lambda: self.progress_label.config(text=f"Found {total_images} images. Starting processing..."))
                self.root.after(0, lambda: self.progress_var.set(5))
                self.root.update()
                
                # Define progress callback for training
                def update_progress(current, total, message):
                    """Update progress bar from training thread"""
                    try:
                        if total > 0:
                            # 5-90% for processing images, 90-95% for training, 95-100% for saving
                            if "Training classifier" in message:
                                progress_pct = 90
                            elif "Saving model" in message:
                                progress_pct = 95
                            elif "Training Complete" in message:
                                progress_pct = 100
                            else:
                                # Processing images: 5-90%
                                progress_pct = 5 + int((current / total) * 85)
                            
                            # Update UI in main thread
                            self.root.after(0, lambda p=progress_pct: self.progress_var.set(p))
                            self.root.after(0, lambda m=message: self.progress_label.config(text=m))
                    except Exception as e:
                        print(f"Progress update error: {e}")
                
                # Train with multiple directories (this will process images internally using image_process_drone)
                success = self.drone_only_classifier.train_multiple(
                    drone_dirs,
                    progress_callback=update_progress
                )
                
                self.root.after(0, lambda: self.progress_var.set(95))
                self.root.after(0, lambda: self.progress_label.config(text="Finalizing..."))
                
                if success:
                    self.drone_only_classifier.save()
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.progress_label.config(text="Training Complete! [OK]"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        f"Drone-Only ML Classifier trained successfully!\n"
                        f"Features: HOG + Hough Transform\n"
                        f"Pipeline: image_process_drone"))
                    # Update status
                    self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: self.update_video_model_list())
                    if self.drone_only_classifier.is_trained:
                        status_text = self.shape_code_status.get()
                        if "Drone-Only ML Ready [OK]" not in status_text:
                            new_status = status_text.replace("Drone-Only ML NOT Trained", "Drone-Only ML Ready [OK]")
                            self.root.after(0, lambda: self.shape_code_status.set(new_status))
                else:
                    self.root.after(0, lambda: self.progress_var.set(0))
                    self.root.after(0, lambda: self.progress_label.config(text="Training Failed"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Drone-Only ML Classifier training failed."))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Error occurred"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Training error: {msg}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        training_thread = threading.Thread(target=_train, daemon=True)
        training_thread.start()
    
    def retrain_drone_only_classifier(self):
        """Clear existing model and train again"""
        if not DRONE_ONLY_AVAILABLE or self.drone_only_classifier is None:
            messagebox.showerror("Error", "Drone-Only ML module not available.")
            return
        
        # Get all selected folders
        drone_dirs = list(self.do_drone_folders_listbox.get(0, tk.END))
        
        if not drone_dirs:
            messagebox.showerror("Error", "Please add at least one dataset folder")
            return
        
        # Validate all folders exist
        for drone_dir in drone_dirs:
            if not os.path.exists(drone_dir):
                messagebox.showerror("Error", f"Folder not found: {drone_dir}")
                return
        
        # Confirm retraining
        if not messagebox.askyesno("Confirm Retrain", 
            "This will clear the existing model and train a new one.\n\nContinue?"):
            return
        
        def _retrain():
            try:
                # Reset progress
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Clearing existing model..."))
                self.root.after(0, lambda: self.status_var.set("Retraining Drone-Only ML Classifier... This may take a while."))
                self.root.update()
                
                # Clear existing model
                from sklearn.preprocessing import StandardScaler
                self.drone_only_classifier.is_trained = False
                self.drone_only_classifier.classifier = None
                self.drone_only_classifier.scaler = StandardScaler()
                
                # Count total images for progress calculation
                def count_images(directories):
                    total = 0
                    for directory in directories:
                        if os.path.exists(directory):
                            for fname in os.listdir(directory):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                    total += 1
                    return total
                
                total_images = count_images(drone_dirs)
                
                if total_images == 0:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No images found in the specified folders."))
                    return
                
                # Update progress
                self.root.after(0, lambda: self.progress_label.config(text=f"Found {total_images} images. Starting processing..."))
                self.root.after(0, lambda: self.progress_var.set(5))
                self.root.update()
                
                # Define progress callback for training
                def update_progress(current, total, message):
                    """Update progress bar from training thread"""
                    try:
                        if total > 0:
                            # 5-90% for processing images, 90-95% for training, 95-100% for saving
                            if "Training classifier" in message:
                                progress_pct = 90
                            elif "Saving model" in message:
                                progress_pct = 95
                            elif "Training Complete" in message:
                                progress_pct = 100
                            else:
                                # Processing images: 5-90%
                                progress_pct = 5 + int((current / total) * 85)
                            
                            # Update UI in main thread
                            self.root.after(0, lambda p=progress_pct: self.progress_var.set(p))
                            self.root.after(0, lambda m=message: self.progress_label.config(text=m))
                    except Exception as e:
                        print(f"Progress update error: {e}")
                
                # Train with multiple directories
                success = self.drone_only_classifier.train_multiple(
                    drone_dirs,
                    progress_callback=update_progress
                )
                
                self.root.after(0, lambda: self.progress_var.set(95))
                self.root.after(0, lambda: self.progress_label.config(text="Finalizing..."))
                
                if success:
                    self.drone_only_classifier.save()
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.progress_label.config(text="Retraining Complete! [OK]"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        f"Drone-Only ML Classifier retrained successfully!\n"
                        f"Features: HOG + Hough Transform\n"
                        f"Pipeline: image_process_drone\n"
                        f"Datasets: {len(drone_dirs)} folder(s)"))
                    # Update status
                    self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: self.update_video_model_list())
                    if self.drone_only_classifier.is_trained:
                        status_text = self.shape_code_status.get()
                        if "Drone-Only ML Ready [OK]" not in status_text:
                            new_status = status_text.replace("Drone-Only ML NOT Trained", "Drone-Only ML Ready [OK]")
                            self.root.after(0, lambda: self.shape_code_status.set(new_status))
                else:
                    self.root.after(0, lambda: self.progress_var.set(0))
                    self.root.after(0, lambda: self.progress_label.config(text="Retraining Failed"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Drone-Only ML Classifier retraining failed."))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Error occurred"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Retraining error: {msg}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        retraining_thread = threading.Thread(target=_retrain, daemon=True)
        retraining_thread.start()
    
    # ================================
    # Basic Mode Helper
    # ================================
    def switch_mode(self):
        """สลับ mode"""
        mode = self.mode_var.get()
        
        # Hide all panels first
        self.image_panel.pack_forget()
        self.video_panel.pack_forget()
        if hasattr(self, 'training_container'):
            self.training_container.pack_forget()
        
        # Hide/show right panel (preview) based on mode
        if hasattr(self, 'right_panel'):
            if mode == "training":
                # Hide preview/result panel in training mode
                self.right_panel.grid_remove()
            else:
                # Show preview/result panel for image/video mode
                self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Stop video if playing
        if self.is_playing_video:
            self.stop_video()
        
        if mode == "image":
            self.image_panel.pack(fill=tk.BOTH, expand=True)
        elif mode == "video":
            self.video_panel.pack(fill=tk.BOTH, expand=True)
        elif mode == "training":
            self.training_container.pack(fill=tk.BOTH, expand=True)
            if self.gps_model and self.gps_model.is_trained:
                self.save_model_btn.config(state=tk.NORMAL)
        
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")
        if hasattr(self, 'result_canvas'):
            self.result_canvas.delete("all")
        if hasattr(self, 'info_text'):
            self.info_text.delete(1.0, tk.END)

    def select_gps_folder(self):
        """Select folder for GPS training"""
        folder = filedialog.askdirectory(title="Select GPS Training Folder")
        if folder:
            self.gps_train_folder = folder
            self.gps_folder_label.config(text=f"Folder: .../{os.path.basename(folder)}")
            self.status_var.set(f"GPS Training folder set to: {os.path.basename(folder)}")
            # Save preference
            if CONFIG_AVAILABLE:
                set_gps_train_folder(folder)

    def _on_detection_mode_change(self):
        """Update UI when detection mode changes"""
        mode = self.detection_mode.get()
        if mode == "logic":
            self.detection_mode_desc.config(
                text="Rule-based detection using thresholding, morphology, and area filtering. No ML model required."
            )
            self.model_status_label.config(text="")
        else:  # model
            self.detection_mode_desc.config(
                text="Drone-Only ML Classifier with HOG and Hough Transform features (image_process_drone pipeline)"
            )
            # Update model status
            if self.drone_only_classifier and self.drone_only_classifier.is_trained:
                self.model_status_label.config(text="Status: Drone-Only ML Ready [OK]", foreground="green")
            else:
                self.model_status_label.config(text="Status: Drone-Only ML NOT Trained - Please train the model", foreground="red")
    
    def on_detection_mode_change(self):
        """Handle detection mode change - supports both Logic and ML modes"""
        # Update UI based on selected mode
        self._on_detection_mode_change()
        # Update model list if in ML mode
        if self.detection_mode.get() == "model":
            self.update_model_list()
        
        # Reprocess if image is loaded
        if self.mode_var.get() == "image" and self.current_image_path:
            self.process_image(self.current_image_path)
    
    def _on_video_detection_mode_change(self):
        """Update UI when video detection mode changes"""
        mode = self.video_detection_mode.get()
        if mode == "logic":
            self.video_detection_mode_desc.config(
                text="Rule-based detection using thresholding, morphology, and area filtering. No ML model required."
            )
            self.video_model_status_label.config(text="")
        else:  # model
            self.video_detection_mode_desc.config(
                text="Drone-Only ML Classifier with HOG and Hough Transform features (image_process_drone pipeline)"
            )
            # Update model status
            if self.drone_only_classifier and self.drone_only_classifier.is_trained:
                self.video_model_status_label.config(text="Status: Drone-Only ML Ready [OK]", foreground="green")
            else:
                self.video_model_status_label.config(text="Status: Drone-Only ML NOT Trained - Please train the model", foreground="red")
    
    def on_video_detection_mode_change(self):
        """Handle video detection mode change - supports both Logic and ML modes"""
        # Update UI based on selected mode
        self._on_video_detection_mode_change()
        # Update model list if in ML mode
        if self.video_detection_mode.get() == "model":
            self.update_video_model_list()
    
    def update_video_model_list(self):
        """Update video model status - Drone-Only ML only"""
        # Update status label
        if hasattr(self, 'video_model_status_label'):
            if self.drone_only_classifier and self.drone_only_classifier.is_trained:
                self.video_model_status_label.config(text="Status: Drone-Only ML Ready [OK]", foreground="green")
            else:
                self.video_model_status_label.config(text="Status: Drone-Only ML NOT Trained - Please train the model", foreground="red")
        
        # Set model name
        self.video_selected_classifier_model.set("Drone-Only ML (image_process_drone + HOG + Hough)")
    
    def on_video_model_selected(self, event=None):
        """Handle video model selection"""
        pass  # Can be used for any action needed when model is selected
    
    def update_model_list(self):
        """Update model status - Drone-Only ML only"""
        # Update status label
        if hasattr(self, 'model_status_label'):
            if self.drone_only_classifier and self.drone_only_classifier.is_trained:
                self.model_status_label.config(text="Status: Drone-Only ML Ready [OK]", foreground="green")
            else:
                self.model_status_label.config(text="Status: Drone-Only ML NOT Trained - Please train the model", foreground="red")
        
        # Set model name
        self.selected_classifier_model.set("Drone-Only ML (image_process_drone + HOG + Hough)")
        
        # Update shape model status in image panel
        if hasattr(self, 'shape_model_status_img'):
            if self.drone_only_classifier and self.drone_only_classifier.is_trained:
                self.shape_model_status_img.set("Drone-Only ML: Ready [OK]")
            else:
                self.shape_model_status_img.set("Drone-Only ML: NOT Trained")
    
    def on_model_selected(self, event=None):
        """Handle model selection"""
        # Reprocess if image is loaded
        if self.mode_var.get() == "image" and self.current_image_path:
            self.process_image(self.current_image_path)
    

    # ================================
    # Helper Functions
    # ================================
    
    def _update_model_status(self, text):
        """Centralized model status update"""
        self.img_model_status.set(text)
    
    def _show_error(self, title, message):
        """Show error with status update"""
        messagebox.showerror(title, message)
        self.status_var.set(f"Error: {message}")
    
    def _show_success(self, title, message):
        """Show success with status update"""
        messagebox.showinfo(title, message)
        self.status_var.set(message)
    
    def _show_info(self, title, message):
        """Show info with status update"""
        messagebox.showinfo(title, message)
        self.status_var.set(message)
    
    def select_image(self):
        """เลือกภาพ"""
        initial_dir = os.getcwd()
        if self.mode_var.get() == "training":
            # Use paths from paths.py if available
            if PATHS_AVAILABLE:
                train_dir = str(paths.GPS_TRAIN_DIR)
            else:
                # Fallback: try new structure first, then old
                train_dir = os.path.join(os.getcwd(), "data", "gps", "P2_DATA_TRAIN")
                if not os.path.exists(train_dir):
                    train_dir = os.path.join(os.getcwd(), "P2_DATA_TRAIN")
            if os.path.exists(train_dir):
                initial_dir = train_dir
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=initial_dir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Update label based on mode
            if self.mode_var.get() == "training":
                pass # self.train_label.config(text=os.path.basename(file_path)) # Not using train_label here
            else:
                self.img_label.config(text=os.path.basename(file_path))
                
            self.status_var.set("Processing image... (Please wait)")
            self.root.config(cursor="watch")
            self.root.update()
            
            # Start processing in a thread to prevent freezing
            threading.Thread(target=self._process_image_thread, args=(file_path,), daemon=True).start()

    def _update_canvas_from_thread(self, img_rgb, canvas_w, canvas_h):
        """Helper to update canvas from thread data"""
        try:
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_w//2, canvas_h//2, image=img_tk, anchor=tk.CENTER)
            self.canvas.image = img_tk  # Keep reference
        except Exception as e:
            print(f"Canvas update error: {e}")
    
    def select_video(self):
        """เลือกวิดีโอ"""
        initial_dir = os.getcwd()
        # Use paths from paths.py if available
        if PATHS_AVAILABLE:
            vids_dir = str(paths.VIDEO_SAMPLE_DIR)
        else:
            # Fallback: try new structure first, then old
            vids_dir = os.path.join(os.getcwd(), "data", "videos", "vids")
            if not os.path.exists(vids_dir):
                vids_dir = os.path.join(os.getcwd(), "vids")
        if os.path.exists(vids_dir):
            initial_dir = vids_dir
        
        file_path = filedialog.askopenfilename(
            title="Select Video",
            initialdir=initial_dir,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_video_path = file_path
            self.video_label.config(text=os.path.basename(file_path))
            
            # เปิดวิดีโอเพื่อดูข้อมูล
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / self.video_fps if self.video_fps > 0 else 0
                
                self.video_info_label.config(
                    text=f"FPS: {self.video_fps:.1f}\nDuration: {duration:.1f}s\nFrames: {frame_count}"
                )
                
                # อ่าน frame แรกเพื่อแสดง
                ret, frame = cap.read()
                if ret:
                    self.display_image(frame)
                
                cap.release()
                self.play_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Video loaded: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "Cannot open video file")
                self.status_var.set("Error: Cannot open video")
    
    def play_video(self):
        """เล่นวิดีโอ"""
        if not self.current_video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return
        
        if self.is_playing_video:
            return
        
        try:
            # สร้าง video processor ถ้ายังไม่มี
            if self.video_processor is None:
                self.video_processor = VideoProcessor(
                    self.tracker, 
                    self.gps_model,
                    self.base_lat,
                    self.base_lon,
                    self.base_alt
                )
            else:
                self.video_processor.reset()
                self.tracker.reset()
            
            self.is_playing_video = True
            self.play_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Playing video...")
            
            # รันใน thread แยก
            self.video_thread = threading.Thread(target=self._video_processing_loop, daemon=True)
            self.video_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start video: {str(e)}")
            self.is_playing_video = False
            self.play_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def stop_video(self):
        """หยุดวิดีโอ"""
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Video stopped")
    
    def _video_processing_loop(self):
        """Loop สำหรับประมวลผลวิดีโอ"""
        try:
            self.video_cap = cv2.VideoCapture(self.current_video_path)
            if not self.video_cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Cannot open video file"))
                self.is_playing_video = False
                self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                return
            
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 24.0  # default FPS
            frame_delay = 1.0 / fps
            
            frame_number = 0
            
            while self.is_playing_video:
                ret, frame = self.video_cap.read()
                if not ret:
                    # วิดีโอจบแล้ว
                    self.root.after(0, self.stop_video)
                    break
                
                if frame is None:
                    continue
                
                # ประมวลผล frame
                try:
                    detections, _ = self._detect_drones(frame, for_video=True)
                    
                    # ใช้ video processor
                    if self.video_processor is None:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Video processor not initialized"))
                        break
                    
                    processed_frame, track_info, timestamp = self.video_processor.process_frame(
                        frame, frame_number, fps, detections, self.using_custom_model
                    )
                except Exception as e:
                    print(f"Error processing frame {frame_number}: {e}")
                    import traceback
                    traceback.print_exc()
                    processed_frame = frame.copy()
                    track_info = {}
                
                # วาดข้อมูลโดรนด้านซ้ายบน
                info_y_start = 30
                for track_id, info in sorted(track_info.items()):
                    color = info['color']
                    y_offset = info_y_start + (track_id * 80)
                    
                    cv2.putText(processed_frame, f"track_id:{track_id}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(processed_frame, f"lat:{info['lat']:.5f}", (10, y_offset + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(processed_frame, f"lon:{info['lon']:.5f}", (10, y_offset + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(processed_frame, f"alt:{info['alt']:.2f}", (10, y_offset + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(processed_frame, f"speed:{info['speed']:.2f} m/s", (10, y_offset + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # แสดง frame (แก้ปัญหา lambda closure และลดภาระ Main Thread)
                # Resize ใน thread นี้เลยเพื่อลดภาระ Main Thread
                try:
                    # Get approximate canvas size (thread-safe enough for this)
                    cv_w = self.canvas.winfo_width()
                    cv_h = self.canvas.winfo_height()
                    if cv_w <= 1: cv_w = 800
                    if cv_h <= 1: cv_h = 600
                    
                    img_h, img_w = processed_frame.shape[:2]
                    scale = min(cv_w / img_w, cv_h / img_h, 1.0)
                    new_w = int(img_w * scale)
                    new_h = int(img_h * scale)
                    
                    resized_frame = cv2.resize(processed_frame, (new_w, new_h))
                    resized_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    
                    # Send prepared image data to main thread
                    self.root.after(0, lambda img_data=resized_rgb, w=cv_w, h=cv_h: self._update_canvas_from_thread(img_data, w, h))
                except Exception as e:
                     print(f"Resize error: {e}")
                
                # อัปเดต info text (แก้ปัญหา lambda closure)
                info_text = f"=== Video Processing ===\n\n"
                info_text += f"Frame: {frame_number}\n"
                info_text += f"FPS: {fps:.1f}\n"
                info_text += f"Tracks: {len(track_info)}\n\n"
                for track_id, info in sorted(track_info.items()):
                    info_text += f"Track {track_id}:\n"
                    info_text += f"  Lat: {info['lat']:.5f}\n"
                    info_text += f"  Lon: {info['lon']:.5f}\n"
                    info_text += f"  Alt: {info['alt']:.2f} m\n"
                    info_text += f"  Speed: {info['speed']:.2f} m/s\n\n"
                
                info_text_copy = info_text
                self.root.after(0, lambda: self._update_info_text(info_text_copy))
                
                frame_number += 1
                
                # หน่วงเวลา
                time.sleep(frame_delay * 0.8) # Sleep slightly less to account for processing time
            
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
                
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"Video processing error: {str(e)}"))
            self.is_playing_video = False
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
    
    def _update_info_text(self, text):
        """อัปเดต info text"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
    
    def _process_image_thread(self, img_path):
        """Wrapper to run process_image in thread"""
        try:
            self.process_image(img_path)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing error: {e}"))
        finally:
            self.root.after(0, lambda: self.root.config(cursor=""))

    def process_image(self, img_path):
        """ประมวลผลภาพ (Actual Logic)"""
        try:
            # ใช้ numpy array decoding เพื่อรองรับภาษาไทยหรือ path ยาวๆ
            # image = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            try:
                stream = open(img_path, "rb")
                bytes = bytearray(stream.read())
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                stream.close()
            except Exception as e:
                error_msg = f"Cannot read image file: {str(e)}"
                print(f"Image read error: {error_msg}")
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.root.after(0, lambda: self.status_var.set("Error: Cannot read image"))
                return

            if image is None:
                error_msg = f"Cannot decode image: {img_path}"
                print(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.root.after(0, lambda: self.status_var.set("Error: Invalid image file"))
                return
            
            # ... (Existing GPS Reading Logic remains similar but carefully handled) ...
            self.gps_data = None
            csv_path = img_path.replace('.jpg', '.csv')
            if not os.path.exists(csv_path):
                csv_path = img_path.replace('.jpg', '.csv').replace('P2_DATA_TEST', 'P2_DATA_TRAIN')
            
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            self.gps_data = row
                            break
                except Exception:
                    pass

            # ประมวลผล
            try:
                detections, intermediates = self._detect_drones(image)
            except Exception as e:
                error_msg = f"Detection error: {str(e)}"
                print(f"Detection error: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {error_msg}"))
                self.root.after(0, lambda: self.status_var.set("Error: Detection failed"))
                return
            
            # เก็บขั้นตอน (ทำใน Thread นี้ได้เลย)
            try:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Create copies specifically for UI to consume later
                new_steps = {}
                new_steps['original'] = image.copy()
                new_steps['grayscale'] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                
                # Safely access intermediates with defaults
                th1 = intermediates.get('th1', np.zeros_like(img_gray))
                th1_inv = intermediates.get('th1_inv', np.zeros_like(img_gray))
                dilated_inv = intermediates.get('dilated_inv', np.zeros_like(img_gray))
                contours = intermediates.get('contours', [])
                
                new_steps['threshold'] = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
                new_steps['inversion'] = cv2.cvtColor(th1_inv, cv2.COLOR_GRAY2BGR)
                new_steps['dilation'] = cv2.cvtColor(dilated_inv, cv2.COLOR_GRAY2BGR)
                
                step5 = image.copy()
                if contours:
                    cv2.drawContours(step5, contours, -1, (0, 255, 255), 2)
                new_steps['contours'] = step5
            except Exception as e:
                error_msg = f"Step visualization error: {str(e)}"
                print(f"Visualization error: {error_msg}")
                # Create minimal steps if visualization fails
                new_steps = {
                    'original': image.copy(),
                    'grayscale': cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
                    'threshold': image.copy(),
                    'inversion': image.copy(),
                    'dilation': image.copy(),
                    'contours': image.copy(),
                    'shape_classification': image.copy(),
                    'final': image.copy()
                }
            
            # Shape Classification visualization (แสดงเฉพาะ drone ที่ผ่าน filtering)
            step6 = image.copy()
            shape_classifications = intermediates.get('shape_classifications', [])
            drone_count = 0
            
            # Check if using ML mode
            using_logic_mode = not any(s.get('used_classifier', False) for s in shape_classifications)
            
            try:
                for shape_info in shape_classifications:
                    try:
                        c = shape_info.get('contour')
                        if c is None:
                            continue
                            
                        corner_x = shape_info.get('corner_x', 0)
                        corner_y = shape_info.get('corner_y', 0)
                        center_x = shape_info.get('center_x', 0)
                        center_y = shape_info.get('center_y', 0)
                        cw = shape_info.get('cw', 0)
                        ch = shape_info.get('ch', 0)
                        is_drone = shape_info.get('is_drone', False)
                        prob = shape_info.get('prob', 0.0)
                        
                        # In ML mode: Only show drones (filter out birds and clouds)
                        # In Logic mode: Show all detected objects as drones
                        if using_logic_mode:
                            # Logic mode: all objects are considered drones
                            drone_count += 1
                            color, label = (0, 255, 0), f"DRONE {prob:.2f}"
                            print(f"Drawing DRONE (Logic): prob={prob:.3f}, is_drone={is_drone}")
                        else:
                            # ML mode: Only show if classified as drone
                            if is_drone:
                                drone_count += 1
                                color, label = (0, 255, 0), f"DRONE {prob:.2f}"
                                print(f"Drawing DRONE (ML): prob={prob:.3f}, is_drone={is_drone}")
                            else:
                                # Skip birds and non-drones in ML mode - don't draw them
                                continue
                        
                        cv2.drawContours(step6, [c], -1, color, 2)
                        cv2.rectangle(step6, (corner_x, corner_y), (corner_x + cw, corner_y + ch), color, 2)
                        cv2.putText(step6, label, (corner_x, max(20, corner_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"Error drawing shape classification: {e}")
                        continue
            except Exception as e:
                print(f"Error in shape classification visualization: {e}")

            # แสดงผลลัพธ์ - แสดงเฉพาะ drone ที่ผ่าน filtering แล้ว
            # ใน ML mode จะแสดงเฉพาะ drone ที่ ML classify เป็น drone ด้วย confidence สูง
            # Bird และ cloud จะถูกกรองออกแล้ว
            if using_logic_mode:
                # Logic mode: all detected objects are drones
                summary_text = f"Detected: {drone_count} DRONE(s) (Logic-Based Detection)"
            else:
                # ML mode: shows only drones (birds and clouds are filtered out)
                summary_text = f"Detected: {drone_count} DRONE(s) (ML-Based Detection)"
            cv2.putText(step6, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            new_steps['shape_classification'] = step6
            
            # Final detection
            final_image = image.copy()
            h, w = image.shape[:2]
            detections.sort(key=lambda x: x['area'], reverse=True)
            detections = detections[:2]
            
            colors = [(0, 255, 0), (0, 255, 255)]
            info_text_accum = f"=== Detection Results ===\n\nDetected: {len(detections)} drones\n\n"

            for i, det in enumerate(detections):
                color = colors[i % len(colors)]
                cv2.rectangle(final_image, (det['corner_x'], det['corner_y']), 
                             (det['corner_x'] + det['cw'], det['corner_y'] + det['ch']), color, 2)
                
                # GPS
                lat, lon, alt = self.calculate_gps(det['center_x'], det['center_y'], h, w)
                
                # Display ID and GPS info on image
                text_y_start = det['corner_y'] - 5
                cv2.putText(final_image, f"ID:{i}", (det['corner_x'], text_y_start),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display lat, lon, alt below the bounding box
                text_y = det['corner_y'] + det['ch'] + 15
                cv2.putText(final_image, f"Lat: {lat:.5f}", (det['corner_x'], text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(final_image, f"Lon: {lon:.5f}", (det['corner_x'], text_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(final_image, f"Alt: {alt:.2f}m", (det['corner_x'], text_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                info_text_accum += f"Drone {i+1}:\n  Lat: {lat:.5f}, Lon: {lon:.5f}, Alt: {alt:.2f} m\n\n"

            cv2.putText(final_image, f"Detected: {len(detections)} drones", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            new_steps['final'] = final_image

            # UPDATE UI SAFELY
            def update_ui():
                try:
                    self.processed_steps = new_steps
                    self.show_step('final')
                    # Also display in Result tab
                    self.display_result_image(final_image)
                    self.info_text.delete(1.0, tk.END)
                    self.info_text.insert(1.0, info_text_accum)
                    self.status_var.set(f"Image processed: {len(detections)} objects detected")
                except Exception as e:
                     print(f"UI update error: {e}")

            self.root.after(0, update_ui)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"Error processing image: {str(e)}"
            print(f"Image processing error:\n{error_details}")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_var.set("Error: Processing failed"))
            # Clear display on error
            self.root.after(0, lambda: self.canvas.delete("all"))
            if hasattr(self, 'result_canvas'):
                self.root.after(0, lambda: self.result_canvas.delete("all"))
    
    def show_step(self, step_key):
        """แสดงขั้นตอน"""
        if step_key in self.processed_steps:
            self.display_image(self.processed_steps[step_key])
    
    def display_image(self, img):
        """แสดงภาพ"""
        try:
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            
            img_height, img_width = img.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img_resized = cv2.resize(img, (new_width, new_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor=tk.CENTER)
            self.canvas.image = img_tk
        except Exception as e:
            print(f"Display error: {e}")
    
    def display_result_image(self, img):
        """Display result image in Result tab"""
        try:
            self.result_canvas.update_idletasks()
            canvas_width = self.result_canvas.winfo_width()
            canvas_height = self.result_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            
            img_height, img_width = img.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img_resized = cv2.resize(img, (new_width, new_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.result_canvas.delete("all")
            self.result_canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor=tk.CENTER)
            self.result_canvas.image = img_tk
        except Exception as e:
            print(f"Result display error: {e}")
    
    def is_drone(self, c_area, cw, ch):
        """ตรวจสอบว่าเป็นโดรนหรือไม่"""
        if c_area < self.min_drone_area or c_area > self.max_drone_area:
            return False
        
        aspect_ratio = cw / ch if ch > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        return True

    def _detect_drones_logic(self, image, debug=None):
        """Logic-based detection following image_process_drone function EXACTLY - matches provided code"""
        if debug is None:
            debug = self.debug_mode
        
        # Convert to grayscale (exactly as provided code)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        
        # Threshold at 40 (exactly as provided code: ret1,th1 = cv2.threshold(img,40,255,cv2.THRESH_BINARY))
        ret1, th1 = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
        
        # Inversion (exactly as provided code: th1_inv = cv2.bitwise_not(th1))
        th1_inv = cv2.bitwise_not(th1)
        
        # Dilation with 3x3 kernel, 3 iterations (exactly as provided code)
        kernel = np.ones((3, 3), np.uint8)
        dilated_inv = cv2.dilate(th1_inv, kernel, iterations=3)
        dilated = dilated_inv
        
        # Find contours (exactly as provided code)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        shape_classifications = []
        mask = np.zeros_like(img)
        
        if contours:
            for c in contours:
                corner_x, corner_y, cw, ch = cv2.boundingRect(c)
                center_x = int(corner_x + (cw / 2))
                center_y = int(corner_y + (ch / 2))
                c_area = cw * ch
                
                # Filter: area > 50 and above 35% threshold (exactly as provided code)
                # y_threshold = int(0.35 * h)
                # if c_area > 50  and corner_y < h - y_threshold:  #must bigger than 50 and higher than 30%
                y_threshold = int(0.35 * h)
                if c_area > 50 and corner_y < h - y_threshold:  # must bigger than 50 and higher than 30%
                    # Skip large objects near bottom (exactly as provided code: if corner_y > 500 and ch > 150: continue)
                    if corner_y > 500 and ch > 150: 
                        continue
                    
                    # Skip time label in top-right corner (exactly as provided code: if corner_x > 1300 and corner_y<90: continue)
                    if corner_x > 1300 and corner_y < 90:  # deals with the time label
                        continue
                    
                    # All objects passing filters are considered drones in logic mode
                    is_drone = True
                    prob = 1.0
                    used_classifier = False
                    
                    shape_classifications.append({
                        'contour': c,
                        'corner_x': corner_x,
                        'corner_y': corner_y,
                        'center_x': center_x,
                        'center_y': center_y,
                        'cw': cw,
                        'ch': ch,
                        'is_drone': is_drone,
                        'prob': prob,
                        'used_classifier': used_classifier
                    })
                    
                    detections.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'corner_x': corner_x,
                        'corner_y': corner_y,
                        'cw': cw,
                        'ch': ch,
                        'area': c_area,
                        'prob': prob
                    })
        
        intermediates = {
            'th1': th1,
            'th1_inv': th1_inv,
            'dilated_inv': dilated_inv,
            'contours': contours,
            'shape_classifications': shape_classifications
        }
        
        return detections, intermediates
    
    def _detect_drones(self, image, debug=None, for_video=False):
        """Shared detection logic - supports both Logic and Model modes"""
        if debug is None:
            debug = self.debug_mode
        
        # Determine which mode to use based on context
        if for_video:
            use_logic = hasattr(self, 'video_detection_mode') and self.video_detection_mode.get() == "logic"
        else:
            use_logic = hasattr(self, 'detection_mode') and self.detection_mode.get() == "logic"
        
        # Use logic-based detection if mode is logic
        if use_logic:
            return self._detect_drones_logic(image, debug)
            
        # Model-based detection with improved logic filtering
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use fixed threshold at 40 like logic mode
        ret, th1 = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
        th1_inv = cv2.bitwise_not(th1)
        kernel = np.ones((3, 3), np.uint8)
        dilated_inv = cv2.dilate(th1_inv, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img_gray.shape
        y_threshold = int(0.35 * h)
        
        detections = []
        shape_classifications = []
        
        # Always use Bird-v-Drone ML (ONLY MODEL)
        for c in contours:
            corner_x, corner_y, cw, ch = cv2.boundingRect(c)
            center_x = int(corner_x + (cw / 2))
            center_y = int(corner_y + (ch / 2))
            c_area = cw * ch
            
            # Apply EXACT same logic filtering as logic mode (hardcoded values)
            if c_area > 50 and corner_y < h - y_threshold:  # must bigger than 50 and higher than 30%
                # Skip large objects near bottom - EXACT hardcoded values from logic mode
                if corner_y > 500 and ch > 150: 
                    continue
                
                # Skip time label - EXACT hardcoded values from logic mode
                if corner_x > 1300 and corner_y < 90:  # deals with the time label
                    continue
                
                # ============================================
                # Bird Shape Detection: Check for fragmented/disconnected components
                # Birds often have multiple disconnected parts (wings, body, etc.)
                # ============================================
                is_likely_bird_shape = False
                try:
                    # Extract ROI for this contour
                    padding = 20
                    roi_x = max(0, corner_x - padding)
                    roi_y = max(0, corner_y - padding)
                    roi_w = min(image.shape[1] - roi_x, cw + 2 * padding)
                    roi_h = min(image.shape[0] - roi_y, ch + 2 * padding)
                    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    if roi.size > 0:
                        # Convert to grayscale and threshold
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                        _, roi_binary = cv2.threshold(roi_gray, 40, 255, cv2.THRESH_BINARY)
                        roi_binary = cv2.bitwise_not(roi_binary)
                        
                        # Find connected components in ROI
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_binary, connectivity=8)
                        
                        # Count significant components (area > 10% of contour area)
                        min_component_area = c_area * 0.1
                        significant_components = 0
                        for i in range(1, num_labels):  # Skip background (label 0)
                            component_area = stats[i, cv2.CC_STAT_AREA]
                            if component_area >= min_component_area:
                                significant_components += 1
                        
                        # Birds typically have 2-4 disconnected components (wings, body, tail)
                        # Drones are usually single connected component
                        if significant_components >= 2:
                            is_likely_bird_shape = True
                            print(f"  Bird shape detected: {significant_components} disconnected components (area={c_area})")
                except Exception as e:
                    print(f"  Error checking bird shape: {e}")
                
                if is_likely_bird_shape:
                    print(f"  -> REJECTED: Fragmented shape (likely bird)")
                    continue
                
                # ============================================
                # Cloud Filtering for ML Mode
                # ============================================
                # Extract shape features to filter out clouds
                try:
                    from image_processing import extract_features
                    features = extract_features(c)
                    if features is not None:
                        # Features: [hu_moments(7), solidity, circularity, aspect_ratio, extent, inertia_ratio]
                        solidity = features[7] if len(features) > 7 else 0.5
                        circularity = features[8] if len(features) > 8 else 0.5
                        aspect_ratio = features[9] if len(features) > 9 else 1.0
                        extent = features[10] if len(features) > 10 else 0.5
                        
                        # Cloud characteristics:
                        # 1. Very large area (clouds are usually much larger than drones)
                        # 2. Low solidity (clouds have irregular shapes)
                        # 3. Low circularity (clouds are not round)
                        # 4. Very wide aspect ratio (clouds spread horizontally)
                        # 5. Low extent (clouds don't fill bounding box well)
                        
                        # Filter clouds based on size and shape
                        is_likely_cloud = False
                        
                        # Very large objects are likely clouds
                        if c_area > 50000:  # Much larger than typical drones
                            is_likely_cloud = True
                        
                        # Large objects with low solidity and low circularity
                        if c_area > 20000 and solidity < 0.6 and circularity < 0.3:
                            is_likely_cloud = True
                        
                        # Very wide objects (clouds spread horizontally)
                        if aspect_ratio > 4.0 and c_area > 10000:
                            is_likely_cloud = True
                        
                        # Large objects with very low extent (irregular shape)
                        if c_area > 15000 and extent < 0.4:
                            is_likely_cloud = True
                        
                        if is_likely_cloud:
                            print(f"Filtered out cloud: area={c_area}, solidity={solidity:.2f}, circularity={circularity:.2f}, aspect={aspect_ratio:.2f}")
                            continue
                    else:
                        # If feature extraction fails, use basic size filtering
                        if c_area > 50000:
                            print(f"Filtered out large object (likely cloud): area={c_area}")
                            continue
                except Exception as e:
                    # If feature extraction fails, use basic size filtering
                    if c_area > 50000:
                        print(f"Filtered out large object (likely cloud): area={c_area}")
                        continue
                
                # Object passed logic filtering, now combine ML + size + shape-code
                is_drone = True  # Default: assume drone if passed logic
                prob = 1.0
                used_classifier = False

                # --------------------------------------------
                # 1) Size-based gating (area / height / width)
                # --------------------------------------------
                size_ok = True
                size_reason = ""
                # Use the same tuned bounds as logic mode for drone size, but be more lenient
                if c_area < self.min_drone_area:
                    size_ok = False
                    size_reason = f"area too small ({c_area} < {self.min_drone_area})"
                elif c_area > self.max_drone_area * 1.5:  # Allow slightly larger objects
                    size_ok = False
                    size_reason = f"area too large ({c_area} > {self.max_drone_area * 1.5})"
                # Very thin or flat objects are unlikely to be drones
                aspect_ratio_bbox = cw / float(ch) if ch > 0 else 1.0
                if aspect_ratio_bbox < 0.1 or aspect_ratio_bbox > 8.0:  # More lenient aspect ratio
                    size_ok = False
                    if not size_reason:
                        size_reason = f"aspect ratio out of range ({aspect_ratio_bbox:.2f})"
                
                if not size_ok:
                    print(f"  Size filter rejected: {size_reason}, area={c_area}, cw={cw}, ch={ch}")

                # --------------------------------------------
                # 2) Drone-Only ML + shape-code fusion
                # --------------------------------------------
                if self.drone_only_classifier and self.drone_only_classifier.is_trained:
                    ml_confidence = 0.0
                    is_drone_ml = True

                    # ML prediction (HOG + Hough) - Binary or One-Class
                    is_drone_ml, prob_ml = self.drone_only_classifier.predict(c, image)
                    ml_confidence = prob_ml
                    used_classifier = True
                    
                    # Check if using binary mode
                    is_binary_mode = getattr(self.drone_only_classifier, 'is_binary', False)
                    mode_str = "Binary" if is_binary_mode else "One-Class"
                    
                    print(f"  Contour at ({corner_x},{corner_y}), area={c_area}: ML Prediction ({mode_str}): is_drone={is_drone_ml}, confidence={ml_confidence:.3f}")
                    
                    # In binary mode, if ML says it's NOT a drone, reject immediately
                    if is_binary_mode and not is_drone_ml:
                        print(f"  -> REJECTED: Binary ML classifier identifies as BIRD (confidence={ml_confidence:.3f})")
                        is_drone = False
                        prob = 0.0
                        shape_classifications.append({
                            'contour': c,
                            'corner_x': corner_x,
                            'corner_y': corner_y,
                            'center_x': center_x,
                            'center_y': center_y,
                            'cw': cw,
                            'ch': ch,
                            'is_drone': False,
                            'prob': 0.0,
                            'used_classifier': True
                        })
                        continue  # Skip this contour - it's a bird

                    # Shape-code classifier (Hu moments) - STRICT FILTER for bird rejection
                    shape_is_drone = True
                    shape_conf = 0.5  # Default neutral confidence if not available
                    shape_code_available = False
                    if self.shape_code_classifier is not None:
                        try:
                            shape_is_drone, shape_conf = self.shape_code_classifier.classify(c, image)
                            shape_code_available = True
                            print(f"  Shape-Code Prediction: is_drone={shape_is_drone}, confidence={shape_conf:.3f}")
                            
                            # STRICT: If shape code says it's a bird (not drone), reject immediately
                            if not shape_is_drone and shape_conf > 0.4:
                                print(f"  -> REJECTED: Shape-code identifies as BIRD (confidence={shape_conf:.3f})")
                                is_drone = False
                                prob = 0.0
                                shape_classifications.append({
                                    'contour': c,
                                    'corner_x': corner_x,
                                    'corner_y': corner_y,
                                    'center_x': center_x,
                                    'center_y': center_y,
                                    'cw': cw,
                                    'ch': ch,
                                    'is_drone': False,
                                    'prob': 0.0,
                                    'used_classifier': True
                                })
                                continue  # Skip this contour - it's a bird
                        except Exception as e:
                            # If shape code fails, fall back to neutral (don't reject)
                            print(f"  Shape-code classification error: {e}, using neutral")
                            shape_is_drone = True
                            shape_conf = 0.5  # Neutral confidence
                    else:
                        # No shape-code classifier - use neutral
                        shape_is_drone = True
                        shape_conf = 0.5
                        print(f"  Shape-code classifier not available")

                    # Final fusion: size + ML + shape-code (STRICT on shape-code)
                    # Shape-code has HIGH priority - if it says bird, reject immediately
                    # (Already handled above with continue statement)
                    
                    base_conf = ml_confidence if used_classifier else 1.0
                    if used_classifier and shape_code_available:
                        # Weight shape-code more heavily (70% shape-code, 30% ML)
                        # This ensures shape-code has strong influence
                        combined_conf = (shape_conf * 0.7) + (base_conf * 0.3)
                    else:
                        combined_conf = base_conf

                    # Threshold: Require both ML and shape-code to agree
                    ml_threshold = 0.3
                    shape_threshold = 0.4  # Shape-code must be confident it's a drone
                    
                    # STRICT fusion logic:
                    # 1. Size must be OK
                    # 2. ML must say drone OR shape-code must strongly say drone
                    # 3. Shape-code must not say bird (already checked above)
                    if not size_ok:
                        is_drone = False
                        prob = combined_conf
                        print(f"  -> REJECTED: Size check failed")
                    elif is_drone_ml and ml_confidence >= ml_threshold:
                        # ML says drone with good confidence
                        if shape_code_available:
                            # Require shape-code to agree (not say bird)
                            if shape_is_drone and shape_conf >= shape_threshold:
                                is_drone = True
                                prob = combined_conf
                                print(f"  -> ACCEPTED: ML ({ml_confidence:.3f}) and Shape-code ({shape_conf:.3f}) both agree - DRONE")
                            elif shape_is_drone:
                                # Shape-code says drone but low confidence - still accept if ML is strong
                                is_drone = True
                                prob = max(combined_conf, ml_confidence * 0.8)
                                print(f"  -> ACCEPTED: ML strong ({ml_confidence:.3f}), Shape-code weak ({shape_conf:.3f}) - DRONE")
                            else:
                                # Shape-code says bird - reject (shouldn't reach here due to early check)
                                is_drone = False
                                prob = 0.0
                                print(f"  -> REJECTED: ML says drone but Shape-code says BIRD")
                        else:
                            # No shape-code - trust ML
                            is_drone = True
                            prob = combined_conf
                            print(f"  -> ACCEPTED: ML confidence {ml_confidence:.3f} >= threshold {ml_threshold} (no shape-code)")
                    elif is_drone_ml and ml_confidence >= 0.2:
                        # ML says drone but low confidence - require strong shape-code agreement
                        if shape_code_available and shape_is_drone and shape_conf >= 0.5:
                            is_drone = True
                            prob = max(combined_conf, 0.4)
                            print(f"  -> ACCEPTED: Low ML ({ml_confidence:.3f}) but strong Shape-code ({shape_conf:.3f}) - DRONE")
                        else:
                            is_drone = False
                            prob = combined_conf
                            print(f"  -> REJECTED: Low ML confidence {ml_confidence:.3f} and weak/no shape-code support")
                    elif shape_code_available and shape_is_drone and shape_conf >= 0.6:
                        # ML doesn't say drone, but shape-code strongly says drone
                        is_drone = True
                        prob = max(shape_conf, 0.5)
                        print(f"  -> ACCEPTED: ML rejected but strong Shape-code ({shape_conf:.3f}) - DRONE")
                    else:
                        # Neither ML nor shape-code strongly support drone
                        is_drone = False
                        prob = combined_conf
                        print(f"  -> REJECTED: ML={is_drone_ml} (conf={ml_confidence:.3f}), Shape-code={shape_is_drone} (conf={shape_conf:.3f})")

                    print(
                        f"  Final decision: size_ok={size_ok}, is_drone_ml={is_drone_ml}, shape_is_drone={shape_is_drone}, "
                        f"ml_conf={ml_confidence:.3f}, shape_conf={shape_conf:.3f}, "
                        f"combined_conf={combined_conf:.3f}, is_drone={is_drone}"
                    )
                else:
                    # Model not trained - fallback to logic-based detection
                    print("Warning: Drone-Only ML classifier not trained. Using logic-based fallback.")
                    # If size is OK and passed all logic filters, consider it a drone
                    if size_ok:
                        is_drone = True
                        prob = 0.7  # Medium confidence for logic-based
                    else:
                        is_drone = False
                        prob = 0.0
                    used_classifier = False
                
                shape_classifications.append({
                    'contour': c,
                    'corner_x': corner_x,
                    'corner_y': corner_y,
                    'center_x': center_x,
                    'center_y': center_y,
                    'cw': cw,
                    'ch': ch,
                    'is_drone': is_drone,
                    'prob': prob,
                    'used_classifier': used_classifier
                })
                
                # Only add if ML says it's a drone (or no ML used)
                if is_drone:
                    detections.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'corner_x': corner_x,
                        'corner_y': corner_y,
                        'cw': cw,
                        'ch': ch,
                        'area': c_area,
                        'prob': prob
                    })
            
        intermediates = {
            'th1': th1,
            'th1_inv': th1_inv,
            'dilated_inv': dilated_inv,
            'contours': contours,
            'shape_classifications': shape_classifications
        }
        
        return detections, intermediates
    
    def calculate_gps(self, center_x, center_y, h, w):
        """คำนวณ GPS"""
        center_img_x = w / 2
        center_img_y = h / 2
        
        pixel_to_degree = 0.00001
        offset_x = (center_x - center_img_x) * pixel_to_degree
        offset_y = (center_img_y - center_y) * pixel_to_degree
        
        # ถ้าอยู่ใน Training Mode และมีข้อมูล GPS จริง ให้ใช้ค่านั้น
        if self.mode_var.get() == "training" and self.gps_data:
            try:
                return float(self.gps_data['Latitude']), float(self.gps_data['Longitude']), float(self.gps_data['Altitude'])
            except (ValueError, KeyError):
                pass

        # ถ้ามี Custom Model ให้ใช้ Model (PyTorch หรือ sklearn)
        if self.using_custom_model:
            # Try PyTorch model first
            if self.pytorch_gps_model and self.pytorch_gps_model.is_trained:
                res = self.pytorch_gps_model.predict(center_x, center_y)
                if res:
                    return res
            # Fallback to sklearn model
            elif self.gps_model.is_trained:
                res = self.gps_model.predict(center_x, center_y)
                if res:
                    return res

        lat = self.base_lat + offset_y
        lon = self.base_lon + offset_x
        y_ratio = center_y / h
        alt = self.base_alt + (1 - y_ratio) * 10
        
        return lat, lon, alt
    
    def load_model(self):
        """โหลดโมเดล"""
        file_path = filedialog.askopenfilename(
            title="Load GPS Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.gps_model.load(file_path):
                self.using_custom_model = True
                # Store path for display
                self.gps_model.model_path = file_path
                status_text = f"Using: {os.path.basename(file_path)}"
                self._update_model_status(status_text)
                
                self._show_success("Success", "GPS Model loaded successfully")
                
                # ถ้าอยู่ใน Image Mode และมีภาพอยู่ ให้ process ใหม่
                if self.mode_var.get() == "image" and self.current_image_path:
                    self.process_image(self.current_image_path)
    
    def save_model(self):
        """บันทึกโมเดล GPS ไปที่ models folder โดยอัตโนมัติ"""
        if not self.gps_model.is_trained:
            messagebox.showwarning("Warning", "No trained model to save")
            return
        
        # Save to models folder automatically
        if PATHS_AVAILABLE:
            paths.ensure_structure()
            file_path = paths.GPS_MODEL_FILE
        else:
            file_path = os.path.join(os.getcwd(), "models", "gps_model.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if self.gps_model.save(file_path):
            messagebox.showinfo("Success", f"GPS Model saved successfully to:\n{file_path}")
        else:
            messagebox.showerror("Error", "Failed to save model")

    def start_training(self):
        """เริ่มเทรนโมเดล GPS"""
        # Check if GPS model already exists
        if PATHS_AVAILABLE and paths.GPS_MODEL_FILE.exists():
            if self.gps_model.is_trained:
                response = messagebox.askyesnocancel(
                    "GPS Model Already Exists",
                    f"GPS Model already exists at:\n{paths.GPS_MODEL_FILE}\n\n"
                    f"Current model is trained.\n\n"
                    f"Would you like to:\n"
                    f"- Yes: Retrain (overwrite existing model)\n"
                    f"- No: Use existing model (skip training)\n"
                    f"- Cancel: Do nothing"
                )
                if response is None:  # Cancel
                    return
                elif response is False:  # No - use existing
                    messagebox.showinfo("Info", "Using existing GPS model. No training needed.")
                    return
                # If Yes, continue with training
        
        # Use selected folder if available, otherwise check default
        if self.gps_train_folder and os.path.exists(self.gps_train_folder):
            train_dir = self.gps_train_folder
        else:
            # Use paths from paths.py if available
            if PATHS_AVAILABLE:
                train_dir = str(paths.GPS_TRAIN_DIR)
            else:
                # Fallback: try new structure first, then old
                train_dir = os.path.join(os.getcwd(), "data", "gps", "P2_DATA_TRAIN")
                if not os.path.exists(train_dir):
                    train_dir = os.path.join(os.getcwd(), "P2_DATA_TRAIN")
            
            if not os.path.exists(train_dir):
                train_dir = filedialog.askdirectory(title="Select Training Data Folder")
                if train_dir:
                    # Save preference
                    if CONFIG_AVAILABLE:
                        set_gps_train_folder(train_dir)
            
        if not train_dir:
            return
        
        # Save preference
        if CONFIG_AVAILABLE and train_dir:
            set_gps_train_folder(train_dir)
            
        self.status_var.set(f"Training started on {os.path.basename(train_dir)}... Please wait")
        self.root.update()
        
        # Run in thread to avoid freezing UI
        threading.Thread(target=self._train_process, args=(train_dir,), daemon=True).start()
        
    def _train_process(self, train_dir):
        """กระบวนการเทรน (Background Thread)"""
        try:
            files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_files = len(files)
            
            if total_files == 0:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No image files found"))
                return
                
            training_data = []
            
            for i, filename in enumerate(files):
                # Update progress bar
                progress = (i / total_files) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                if i % 10 == 0:
                    self.root.after(0, lambda p=i, t=total_files: 
                        self.status_var.set(f"Training: {p}/{t} ({p/t*100:.1f}%)"))
                
                img_path = os.path.join(train_dir, filename)
                csv_path = img_path.replace('.jpg', '.csv')
                
                if not os.path.exists(csv_path):
                    continue
                    
                # Read CSV
                lat, lon, alt = None, None, None
                try:
                    # Try reading with utf-8-sig first
                    with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                lat = float(row.get('Latitude', 0))
                                lon = float(row.get('Longitude', 0))
                                alt = float(row.get('Altitude', 0))
                            except ValueError:
                                continue
                            break
                except Exception:
                    continue
                    
                if lat is None or lon is None:
                    continue
                    
                # Read Image and detect object
                # (Same secure reading logic as process_image)
                try:
                    stream = open(img_path, "rb")
                    bytes = bytearray(stream.read())
                    numpyarray = np.asarray(bytes, dtype=np.uint8)
                    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                    stream.close()
                    
                    if img is None: continue
                    
                    # Detect
                    detections, _ = self._detect_drones(img, debug=False)
                    if detections:
                        # Use the largest detection
                        det = max(detections, key=lambda x: x['area'])
                        training_data.append({
                            'pixel_x': det['center_x'],
                            'pixel_y': det['center_y'],
                            'lat': lat,
                            'lon': lon,
                            'alt': alt
                        })
                except Exception:
                    continue

            # Train GPS Model
            if len(training_data) > 5:
                # Prepare data
                data_points = [(d['pixel_x'], d['pixel_y'], d['lat'], d['lon'], d['alt']) for d in training_data]
                
                # Choose model type
                model_type = self.gps_model_type.get() if hasattr(self, 'gps_model_type') else "sklearn"
                
                if model_type == "pytorch" and self.pytorch_gps_model is not None:
                    # Train PyTorch model
                    result = self.pytorch_gps_model.train(data_points, epochs=100, batch_size=32)
                    if result:
                        # Save to models folder
                        if PATHS_AVAILABLE:
                            paths.ensure_structure()
                            pytorch_model_path = paths.GPS_MODEL_PYTORCH_FILE
                            if self.pytorch_gps_model.save(str(pytorch_model_path)):
                                print(f"[OK] PyTorch GPS Model saved to {pytorch_model_path}")
                            else:
                                print(f"[WARNING] Failed to save PyTorch GPS model")
                        else:
                            self.pytorch_gps_model.save('gps_model_pytorch.pkl')
                        self.using_custom_model = True
                        self.root.after(0, lambda: self._on_training_complete(True, len(training_data)))
                    else:
                        self.root.after(0, lambda: self._on_training_complete(False, "PyTorch training failed"))
                else:
                    # Train sklearn model - can use either format
                    result = self.gps_model.train(data_points=data_points)
                    
                    if result:
                        self.using_custom_model = True
                        # Auto-save GPS model to models folder
                        if PATHS_AVAILABLE:
                            paths.ensure_structure()
                            gps_model_path = paths.GPS_MODEL_FILE
                            if self.gps_model.save(gps_model_path):
                                print(f"[OK] GPS Model auto-saved to {gps_model_path}")
                            else:
                                print(f"[WARNING] Failed to auto-save GPS model to {gps_model_path}")
                        self.root.after(0, lambda: self._on_training_complete(True, len(training_data)))
                    else:
                        self.root.after(0, lambda: self._on_training_complete(False, "Training failed"))
            else:
                self.root.after(0, lambda: self._on_training_complete(False, "Not enough valid training data"))
                
        except Exception as e:
            self.root.after(0, lambda: self._on_training_complete(False, str(e)))
            
    
    def _on_training_complete(self, success, message):
        """Callback when training finishes"""
        self.progress_var.set(100)
        self.status_var.set("GPS Training finished")
        if success:
            messagebox.showinfo("Success", f"GPS Model trained successfully with {message} samples.")
            self.save_model_btn.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", f"Training failed: {message}")

def main():
    root = tk.Tk()
    # Setup styles
    style = ttk.Style()
    style.theme_use('clam')
    
    app = DroneDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
