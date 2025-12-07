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
    from bird_v_drone_ml import get_bird_v_drone_classifier
    BIRD_V_DRONE_AVAILABLE = True
except ImportError as e:
    print(f"Bird-v-Drone ML module not available: {e}")
    BIRD_V_DRONE_AVAILABLE = False


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
        self.selected_classifier_model = tk.StringVar(value="Bird-v-Drone ML (HOG+Hough+Shape)")  # For ML mode
        self.video_selected_classifier_model = tk.StringVar(value="Bird-v-Drone ML (HOG+Hough+Shape)")  # For ML mode
        
        # Model paths for tracking latest
        self.model_paths = {}  # Store model paths by type
        self.latest_model_path = None
        
        # Drone Tracker
        self.tracker = DroneTracker(max_drones=2, max_lost_frames=30, distance_threshold=500)
        
        # Video Processor
        self.video_processor = None
        
        # GPS Model
        self.gps_model = GPSModel()
        self.gps_data = None
        self.using_custom_model = False
        
        # GPS Base
        self.base_lat = 14.30492
        self.base_lon = 101.17255
        self.base_alt = 46.91
        
        # Bird-v-Drone Enhanced ML Classifier (HOG + Hough + Shape) - ONLY MODEL
        # Will be initialized after UI is created to use the selected device
        self.bird_v_drone_classifier = None
        self.bvd_use_gpu = None
        
        # Old models removed - using only Bird-v-Drone ML
        self.classifier = None  # Old Random Forest - removed
        self.pytorch_classifier = None  # Old PyTorch - removed
        self.pytorch_gps_model = None
        self.shape_code_classifier = None  # Old Shape Code - removed
        self.pytorch_available = False

        # สร้าง UI - creating widgets sets up status_var and canvas
        self.create_widgets()
        
        # Initialize Bird-v-Drone classifier after UI is created (to use selected device)
        if BIRD_V_DRONE_AVAILABLE:
            try:
                # Use the device selection from UI (radio button is created in create_widgets)
                use_gpu = False
                if hasattr(self, 'bvd_use_gpu') and self.bvd_use_gpu is not None:
                    use_gpu = self.bvd_use_gpu.get()
                
                # Initialize classifier - if GPU requested but not available, it will raise error
                # For initialization, we allow fallback to CPU to load existing models
                try:
                    self.bird_v_drone_classifier = get_bird_v_drone_classifier(use_gpu=use_gpu)
                except RuntimeError as e:
                    # GPU requested but not available - fallback to CPU for loading existing model only
                    if use_gpu:
                        print(f"Warning: GPU requested but not available during initialization: {e}")
                        print("Falling back to CPU mode for loading existing model...")
                        use_gpu = False
                        if hasattr(self, 'bvd_use_gpu') and self.bvd_use_gpu is not None:
                            self.bvd_use_gpu.set(False)  # Update UI to reflect CPU mode
                        self.bird_v_drone_classifier = get_bird_v_drone_classifier(use_gpu=False)
                    else:
                        raise
                
                if self.bird_v_drone_classifier.load():
                    device_type = "GPU" if (hasattr(self.bird_v_drone_classifier, 'pytorch_model') and self.bird_v_drone_classifier.pytorch_model is not None) else "CPU"
                    print(f"Loaded Bird-v-Drone Enhanced ML Classifier (HOG + Hough + Shape) on {device_type}")
            except Exception as e:
                print(f"Error loading Bird-v-Drone classifier: {e}")
                self.bird_v_drone_classifier = None
        
        # Post-UI Initialization - Bird-v-Drone ML only
        if self.bird_v_drone_classifier and not self.bird_v_drone_classifier.is_trained:
            print("Bird-v-Drone ML Classifier not found. Please train the model.")
            drone_dir = os.path.join(os.getcwd(), "Drones")
            bird_dir = os.path.join(os.getcwd(), "Birds")
            if os.path.exists(drone_dir) and os.path.exists(bird_dir):
                # Delay slightly to ensure UI is fully rendered and mainloop is active
                self.root.after(1000, lambda: self.auto_train_bird_v_drone(drone_dir, bird_dir))
            else:
                 print("Training folders (Birds/Drones) not found. Please train the model manually.")
        
        # Initialize model status
        self.root.after(500, lambda: [self.update_model_list(), self.update_video_model_list()])
    
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
        
        ttk.Radiobutton(mode_selection_frame, text="ML-Based (Bird-v-Drone ML)", 
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
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            steps_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        steps_canvas.bind_all("<MouseWheel>", lambda e: on_mousewheel(e) if steps_canvas.winfo_containing(e.x_root, e.y_root) else None)
        
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
        
        self.shape_model_status_img = tk.StringVar(value="Bird-v-Drone ML Classifier")
        if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
             self.shape_model_status_img.set("Bird-v-Drone ML: Ready ✓")
        else:
             self.shape_model_status_img.set("Bird-v-Drone ML: NOT Trained")
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
        
        ttk.Radiobutton(video_mode_selection_frame, text="ML-Based (Bird-v-Drone ML)", 
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
        
        # Mouse wheel scrolling
        def on_training_mousewheel(event):
            training_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        training_canvas.bind_all("<MouseWheel>", lambda e: on_training_mousewheel(e) if training_canvas.winfo_containing(e.x_root, e.y_root) else None)
        
        training_scrollbar.pack(side="right", fill="y")
        training_canvas.pack(side="left", fill="both", expand=True)
        
        # 1. GPS Model Training
        gps_train_frame = ttk.LabelFrame(self.training_panel, text="1. GPS Localization Training", padding="10")
        gps_train_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(gps_train_frame, text="Train a model to predict GPS from pixel coordinates.").pack(anchor=tk.W, pady=(0,5))
        
        self.gps_folder_label = ttk.Label(gps_train_frame, text="Folder: Default (P2_DATA_TRAIN)", wraplength=250)
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
        
        # Bird-v-Drone Enhanced ML Training (HOG + Hough + Shape) - ONLY MODEL
        bird_v_drone_frame = ttk.LabelFrame(self.training_panel, text="Bird-v-Drone Enhanced ML Training (HOG+Hough+Shape)", padding="10")
        bird_v_drone_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(bird_v_drone_frame, text="Train enhanced classifier with HOG, Hough Transform, and Shape features:").pack(anchor=tk.W, pady=(0,5))
        ttk.Label(bird_v_drone_frame, text="Based on Bird-v-Drone repository approach", foreground="gray", font=("TkDefaultFont", 8)).pack(anchor=tk.W, pady=(0,5))
        
        if not BIRD_V_DRONE_AVAILABLE:
            ttk.Label(bird_v_drone_frame, text="⚠ Warning: scikit-image not installed. Please run: pip install scikit-image", 
                     foreground="red", font=("TkDefaultFont", 9)).pack(anchor=tk.W, pady=(0,5))
        
        # Folder selection
        bvd_folder_frame = ttk.Frame(bird_v_drone_frame)
        bvd_folder_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(bvd_folder_frame, text="Drones Folder:").pack(anchor=tk.W)
        bvd_drone_frame = ttk.Frame(bvd_folder_frame)
        bvd_drone_frame.pack(fill=tk.X, pady=2)
        self.bvd_drone_folder = tk.StringVar(value=os.path.join(os.getcwd(), "Drones"))
        ttk.Entry(bvd_drone_frame, textvariable=self.bvd_drone_folder, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(bvd_drone_frame, text="Browse", command=lambda: self.select_folder(self.bvd_drone_folder, "Select Drones Folder")).pack(side=tk.LEFT)
        
        ttk.Label(bvd_folder_frame, text="Birds Folder:").pack(anchor=tk.W, pady=(5,0))
        bvd_bird_frame = ttk.Frame(bvd_folder_frame)
        bvd_bird_frame.pack(fill=tk.X, pady=2)
        self.bvd_bird_folder = tk.StringVar(value=os.path.join(os.getcwd(), "Birds"))
        ttk.Entry(bvd_bird_frame, textvariable=self.bvd_bird_folder, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(bvd_bird_frame, text="Browse", command=lambda: self.select_folder(self.bvd_bird_folder, "Select Birds Folder")).pack(side=tk.LEFT)
        
        # Device selection (CPU/GPU)
        device_frame = ttk.Frame(bird_v_drone_frame)
        device_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Label(device_frame, text="Training Device:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Check GPU availability
        gpu_available = False
        gpu_name = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
        except:
            pass
        
        self.bvd_use_gpu = tk.BooleanVar()
        
        # Set default based on GPU availability
        if gpu_available:
            self.bvd_use_gpu.set(True)  # Default to GPU if available
            device_info = f"GPU: {gpu_name}"
        else:
            self.bvd_use_gpu.set(False)  # Default to CPU if no GPU
            device_info = "GPU: Not available"
        
        device_selection_frame = ttk.Frame(device_frame)
        device_selection_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Radiobutton(device_selection_frame, text="CPU (RandomForest)", 
                       variable=self.bvd_use_gpu, value=False).pack(side=tk.LEFT, padx=(0, 15))
        
        if gpu_available:
            ttk.Radiobutton(device_selection_frame, text=f"GPU (PyTorch) - {gpu_name}", 
                           variable=self.bvd_use_gpu, value=True).pack(side=tk.LEFT)
        else:
            ttk.Radiobutton(device_selection_frame, text="GPU (PyTorch) - Not available", 
                           variable=self.bvd_use_gpu, value=True, state=tk.DISABLED).pack(side=tk.LEFT)
            ttk.Label(device_selection_frame, text="(PyTorch not installed or no GPU)", 
                     foreground="gray", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(bird_v_drone_frame, text="Train Bird-v-Drone Enhanced ML Classifier", command=self.train_bird_v_drone_classifier_custom).pack(fill=tk.X, pady=5)
        
        # Status display - Bird-v-Drone ML only
        self.shape_code_status = tk.StringVar(value="Classifier Checked")
        if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
            status_text = "Status: Bird-v-Drone ML Ready ✓"
        else:
            status_text = "Status: Bird-v-Drone ML NOT Trained - Please train the model"
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
        
        # Status display - Bird-v-Drone ML only
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
        
    def auto_train_bird_v_drone(self, drone_dir, bird_dir):
        """Auto-train Bird-v-Drone ML classifier on startup if not trained"""
        if not BIRD_V_DRONE_AVAILABLE or self.bird_v_drone_classifier is None:
            return
        
        if self.bird_v_drone_classifier.is_trained:
            return  # Already trained
        
        def _train():
            try:
                self.status_var.set("Auto-training Bird-v-Drone ML Classifier... This may take a while.")
                self.root.update()
                
                success = self.bird_v_drone_classifier.train(
                    drone_dir, 
                    bird_dir,
                    use_hog=True,
                    use_hough=True,
                    use_shape=True,
                    apply_noise_filter=True
                )
                
                if success:
                    self.bird_v_drone_classifier.save()
                    self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: self.update_video_model_list())
                    self.root.after(0, lambda: self.shape_code_status.set("Status: Bird-v-Drone ML Ready ✓"))
                else:
                    self.root.after(0, lambda: self.shape_code_status.set("Status: Auto-train failed - Please train manually"))
            except Exception as e:
                print(f"Auto-train error: {e}")
                self.root.after(0, lambda: self.shape_code_status.set("Status: Auto-train failed - Please train manually"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        training_thread = threading.Thread(target=_train, daemon=True)
        training_thread.start()
    
    def train_classifier_thread(self, drone_dir, bird_dir):
        """Legacy function - redirects to Bird-v-Drone ML"""
        self.auto_train_bird_v_drone(drone_dir, bird_dir)

    def select_folder(self, folder_var, title):
        """Helper function to select a folder"""
        folder = filedialog.askdirectory(title=title)
        if folder:
            folder_var.set(folder)
    
    def train_random_forest_custom(self):
        """Train Random Forest classifier with custom folders"""
        drone_dir = self.rf_drone_folder.get()
        bird_dir = self.rf_bird_folder.get()
        
        if not os.path.exists(drone_dir):
            messagebox.showerror("Error", f"Drones folder not found: {drone_dir}")
            return
        if not os.path.exists(bird_dir):
            messagebox.showerror("Error", f"Birds folder not found: {bird_dir}")
            return
        
        self.train_classifier_thread(drone_dir, bird_dir)
    
    def re_train_classifier(self):
        """Manual trigger for training Random Forest classifier (legacy - uses default folders)"""
        drone_dir = os.path.join(os.getcwd(), "Drones")
        bird_dir = os.path.join(os.getcwd(), "Birds")
        if not os.path.exists(drone_dir) or not os.path.exists(bird_dir):
            messagebox.showerror("Error", "Birds/Drones folders not found in current directory.")
            return
        self.train_classifier_thread(drone_dir, bird_dir)
    
    def train_pytorch_classifier_custom(self):
        """Train PyTorch CNN classifier with custom folders"""
        if self.pytorch_classifier is None:
            messagebox.showerror("Error", "PyTorch not available. Please install torch and torchvision.")
            return
        
        drone_dir = self.pt_drone_folder.get()
        bird_dir = self.pt_bird_folder.get()
        
        if not os.path.exists(drone_dir):
            messagebox.showerror("Error", f"Drones folder not found: {drone_dir}")
            return
        if not os.path.exists(bird_dir):
            messagebox.showerror("Error", f"Birds folder not found: {bird_dir}")
            return
        
        def _train():
            try:
                self.status_var.set("Training PyTorch CNN Classifier... This may take a while.")
                self.root.update()
                
                success = self.pytorch_classifier.train(drone_dir, bird_dir, epochs=50, batch_size=32)
                
                if success:
                    self.pytorch_classifier.save()
                    self.root.after(0, lambda: messagebox.showinfo("Success", "PyTorch CNN Classifier trained successfully!"))
                    self.root.after(0, lambda: self.shape_code_status.set("Status: PyTorch Ready"))
                    # Update model list and auto-select latest
                    if hasattr(self, 'detection_mode') and self.detection_mode.get() == "model":
                        self.root.after(0, lambda: self.update_model_list())
                        self.root.after(0, lambda: self.selected_classifier_model.set("PyTorch CNN"))
                    if hasattr(self, 'video_detection_mode') and self.video_detection_mode.get() == "model":
                        self.root.after(0, lambda: self.update_video_model_list())
                        self.root.after(0, lambda: self.video_selected_classifier_model.set("PyTorch CNN"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Training failed. Check console."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training error: {e}"))
        
        threading.Thread(target=_train, daemon=True).start()
    
    def train_bird_v_drone_classifier_custom(self):
        """Train Bird-v-Drone Enhanced ML classifier with custom folders"""
        if not BIRD_V_DRONE_AVAILABLE or self.bird_v_drone_classifier is None:
            messagebox.showerror("Error", "Bird-v-Drone ML module not available. Please install scikit-image.")
            return
        
        drone_dir = self.bvd_drone_folder.get()
        bird_dir = self.bvd_bird_folder.get()
        
        if not os.path.exists(drone_dir):
            messagebox.showerror("Error", f"Drones folder not found: {drone_dir}")
            return
        if not os.path.exists(bird_dir):
            messagebox.showerror("Error", f"Birds folder not found: {bird_dir}")
            return
        
        # Get device selection from UI
        use_gpu = self.bvd_use_gpu.get() if hasattr(self, 'bvd_use_gpu') and self.bvd_use_gpu is not None else False
        
        # Verify GPU availability if GPU is selected - show error instead of fallback
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    # Check if PyTorch is installed but CUDA is not available
                    cuda_available = torch.cuda.is_available()
                    if not cuda_available:
                        error_msg = (
                            "GPU mode selected but CUDA is not available!\n\n"
                            "Possible reasons:\n"
                            "1. No NVIDIA GPU detected\n"
                            "2. CUDA drivers not installed\n"
                            "3. PyTorch not compiled with CUDA support\n\n"
                            "Please:\n"
                            "- Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
                            "- Or select CPU mode instead"
                        )
                        messagebox.showerror("GPU Not Available", error_msg)
                        return
            except ImportError:
                error_msg = (
                    "GPU mode selected but PyTorch is not installed!\n\n"
                    "Please install PyTorch with CUDA support:\n"
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n\n"
                    "Or select CPU mode instead"
                )
                messagebox.showerror("PyTorch Not Installed", error_msg)
                return
        
        # Reinitialize classifier with selected device
        device_name = "GPU (PyTorch)" if use_gpu else "CPU (RandomForest)"
        print(f"Initializing classifier for training on {device_name}...")
        
        try:
            self.bird_v_drone_classifier = get_bird_v_drone_classifier(use_gpu=use_gpu)
        except RuntimeError as e:
            error_msg = str(e)
            messagebox.showerror("GPU Initialization Error", error_msg)
            return
        except Exception as e:
            error_msg = f"Error initializing classifier: {str(e)}"
            messagebox.showerror("Initialization Error", error_msg)
            return
        
        def _train():
            try:
                # Reset progress
                device_text = "GPU (PyTorch)" if use_gpu else "CPU (RandomForest)"
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Initializing..."))
                self.root.after(0, lambda: self.status_var.set(f"Training Bird-v-Drone Enhanced ML Classifier on {device_text}... This may take a while."))
                self.root.update()
                
                # Count total images for progress calculation
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
                
                if total_images == 0:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No images found in the specified folders."))
                    return
                
                # Update progress - estimate: 80% for processing, 20% for training
                self.root.after(0, lambda: self.progress_label.config(text=f"Found {total_images} images. Starting processing..."))
                self.root.after(0, lambda: self.progress_var.set(5))
                self.root.update()
                
                # Define progress callback for training
                def update_progress(current, total, message):
                    """Update progress bar from training thread"""
                    try:
                        if total > 0:
                            # 5-80% for processing images, 80-90% for training, 90-100% for saving
                            if "Training classifier" in message:
                                progress_pct = 80
                            elif "Saving model" in message:
                                progress_pct = 90
                            elif "Training Complete" in message:
                                progress_pct = 100
                            else:
                                # Processing images: 5-80%
                                progress_pct = 5 + int((current / total) * 75)
                            
                            # Update UI in main thread
                            self.root.after(0, lambda p=progress_pct: self.progress_var.set(p))
                            self.root.after(0, lambda m=message: self.progress_label.config(text=m))
                    except Exception as e:
                        print(f"Progress update error: {e}")
                
                # Train (this will process images internally)
                success = self.bird_v_drone_classifier.train(
                    drone_dir, 
                    bird_dir,
                    use_hog=True,
                    use_hough=True,
                    use_shape=True,
                    apply_noise_filter=True,
                    progress_callback=update_progress
                )
                
                self.root.after(0, lambda: self.progress_var.set(95))
                self.root.after(0, lambda: self.progress_label.config(text="Finalizing..."))
                
                if success:
                    self.bird_v_drone_classifier.save()
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.progress_label.config(text="Training Complete! ✓"))
                    device_used = "GPU (PyTorch)" if (use_gpu and hasattr(self.bird_v_drone_classifier, 'pytorch_model') and self.bird_v_drone_classifier.pytorch_model is not None) else "CPU (RandomForest)"
                    self.root.after(0, lambda d=device_used: messagebox.showinfo("Success", 
                        f"Bird-v-Drone Enhanced ML Classifier trained successfully!\n"
                        f"Device: {d}\n"
                        f"Features: HOG + Hough Transform + Shape"))
                    # Update model list and auto-select latest
                    self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: self.update_video_model_list())
                    if self.bird_v_drone_classifier.is_trained:
                        self.root.after(0, lambda: self.selected_classifier_model.set("Bird-v-Drone ML (HOG+Hough+Shape)"))
                        self.root.after(0, lambda: self.video_selected_classifier_model.set("Bird-v-Drone ML (HOG+Hough+Shape)"))
                else:
                    self.root.after(0, lambda: self.progress_var.set(0))
                    self.root.after(0, lambda: self.progress_label.config(text="Training Failed"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Bird-v-Drone Enhanced ML Classifier training failed."))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Error occurred"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Training error: {msg}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        training_thread = threading.Thread(target=_train, daemon=True)
        training_thread.start()
    
    def train_pytorch_classifier(self):
        """Train PyTorch CNN classifier (legacy - uses default folders)"""
        if self.pytorch_classifier is None:
            messagebox.showerror("Error", "PyTorch not available. Please install torch and torchvision.")
            return
        
        drone_dir = os.path.join(os.getcwd(), "Drones")
        bird_dir = os.path.join(os.getcwd(), "Birds")
        if not os.path.exists(drone_dir) or not os.path.exists(bird_dir):
            messagebox.showerror("Error", "Birds/Drones folders not found in current directory.")
            return
        
        def _train():
            try:
                self.status_var.set("Training PyTorch CNN Classifier... This may take a while.")
                self.root.update()
                
                success = self.pytorch_classifier.train(drone_dir, bird_dir, epochs=50, batch_size=32)
                
                if success:
                    self.pytorch_classifier.save()
                    self.root.after(0, lambda: messagebox.showinfo("Success", "PyTorch CNN Classifier trained successfully!"))
                    self.root.after(0, lambda: self.shape_code_status.set("Status: PyTorch Ready"))
                    # Update model list and auto-select latest
                    if hasattr(self, 'detection_mode') and self.detection_mode.get() == "model":
                        self.root.after(0, lambda: self.update_model_list())
                        self.root.after(0, lambda: self.selected_classifier_model.set("PyTorch CNN"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Training failed. Check console."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training error: {e}"))
        
        threading.Thread(target=_train, daemon=True).start()
    
    def train_shape_codes_custom(self):
        """Train shape code generator with OCR using custom folders"""
        if self.shape_code_classifier is None:
            messagebox.showerror("Error", "Shape code generator not available.")
            return
        
        drone_dir = self.sc_drone_folder.get()
        bird_dir = self.sc_bird_folder.get()
        
        if not os.path.exists(drone_dir):
            messagebox.showerror("Error", f"Drones folder not found: {drone_dir}")
            return
        if not os.path.exists(bird_dir):
            messagebox.showerror("Error", f"Birds folder not found: {bird_dir}")
            return
        
        def _train():
            try:
                self.status_var.set("Generating Shape Codes with OCR... This may take a while.")
                self.root.update()
                
                success = self.shape_code_classifier.train(drone_dir, bird_dir)
                
                if success:
                    status_text = self.shape_code_status.get()
                    if "Shape Codes Ready" not in status_text:
                        self.shape_code_status.set(status_text + ", Shape Codes Ready")
                    # Update model list and auto-select latest
                    if hasattr(self, 'detection_mode') and self.detection_mode.get() == "model":
                        self.root.after(0, lambda: self.update_model_list())
                    if hasattr(self, 'video_detection_mode') and self.video_detection_mode.get() == "model":
                        self.root.after(0, lambda: self.update_video_model_list())
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Shape Codes generated successfully!"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Shape code generation failed."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {e}"))
        
        threading.Thread(target=_train, daemon=True).start()
    
    def train_shape_codes(self):
        """Train shape code generator with OCR (legacy - uses default folders)"""
        if self.shape_code_classifier is None:
            messagebox.showerror("Error", "Shape code generator not available.")
            return
        
        drone_dir = os.path.join(os.getcwd(), "Drones")
        bird_dir = os.path.join(os.getcwd(), "Birds")
        if not os.path.exists(drone_dir) or not os.path.exists(bird_dir):
            messagebox.showerror("Error", "Birds/Drones folders not found in current directory.")
            return
        
        def _train():
            try:
                self.status_var.set("Generating Shape Codes with OCR... This may take a while.")
                self.root.update()
                
                success = self.shape_code_classifier.train(drone_dir, bird_dir)
                
                if success:
                    status_text = self.shape_code_status.get()
                    if "Shape Codes Ready" not in status_text:
                        self.shape_code_status.set(status_text + ", Shape Codes Ready")
                    # Update model list and auto-select latest
                    if hasattr(self, 'detection_mode') and self.detection_mode.get() == "model":
                        self.root.after(0, lambda: self.update_model_list())
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Shape Codes generated successfully!"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Shape code generation failed."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {e}"))
        
        threading.Thread(target=_train, daemon=True).start()
        
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
                text="Enhanced ML Classifier with HOG, Hough Transform, and Shape features"
            )
            # Update model status
            if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
                self.model_status_label.config(text="Status: Bird-v-Drone ML Ready ✓", foreground="green")
            else:
                self.model_status_label.config(text="Status: Bird-v-Drone ML NOT Trained - Please train the model", foreground="red")
    
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
                text="Enhanced ML Classifier with HOG, Hough Transform, and Shape features"
            )
            # Update model status
            if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
                self.video_model_status_label.config(text="Status: Bird-v-Drone ML Ready ✓", foreground="green")
            else:
                self.video_model_status_label.config(text="Status: Bird-v-Drone ML NOT Trained - Please train the model", foreground="red")
    
    def on_video_detection_mode_change(self):
        """Handle video detection mode change - supports both Logic and ML modes"""
        # Update UI based on selected mode
        self._on_video_detection_mode_change()
        # Update model list if in ML mode
        if self.video_detection_mode.get() == "model":
            self.update_video_model_list()
    
    def update_video_model_list(self):
        """Update video model status - Bird-v-Drone ML only"""
        # Update status label
        if hasattr(self, 'video_model_status_label'):
            if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
                self.video_model_status_label.config(text="Status: Ready ✓", foreground="green")
            else:
                self.video_model_status_label.config(text="Status: NOT Trained - Please train the model", foreground="red")
        
        # Force Bird-v-Drone ML
        self.video_selected_classifier_model.set("Bird-v-Drone ML (HOG+Hough+Shape)")
    
    def on_video_model_selected(self, event=None):
        """Handle video model selection"""
        pass  # Can be used for any action needed when model is selected
    
    def update_model_list(self):
        """Update model status - Bird-v-Drone ML only"""
        # Update status label
        if hasattr(self, 'model_status_label'):
            if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
                self.model_status_label.config(text="Status: Ready ✓", foreground="green")
            else:
                self.model_status_label.config(text="Status: NOT Trained - Please train the model", foreground="red")
        
        # Force Bird-v-Drone ML
        self.selected_classifier_model.set("Bird-v-Drone ML (HOG+Hough+Shape)")
        
        # Update shape model status in image panel
        if hasattr(self, 'shape_model_status_img'):
            if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
                self.shape_model_status_img.set("Bird-v-Drone ML: Ready ✓")
            else:
                self.shape_model_status_img.set("Bird-v-Drone ML: NOT Trained")
    
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
            
            # Shape Classification visualization (แสดงทุก contour ที่ผ่าน filter พร้อม classification result)
            step6 = image.copy()
            shape_classifications = intermediates.get('shape_classifications', [])
            drone_count = 0
            bird_count = 0
            
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
                        
                        if is_drone:
                            drone_count += 1
                            color, label = (0, 255, 0), f"DRONE {prob:.2f}"
                            print(f"Drawing DRONE: prob={prob:.3f}, is_drone={is_drone}")
                        else:
                            bird_count += 1
                            # prob is drone_probability, so bird_prob = 1 - prob
                            bird_prob = 1.0 - prob
                            color, label = (0, 165, 255), f"BIRD {bird_prob:.2f}"
                            print(f"Drawing BIRD: drone_prob={prob:.3f}, bird_prob={bird_prob:.3f}, is_drone={is_drone}")
                        
                        cv2.drawContours(step6, [c], -1, color, 2)
                        cv2.rectangle(step6, (corner_x, corner_y), (corner_x + cw, corner_y + ch), color, 2)
                        cv2.putText(step6, label, (corner_x, max(20, corner_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"Error drawing shape classification: {e}")
                        continue
            except Exception as e:
                print(f"Error in shape classification visualization: {e}")

            # แสดงผลลัพธ์ - ถ้าใช้ classifier จะแสดงทั้ง drone และ bird, ถ้าไม่ใช้จะแสดงแค่ drone
            # Check if using logic mode (no classifier) or ML mode (with classifier)
            using_logic_mode = not any(s.get('used_classifier', False) for s in shape_classifications)
            if using_logic_mode:
                # Logic mode: all detected objects are drones
                summary_text = f"Detected: {drone_count} DRONE(s) (Logic-Based Detection)"
            else:
                # ML mode: shows both drones and birds
                summary_text = f"Detected: {drone_count} DRONE(s), {bird_count} BIRD(s)"
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
                
                cv2.putText(final_image, f"ID:{i}", (det['corner_x'], det['corner_y'] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
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
        # Use same threshold as logic mode for consistency
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
                
                # Object passed logic filtering, now check with ML if model is selected
                is_drone = True  # Default: assume drone if passed logic
                prob = 1.0
                used_classifier = False
                
                # Use Bird-v-Drone ML classifier (ONLY MODEL)
                ml_confidence = 0.0
                if self.bird_v_drone_classifier and self.bird_v_drone_classifier.is_trained:
                    is_drone_ml, prob_ml = self.bird_v_drone_classifier.predict(c, image)
                    ml_confidence = prob_ml
                    used_classifier = True
                    
                    # Debug output
                    print(f"ML Prediction: is_drone={is_drone_ml}, confidence={ml_confidence:.3f}")
                    
                    # Use ML result with adjustable threshold (lowered for better detection)
                    ml_threshold = 0.5  # Lower threshold - was 0.75, too strict
                    is_drone = is_drone_ml and (ml_confidence >= ml_threshold)
                    
                    # If model says it's a drone but confidence is low, still consider it
                    # (model might be conservative)
                    if is_drone_ml and ml_confidence >= 0.3:
                        is_drone = True
                    
                    prob = ml_confidence
                    print(f"Final decision: is_drone={is_drone}, threshold={ml_threshold}, confidence={ml_confidence:.3f}")
                else:
                    # Model not trained - show warning and skip
                    print("Warning: Bird-v-Drone ML classifier not trained. Please train the model first.")
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
        """บันทึกโมเดล"""
        if not self.gps_model.is_trained:
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save GPS Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.gps_model.save(file_path):
                messagebox.showinfo("Success", "Model saved successfully")
            else:
                messagebox.showerror("Error", "Failed to save model")

    def start_training(self):
        """เริ่มเทรนโมเดล"""
        # Use selected folder if available, otherwise check default
        if self.gps_train_folder and os.path.exists(self.gps_train_folder):
            train_dir = self.gps_train_folder
        else:
            train_dir = os.path.join(os.getcwd(), "P2_DATA_TRAIN")
            if not os.path.exists(train_dir):
                train_dir = filedialog.askdirectory(title="Select Training Data Folder")
            
        if not train_dir:
            return
            
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
