"""
Drone/Bird Classifier using PyTorch CNN
ใช้ Convolutional Neural Network สำหรับ classification
"""

import os
import cv2
import numpy as np
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available: {e}")


if TORCH_AVAILABLE:
    class DroneBirdDataset(Dataset):
        """Dataset สำหรับ Drone/Bird classification"""
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            # Load image
            img_path = self.image_paths[idx]
            try:
                # Handle unicode paths
                stream = open(img_path, "rb")
                bytes = bytearray(stream.read())
                numpyarray = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                stream.close()
                
                if img is None:
                    # Return black image if failed
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                if self.transform:
                    img = self.transform(img)
                
                label = torch.LongTensor([self.labels[idx]])[0]
                return img, label
            except Exception as e:
                # Return black image and label 0 on error
                img = Image.new('RGB', (224, 224), (0, 0, 0))
                if self.transform:
                    img = self.transform(img)
                label = torch.LongTensor([0])[0]
                return img, label
else:
    # Dummy class if PyTorch not available
    class DroneBirdDataset:
        pass


if TORCH_AVAILABLE:
    class DroneBirdCNN(nn.Module):
        """CNN สำหรับ Drone/Bird classification"""
        def __init__(self, num_classes=2):
            super(DroneBirdCNN, self).__init__()
            
            # Feature extractor
            self.features = nn.Sequential(
                # First block
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                # Second block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                # Third block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                # Fourth block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 14 * 14, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
else:
    # Dummy class if PyTorch not available
    class DroneBirdCNN:
        pass


class DroneClassifierPyTorch:
    def __init__(self, model_path='drone_classifier_pytorch.pkl'):
        if not TORCH_AVAILABLE:
            self.is_trained = False
            self.model = None
            self.device = None
            self.model_path = model_path
            self.training_stats = {}
            self.train_transform = None
            self.val_transform = None
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DroneBirdCNN(num_classes=2).to(self.device)
        self.model_path = model_path
        self.is_trained = False
        self.training_stats = {}
        
        # Image transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, drone_dir, bird_dir, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the classifier
        
        Args:
            drone_dir: Directory containing drone images
            bird_dir: Directory containing bird images
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
        """
        if not TORCH_AVAILABLE:
            print("Error: PyTorch is not available. Cannot train.")
            return False
        # Collect image paths
        drone_images = []
        bird_images = []
        
        for fname in os.listdir(drone_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                drone_images.append(os.path.join(drone_dir, fname))
        
        for fname in os.listdir(bird_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                bird_images.append(os.path.join(bird_dir, fname))
        
        if len(drone_images) == 0 or len(bird_images) == 0:
            print(f"Error: Need images in both directories. Drones: {len(drone_images)}, Birds: {len(bird_images)}")
            return False
        
        # Create labels: 0 = Bird, 1 = Drone
        all_images = bird_images + drone_images
        all_labels = [0] * len(bird_images) + [1] * len(drone_images)
        
        # Shuffle
        indices = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Split train/val
        n_val = int(len(all_images) * validation_split)
        if n_val < 2:
            n_val = 0
        
        if n_val > 0:
            train_images = all_images[:-n_val]
            train_labels = all_labels[:-n_val]
            val_images = all_images[-n_val:]
            val_labels = all_labels[-n_val:]
        else:
            train_images = all_images
            train_labels = all_labels
            val_images = all_images
            val_labels = all_labels
        
        # Create datasets
        train_dataset = DroneBirdDataset(train_images, train_labels, transform=self.train_transform)
        val_dataset = DroneBirdDataset(val_images, val_labels, transform=self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        print(f"Training PyTorch Drone/Bird Classifier on {self.device}...")
        print(f"Training: {len(train_images)} images ({train_labels.count(1)} drones, {train_labels.count(0)} birds)")
        print(f"Validation: {len(val_images)} images ({val_labels.count(1)} drones, {val_labels.count(0)} birds)")
        
        best_val_acc = 0.0
        train_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_accs.append(val_acc)
            scheduler.step(train_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        self.training_stats = {
            'train_loss': train_losses[-1],
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'n_drone_samples': len(drone_images),
            'n_bird_samples': len(bird_images),
            'epochs': epochs,
            'model_type': 'PyTorch_CNN'
        }
        
        self.is_trained = True
        print(f"✓ Training complete")
        print(f"  - Best Validation Accuracy: {best_val_acc:.2f}%")
        
        return True
    
    def predict(self, contour, image=None):
        """
        Predict if contour is drone or bird
        
        Args:
            contour: OpenCV contour
            image: Original image (required for better accuracy)
        
        Returns:
            (is_drone, probability)
        """
        if not TORCH_AVAILABLE or not self.is_trained or self.model is None:
            return False, 0.0
        
        try:
            # Extract ROI from contour - need image
            if image is None:
                # If no image provided, return False (can't predict without image)
                return False, 0.0
            
            x, y, w, h = cv2.boundingRect(contour)
            # Add padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            roi = image[y:y+h, x:x+w]
            if roi.size == 0 or w < 10 or h < 10:
                return False, 0.0
            
            # Preprocess
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)
            roi_tensor = self.val_transform(roi_pil).unsqueeze(0).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(roi_tensor)
                probs = torch.softmax(outputs, dim=1)
                drone_prob = probs[0][1].item()
            
            return drone_prob > 0.5, drone_prob
        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0
    
    def save(self, filepath=None):
        """Save model"""
        if not self.is_trained:
            print("Error: Model is not trained yet")
            return False
        
        filepath = filepath or self.model_path
        
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'training_stats': self.training_stats
            }
            
            if not filepath.endswith('.pkl'):
                filepath = filepath.replace('.json', '.pkl')
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load(self, filepath=None):
        """Load model"""
        filepath = filepath or self.model_path
        
        try:
            if not os.path.exists(filepath):
                if filepath.endswith('.json'):
                    filepath = filepath.replace('.json', '.pkl')
                elif not filepath.endswith('.pkl'):
                    filepath = filepath + '.pkl'
            
            if not os.path.exists(filepath):
                print(f"Error: File not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model.load_state_dict(model_data['model_state_dict'])
            self.training_stats = model_data.get('training_stats', {})
            self.is_trained = True
            
            print(f"✓ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
