"""
GPS Model using PyTorch Neural Network
ใช้ Neural Network สำหรับความแม่นยำสูงและรองรับข้อมูลที่ซับซ้อน
"""

import numpy as np
import os
import pickle

try:
    from . import paths
except ImportError:
    # Fallback for when run as script
    import paths

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available: {e}")


if TORCH_AVAILABLE:
    class GPSDataset(Dataset):
        """Dataset สำหรับ GPS training"""
        def __init__(self, X, y_lat, y_lon, y_alt):
            self.X = torch.FloatTensor(X)
            self.y_lat = torch.FloatTensor(y_lat)
            self.y_lon = torch.FloatTensor(y_lon)
            self.y_alt = torch.FloatTensor(y_alt)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], (self.y_lat[idx], self.y_lon[idx], self.y_alt[idx])


    class GPSNet(nn.Module):
        """Neural Network สำหรับ GPS prediction"""
        def __init__(self, input_dim=2, hidden_dims=[64, 128, 64], dropout=0.2):
            super(GPSNet, self).__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            self.shared_layers = nn.Sequential(*layers)
            
            # Separate heads for lat, lon, alt
            self.lat_head = nn.Linear(prev_dim, 1)
            self.lon_head = nn.Linear(prev_dim, 1)
            self.alt_head = nn.Linear(prev_dim, 1)
        
        def forward(self, x):
            shared = self.shared_layers(x)
            lat = self.lat_head(shared)
            lon = self.lon_head(shared)
            alt = self.alt_head(shared)
            return lat.squeeze(), lon.squeeze(), alt.squeeze()
else:
    # Dummy classes if PyTorch not available
    class GPSDataset:
        pass
    class GPSNet:
        pass


class GPSModelPyTorch:
    def __init__(self, hidden_dims=[64, 128, 64], dropout=0.2, learning_rate=0.001):
        """
        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        if not TORCH_AVAILABLE:
            self.is_trained = False
            self.model = None
            self.device = None
            self.learning_rate = learning_rate
            self.scaler_X = None
            self.scaler_lat = None
            self.scaler_lon = None
            self.scaler_alt = None
            self.training_stats = {}
            self.model_path = str(paths.GPS_MODEL_PYTORCH_FILE)
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPSNet(hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        self.learning_rate = learning_rate
        
        # Feature scalers
        self.scaler_X = StandardScaler()
        self.scaler_lat = StandardScaler()
        self.scaler_lon = StandardScaler()
        self.scaler_alt = StandardScaler()
        
        self.is_trained = False
        self.training_stats = {}
        self.model_path = str(paths.GPS_MODEL_PYTORCH_FILE)
    
    def train(self, data_points, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model using data points.
        
        Args:
            data_points: List of tuples (center_x, center_y, lat, lon, alt)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
            
        Returns:
            bool: Success status
        """
        if not data_points or len(data_points) < 10:
            print(f"Error: Need at least 10 data points (got {len(data_points)})")
            return False
        
        # Prepare data
        X = []
        Y_lat = []
        Y_lon = []
        Y_alt = []
        
        for x, y, lat, lon, alt in data_points:
            X.append([x, y])
            Y_lat.append(lat)
            Y_lon.append(lon)
            Y_alt.append(alt)
        
        X = np.array(X)
        Y_lat = np.array(Y_lat)
        Y_lon = np.array(Y_lon)
        Y_alt = np.array(Y_alt)
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        Y_lat_scaled = self.scaler_lat.fit_transform(Y_lat.reshape(-1, 1)).flatten()
        Y_lon_scaled = self.scaler_lon.fit_transform(Y_lon.reshape(-1, 1)).flatten()
        Y_alt_scaled = self.scaler_alt.fit_transform(Y_alt.reshape(-1, 1)).flatten()
        
        # Split train/validation
        n_val = int(len(X_scaled) * validation_split)
        if n_val < 2:
            n_val = 0
        
        if n_val > 0:
            X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
            y_lat_train, y_lat_val = Y_lat_scaled[:-n_val], Y_lat_scaled[-n_val:]
            y_lon_train, y_lon_val = Y_lon_scaled[:-n_val], Y_lon_scaled[-n_val:]
            y_alt_train, y_alt_val = Y_alt_scaled[:-n_val], Y_alt_scaled[-n_val:]
        else:
            X_train, X_val = X_scaled, X_scaled
            y_lat_train, y_lat_val = Y_lat_scaled, Y_lat_scaled
            y_lon_train, y_lon_val = Y_lon_scaled, Y_lon_scaled
            y_alt_train, y_alt_val = Y_alt_scaled, Y_alt_scaled
        
        # Create datasets
        train_dataset = GPSDataset(X_train, y_lat_train, y_lon_train, y_alt_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print(f"Training PyTorch GPS Model on {self.device}...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, (y_lat_batch, y_lon_batch, y_alt_batch) in train_loader:
                X_batch = X_batch.to(self.device)
                y_lat_batch = y_lat_batch.to(self.device)
                y_lon_batch = y_lon_batch.to(self.device)
                y_alt_batch = y_alt_batch.to(self.device)
                
                optimizer.zero_grad()
                lat_pred, lon_pred, alt_pred = self.model(X_batch)
                
                loss = (criterion(lat_pred, y_lat_batch) + 
                       criterion(lon_pred, y_lon_batch) + 
                       criterion(alt_pred, y_alt_batch)) / 3
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if n_val > 0:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_lat_val_tensor = torch.FloatTensor(y_lat_val).to(self.device)
                    y_lon_val_tensor = torch.FloatTensor(y_lon_val).to(self.device)
                    y_alt_val_tensor = torch.FloatTensor(y_alt_val).to(self.device)
                    
                    lat_pred, lon_pred, alt_pred = self.model(X_val_tensor)
                    
                    val_loss = (criterion(lat_pred, y_lat_val_tensor) + 
                              criterion(lon_pred, y_lon_val_tensor) + 
                              criterion(alt_pred, y_alt_val_tensor)) / 3
                    val_loss = val_loss.item()
                    val_losses.append(val_loss)
                    
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
            else:
                val_loss = train_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}" + 
                      (f", Val Loss: {val_loss:.6f}" if n_val > 0 else ""))
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            lat_pred, lon_pred, alt_pred = self.model(X_tensor)
            
            # Unscale predictions
            lat_pred_unscaled = self.scaler_lat.inverse_transform(lat_pred.cpu().numpy().reshape(-1, 1)).flatten()
            lon_pred_unscaled = self.scaler_lon.inverse_transform(lon_pred.cpu().numpy().reshape(-1, 1)).flatten()
            alt_pred_unscaled = self.scaler_alt.inverse_transform(alt_pred.cpu().numpy().reshape(-1, 1)).flatten()
            
            # Calculate R² scores
            from sklearn.metrics import r2_score
            r2_lat = r2_score(Y_lat, lat_pred_unscaled)
            r2_lon = r2_score(Y_lon, lon_pred_unscaled)
            r2_alt = r2_score(Y_alt, alt_pred_unscaled)
        
        self.training_stats = {
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1] if val_losses else train_losses[-1],
            'r2_lat': r2_lat,
            'r2_lon': r2_lon,
            'r2_alt': r2_alt,
            'n_samples': len(data_points),
            'epochs': epochs,
            'model_type': 'PyTorch_NeuralNetwork'
        }
        
        self.is_trained = True
        print(f"[OK] Training complete with {len(data_points)} samples")
        print(f"  - Lat R²: {r2_lat:.4f}")
        print(f"  - Lon R²: {r2_lon:.4f}")
        print(f"  - Alt R²: {r2_alt:.4f}")
        
        return True
    
    def predict(self, center_x, center_y):
        """Predict GPS coordinates from pixel position"""
        if not self.is_trained:
            return None
        
        try:
            self.model.eval()
            with torch.no_grad():
                X = np.array([[center_x, center_y]])
                X_scaled = self.scaler_X.transform(X)
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                lat_pred, lon_pred, alt_pred = self.model(X_tensor)
                
                # Unscale predictions
                lat = self.scaler_lat.inverse_transform(lat_pred.cpu().numpy().reshape(-1, 1))[0, 0]
                lon = self.scaler_lon.inverse_transform(lon_pred.cpu().numpy().reshape(-1, 1))[0, 0]
                alt = self.scaler_alt.inverse_transform(alt_pred.cpu().numpy().reshape(-1, 1))[0, 0]
                
                return lat, lon, alt
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def save(self, filepath):
        """Save model"""
        if not self.is_trained:
            print("Error: Model is not trained yet")
            return False
        
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler_X': self.scaler_X,
                'scaler_lat': self.scaler_lat,
                'scaler_lon': self.scaler_lon,
                'scaler_alt': self.scaler_alt,
                'training_stats': self.training_stats,
                'model_config': {
                    'hidden_dims': [64, 128, 64],
                    'dropout': 0.2
                }
            }
            
            if not filepath.endswith('.pkl'):
                filepath = filepath.replace('.json', '.pkl')
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"[OK] Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load(self, filepath):
        """Load model"""
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
            self.scaler_X = model_data['scaler_X']
            self.scaler_lat = model_data['scaler_lat']
            self.scaler_lon = model_data['scaler_lon']
            self.scaler_alt = model_data['scaler_alt']
            self.training_stats = model_data.get('training_stats', {})
            
            self.is_trained = True
            print(f"[OK] Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()
            return False
