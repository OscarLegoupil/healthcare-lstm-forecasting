import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareSequenceDataset(Dataset):
    """PyTorch Dataset for healthcare reimbursement sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, clusters: np.ndarray, target_cluster: int = None):
        if target_cluster is not None:
            mask = clusters == target_cluster
            self.X = torch.FloatTensor(X[mask])
            self.y = torch.FloatTensor(y[mask])
            self.clusters = torch.LongTensor(clusters[mask])
        else:
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.clusters = torch.LongTensor(clusters)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.clusters[idx]


class ClusterLSTM(nn.Module):
    """
    Specialized LSTM model for a specific client cluster.
    Optimized architecture based on cluster behavioral patterns.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, prediction_horizon: int = 6):
        super(ClusterLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, prediction_horizon)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_hidden)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Ensure positive predictions for reimbursements
        out = torch.abs(out)
        
        return out


class ClusterLSTMEnsemble:
    """
    Ensemble of cluster-specific LSTM models for healthcare forecasting.
    Each cluster gets a specialized model optimized for its behavioral patterns.
    """
    
    def __init__(self, n_clusters: int = 3, input_size: int = 15, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3, prediction_horizon: int = 6):
        
        self.n_clusters = n_clusters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_horizon = prediction_horizon
        
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.training_history = {}
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models for each cluster
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LSTM models for each cluster."""
        for cluster_id in range(self.n_clusters):
            model = ClusterLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                prediction_horizon=self.prediction_horizon
            ).to(self.device)
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=10, verbose=True
            )
            
            self.models[cluster_id] = model
            self.optimizers[cluster_id] = optimizer
            self.schedulers[cluster_id] = scheduler
            self.training_history[cluster_id] = {'train_loss': [], 'val_loss': []}
    
    def _get_cluster_specific_hyperparams(self, cluster_id: int) -> Dict:
        """Get optimized hyperparameters for each cluster."""
        # These would typically be determined through hyperparameter optimization
        hyperparams = {
            0: {'lr': 0.001, 'batch_size': 64, 'hidden_size': 64},  # Senior Stable
            1: {'lr': 0.0015, 'batch_size': 32, 'hidden_size': 96}, # Young Volatile  
            2: {'lr': 0.0008, 'batch_size': 48, 'hidden_size': 80}  # Middle Moderate
        }
        return hyperparams.get(cluster_id, {'lr': 0.001, 'batch_size': 64, 'hidden_size': 64})
    
    def train_cluster_model(self, cluster_id: int, X: np.ndarray, y: np.ndarray, 
                           clusters: np.ndarray, epochs: int = 100, batch_size: int = 64,
                           validation_split: float = 0.2) -> Dict:
        """Train LSTM model for a specific cluster."""
        logger.info(f"Training model for cluster {cluster_id}...")
        
        # Filter data for this cluster
        cluster_mask = clusters == cluster_id
        if not np.any(cluster_mask):
            logger.warning(f"No data found for cluster {cluster_id}")
            return {}
        
        X_cluster = X[cluster_mask]
        y_cluster = y[cluster_mask]
        
        logger.info(f"Cluster {cluster_id}: {len(X_cluster):,} sequences")
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_cluster, y_cluster, test_size=validation_split, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = HealthcareSequenceDataset(
            X_train, y_train, np.full(len(X_train), cluster_id)
        )
        val_dataset = HealthcareSequenceDataset(
            X_val, y_val, np.full(len(X_val), cluster_id)
        )
        
        # Get cluster-specific hyperparameters
        hyperparams = self._get_cluster_specific_hyperparams(cluster_id)
        actual_batch_size = min(batch_size, len(train_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)
        
        model = self.models[cluster_id]
        optimizer = self.optimizers[cluster_id]
        scheduler = self.schedulers[cluster_id]
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_X, batch_y, _ in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y, _ in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            self.training_history[cluster_id]['train_loss'].append(avg_train_loss)
            self.training_history[cluster_id]['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"models/cluster_{cluster_id}_best.pth")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 20:
                logger.info(f"Cluster {cluster_id}, Epoch {epoch}: Train Loss {avg_train_loss:.4f}, "
                           f"Val Loss {avg_val_loss:.4f}")
            
            if patience_counter >= 20:
                logger.info(f"Early stopping for cluster {cluster_id} at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f"models/cluster_{cluster_id}_best.pth"))
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
    
    def train_all_clusters(self, X: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                          epochs: int = 100, batch_size: int = 64) -> Dict:
        """Train all cluster-specific models."""
        logger.info("Training all cluster models...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        training_results = {}
        
        for cluster_id in range(self.n_clusters):
            results = self.train_cluster_model(
                cluster_id, X, y, clusters, epochs, batch_size
            )
            training_results[cluster_id] = results
        
        # Save ensemble configuration
        self.save_models()
        
        logger.info("All cluster models trained successfully!")
        return training_results
    
    def predict(self, X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Make predictions using the appropriate cluster model."""
        predictions = np.zeros((len(X), self.prediction_horizon))
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = clusters == cluster_id
            if not np.any(cluster_mask):
                continue
            
            X_cluster = X[cluster_mask]
            if len(X_cluster) == 0:
                continue
            
            model = self.models[cluster_id]
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_cluster).to(self.device)
                cluster_predictions = model(X_tensor).cpu().numpy()
                predictions[cluster_mask] = cluster_predictions
        
        return predictions
    
    def save_models(self, save_dir: str = "models"):
        """Save all trained models and configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save model state dicts
        for cluster_id, model in self.models.items():
            torch.save(model.state_dict(), save_path / f"cluster_{cluster_id}_model.pth")
        
        # Save ensemble configuration
        config = {
            'n_clusters': self.n_clusters,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'prediction_horizon': self.prediction_horizon,
            'training_history': self.training_history
        }
        
        joblib.dump(config, save_path / "ensemble_config.pkl")
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, load_dir: str = "models"):
        """Load trained models and configuration."""
        load_path = Path(load_dir)
        
        # Load configuration
        config = joblib.load(load_path / "ensemble_config.pkl")
        
        self.n_clusters = config['n_clusters']
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.prediction_horizon = config['prediction_horizon']
        self.training_history = config['training_history']
        
        # Reinitialize and load models
        self._initialize_models()
        
        for cluster_id in range(self.n_clusters):
            model_path = load_path / f"cluster_{cluster_id}_model.pth"
            if model_path.exists():
                self.models[cluster_id].load_state_dict(torch.load(model_path, map_location=self.device))
        
        logger.info(f"Models loaded from {load_path}")


def main():
    """Example usage of cluster-specific LSTM ensemble."""
    # Load preprocessed data
    X = np.load("data/processed/X_sequences.npy")
    y = np.load("data/processed/y_sequences.npy")
    clusters = np.load("data/processed/sequence_clusters.npy")
    
    # Initialize and train ensemble
    ensemble = ClusterLSTMEnsemble(
        n_clusters=3,
        input_size=X.shape[-1],
        prediction_horizon=y.shape[-1]
    )
    
    # Train models
    results = ensemble.train_all_clusters(X, y, clusters, epochs=50)
    
    # Make predictions
    predictions = ensemble.predict(X[:100], clusters[:100])
    
    print("Cluster LSTM ensemble training completed!")
    print(f"Training results: {results}")


if __name__ == "__main__":
    main()