import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

import torch
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class ModelMetadata:
    """Comprehensive model metadata tracking"""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ''
    version: str = '0.1.0'
    created_at: datetime = field(default_factory=datetime.now)
    framework: str = ''
    task_type: str = ''
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_dataset: str = ''
    training_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat()
        }

class ModelPerformanceTracker:
    """Advanced model performance tracking and evaluation"""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    
    @staticmethod
    def calculate_model_size(model) -> int:
        """Calculate model size in bytes"""
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() * p.element_size() for p in model.parameters())
        elif isinstance(model, tf.keras.Model):
            return sum(np.prod(w.shape) * w.dtype.size for w in model.weights)
        return 0

class ModelRegistry:
    """Comprehensive ML model registry and version control"""
    
    def __init__(
        self, 
        registry_dir: str = '/home/veronicae/CascadeProjects/MLSP/model_registry'
    ):
        """Initialize model registry"""
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
    
    def _generate_model_hash(self, model_data: bytes) -> str:
        """Generate unique model hash"""
        return hashlib.sha256(model_data).hexdigest()
    
    def register_model(
        self, 
        model,
        metadata: ModelMetadata,
        performance_data: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Register a new model version
        
        :param model: Machine learning model
        :param metadata: Model metadata
        :param performance_data: Optional performance metrics
        :return: Model registration path
        """
        # Serialize model
        if isinstance(model, torch.nn.Module):
            model_data = torch.save(model.state_dict(), buffer=io.BytesIO()).getvalue()
            model_path = os.path.join(self.registry_dir, f'{metadata.model_id}.pth')
            torch.save(model.state_dict(), model_path)
        elif isinstance(model, tf.keras.Model):
            model_data = model.to_json().encode()
            model_path = os.path.join(self.registry_dir, f'{metadata.model_id}.h5')
            model.save(model_path)
        else:
            raise ValueError("Unsupported model type")
        
        # Calculate model hash
        model_hash = self._generate_model_hash(model_data)
        
        # Performance tracking
        performance_metrics = performance_data or {}
        performance_metrics['model_size'] = ModelPerformanceTracker.calculate_model_size(model)
        
        # Create registration metadata
        registration_metadata = {
            'model_id': metadata.model_id,
            'model_path': model_path,
            'model_hash': model_hash,
            'metadata': metadata.to_dict(),
            'performance_metrics': performance_metrics,
            'registered_at': datetime.now().isoformat()
        }
        
        # Save registration metadata
        metadata_path = os.path.join(
            self.registry_dir, 
            f'{metadata.model_id}_metadata.json'
        )
        with open(metadata_path, 'w') as f:
            json.dump(registration_metadata, f, indent=2)
        
        return metadata.model_id
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []
        for file in os.listdir(self.registry_dir):
            if file.endswith('_metadata.json'):
                with open(os.path.join(self.registry_dir, file), 'r') as f:
                    models.append(json.load(f))
        return models
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by ID"""
        metadata_path = os.path.join(
            self.registry_dir, 
            f'{model_id}_metadata.json'
        )
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def compare_models(
        self, 
        model_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Compare multiple model performances"""
        return [self.get_model(model_id) for model_id in model_ids]

def main():
    """Demonstration of model registry"""
    # Initialize registry
    registry = ModelRegistry()
    
    # Example PyTorch model
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = ExampleModel()
    
    # Create model metadata
    metadata = ModelMetadata(
        name='TerrainGenerationModel',
        version='0.1.0',
        framework='PyTorch',
        task_type='Terrain Generation',
        input_shape=[10],
        output_shape=[5],
        hyperparameters={
            'learning_rate': 0.001,
            'batch_size': 32
        },
        training_dataset='MinecraftTerrainDataset'
    )
    
    # Register model
    model_id = registry.register_model(
        model, 
        metadata, 
        performance_data={
            'accuracy': 0.95,
            'loss': 0.05
        }
    )
    
    # List and compare models
    print("Registered Models:", registry.list_models())
    print("Model Details:", registry.get_model(model_id))

if __name__ == '__main__':
    main()
