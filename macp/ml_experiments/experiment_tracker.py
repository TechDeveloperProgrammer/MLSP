import os
import uuid
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import tensorflow as tf
import mlflow

class ExperimentStatus:
    """Experiment tracking status"""
    INITIALIZED = 'initialized'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    INTERRUPTED = 'interrupted'

@dataclass
class MLExperiment:
    """Comprehensive ML experiment tracking"""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = 'Unnamed Experiment'
    description: str = ''
    status: str = ExperimentStatus.INITIALIZED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate experiment duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

class ExperimentTracker:
    """
    Advanced Machine Learning Experiment Tracking System
    
    Features:
    - Comprehensive experiment logging
    - MLflow integration
    - Artifact management
    - Performance tracking
    """
    
    def __init__(
        self, 
        tracking_uri: str = '/home/veronicae/CascadeProjects/MLSP/ml_experiments',
        experiment_name: str = 'MLSP_Experiments'
    ):
        """
        Initialize experiment tracker
        
        :param tracking_uri: MLflow tracking URI
        :param experiment_name: MLflow experiment name
        """
        # Create tracking directory
        os.makedirs(tracking_uri, exist_ok=True)
        
        # Configure MLflow
        mlflow.set_tracking_uri(f'file:{tracking_uri}')
        mlflow.set_experiment(experiment_name)
        
        # Experiment storage
        self.experiments_dir = os.path.join(tracking_uri, 'experiments')
        os.makedirs(self.experiments_dir, exist_ok=True)
    
    def start_experiment(
        self, 
        name: str, 
        description: str = '',
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> MLExperiment:
        """
        Start a new machine learning experiment
        
        :param name: Experiment name
        :param description: Experiment description
        :param parameters: Experiment parameters
        :param tags: Experiment tags
        :return: Initialized experiment
        """
        experiment = MLExperiment(
            name=name,
            description=description,
            parameters=parameters or {},
            tags=tags or [],
            status=ExperimentStatus.RUNNING
        )
        
        # Start MLflow run
        mlflow.start_run(run_name=name)
        
        # Log initial parameters
        for key, value in (parameters or {}).items():
            mlflow.log_param(key, value)
        
        # Log tags
        for tag in (tags or []):
            mlflow.set_tag(tag, 'true')
        
        # Save experiment metadata
        self._save_experiment(experiment)
        
        return experiment
    
    def log_metrics(
        self, 
        experiment_id: str, 
        metrics: Dict[str, float]
    ):
        """
        Log metrics for an experiment
        
        :param experiment_id: Experiment identifier
        :param metrics: Metrics dictionary
        """
        experiment = self._load_experiment(experiment_id)
        
        # Update experiment metrics
        for key, value in metrics.items():
            if key not in experiment.metrics:
                experiment.metrics[key] = []
            experiment.metrics[key].append(value)
        
        # Log to MLflow
        for key, values in metrics.items():
            mlflow.log_metric(key, values[-1])
        
        self._save_experiment(experiment)
    
    def log_artifact(
        self, 
        experiment_id: str, 
        artifact_path: str
    ):
        """
        Log experiment artifacts
        
        :param experiment_id: Experiment identifier
        :param artifact_path: Path to artifact
        """
        experiment = self._load_experiment(experiment_id)
        
        # Log artifact to MLflow
        mlflow.log_artifact(artifact_path)
        
        # Update experiment artifacts
        experiment.artifacts.append(artifact_path)
        self._save_experiment(experiment)
    
    def end_experiment(
        self, 
        experiment_id: str, 
        status: str = ExperimentStatus.COMPLETED
    ):
        """
        End an experiment
        
        :param experiment_id: Experiment identifier
        :param status: Experiment final status
        """
        experiment = self._load_experiment(experiment_id)
        
        # Update experiment status
        experiment.status = status
        experiment.end_time = time.time()
        
        # End MLflow run
        mlflow.end_run()
        
        self._save_experiment(experiment)
    
    def _save_experiment(self, experiment: MLExperiment):
        """
        Save experiment metadata
        
        :param experiment: Experiment to save
        """
        experiment_path = os.path.join(
            self.experiments_dir, 
            f'{experiment.experiment_id}.json'
        )
        
        with open(experiment_path, 'w') as f:
            json.dump(asdict(experiment), f, indent=2)
    
    def _load_experiment(
        self, 
        experiment_id: str
    ) -> MLExperiment:
        """
        Load experiment metadata
        
        :param experiment_id: Experiment identifier
        :return: Loaded experiment
        """
        experiment_path = os.path.join(
            self.experiments_dir, 
            f'{experiment_id}.json'
        )
        
        with open(experiment_path, 'r') as f:
            experiment_data = json.load(f)
        
        return MLExperiment(**experiment_data)
    
    def list_experiments(
        self, 
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[MLExperiment]:
        """
        List experiments with optional filtering
        
        :param status: Filter by experiment status
        :param tags: Filter by experiment tags
        :return: List of experiments
        """
        experiments = []
        
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.experiments_dir, filename), 'r') as f:
                    experiment_data = json.load(f)
                    experiment = MLExperiment(**experiment_data)
                    
                    # Apply filters
                    if status and experiment.status != status:
                        continue
                    
                    if tags and not all(tag in experiment.tags for tag in tags):
                        continue
                    
                    experiments.append(experiment)
        
        return experiments

def main():
    """Demonstration of experiment tracking system"""
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Start an experiment
    experiment = tracker.start_experiment(
        name='Terrain Generation Model Training',
        description='Advanced ML model for procedural terrain generation',
        parameters={
            'learning_rate': 0.001,
            'batch_size': 32,
            'model_type': 'transformer'
        },
        tags=['terrain', 'world-generation', 'ml-model']
    )
    
    # Simulate training process
    for epoch in range(10):
        # Log metrics
        tracker.log_metrics(
            experiment.experiment_id, 
            {
                'loss': np.random.uniform(0.1, 0.5),
                'accuracy': np.random.uniform(0.6, 0.9)
            }
        )
    
    # Log model artifact
    tracker.log_artifact(
        experiment.experiment_id, 
        '/path/to/trained/model.pth'
    )
    
    # End experiment
    tracker.end_experiment(
        experiment.experiment_id, 
        status=ExperimentStatus.COMPLETED
    )
    
    # List experiments
    print("Completed Experiments:")
    completed_experiments = tracker.list_experiments(
        status=ExperimentStatus.COMPLETED,
        tags=['terrain']
    )
    
    for exp in completed_experiments:
        print(json.dumps(asdict(exp), indent=2))

if __name__ == '__main__':
    main()
