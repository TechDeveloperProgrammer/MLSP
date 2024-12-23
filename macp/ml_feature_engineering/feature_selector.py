import os
import json
import uuid
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.feature_selection
import sklearn.preprocessing
import sklearn.ensemble
import shap

class FeatureSelectionStrategy(Enum):
    """Comprehensive feature selection strategies"""
    STATISTICAL = auto()
    CORRELATION = auto()
    MUTUAL_INFORMATION = auto()
    RECURSIVE_FEATURE_ELIMINATION = auto()
    TREE_BASED = auto()
    SHAP_BASED = auto()
    CUSTOM = auto()

class FeatureTransformationType(Enum):
    """Feature transformation techniques"""
    SCALING = auto()
    NORMALIZATION = auto()
    STANDARDIZATION = auto()
    POLYNOMIAL = auto()
    LOG_TRANSFORM = auto()
    BINNING = auto()
    ONE_HOT_ENCODING = auto()
    CUSTOM = auto()

@dataclass
class FeatureEngineeringConfig:
    """Comprehensive feature engineering configuration"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    selection_strategy: FeatureSelectionStrategy = FeatureSelectionStrategy.STATISTICAL
    transformation_techniques: List[FeatureTransformationType] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_feature: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class FeatureSelector:
    """
    Advanced Machine Learning Feature Engineering and Selection Framework
    
    Features:
    - Multi-strategy feature selection
    - Advanced feature transformation
    - Statistical and model-based feature importance
    - Customizable engineering techniques
    """
    
    def __init__(
        self, 
        output_dir: str = '/home/veronicae/CascadeProjects/MLSP/feature_engineering'
    ):
        """
        Initialize feature selector
        
        :param output_dir: Directory to store feature engineering outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature engineering history storage
        self.engineering_history_file = os.path.join(output_dir, 'engineering_history.json')
        self.engineering_history: Dict[str, Dict[str, Any]] = self._load_engineering_history()
    
    def _load_engineering_history(self) -> Dict[str, Dict[str, Any]]:
        """Load existing feature engineering history"""
        if os.path.exists(self.engineering_history_file):
            with open(self.engineering_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_engineering_history(
        self, 
        config_id: str, 
        engineering_data: Dict[str, Any]
    ):
        """Save feature engineering results to history"""
        self.engineering_history[config_id] = engineering_data
        
        with open(self.engineering_history_file, 'w') as f:
            json.dump(self.engineering_history, f, indent=2)
    
    def transform_features(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        config: Optional[FeatureEngineeringConfig] = None
    ) -> Dict[str, Any]:
        """
        Apply feature transformations
        
        :param data: Input data
        :param config: Feature engineering configuration
        :return: Transformed features and metadata
        """
        # Create default configuration
        if not config:
            config = FeatureEngineeringConfig(
                transformation_techniques=[
                    FeatureTransformationType.SCALING,
                    FeatureTransformationType.NORMALIZATION
                ]
            )
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Feature transformation results
        transformed_data = data.copy()
        transformation_metadata = {}
        
        for technique in config.transformation_techniques:
            if technique == FeatureTransformationType.SCALING:
                # Min-Max scaling
                scaler = sklearn.preprocessing.MinMaxScaler()
                transformed_data = pd.DataFrame(
                    scaler.fit_transform(transformed_data),
                    columns=transformed_data.columns
                )
                transformation_metadata['scaling'] = {
                    'min': scaler.data_min_.tolist(),
                    'max': scaler.data_max_.tolist()
                }
            
            elif technique == FeatureTransformationType.NORMALIZATION:
                # L2 normalization
                normalizer = sklearn.preprocessing.Normalizer()
                transformed_data = pd.DataFrame(
                    normalizer.fit_transform(transformed_data),
                    columns=transformed_data.columns
                )
            
            elif technique == FeatureTransformationType.STANDARDIZATION:
                # Z-score standardization
                standardizer = sklearn.preprocessing.StandardScaler()
                transformed_data = pd.DataFrame(
                    standardizer.fit_transform(transformed_data),
                    columns=transformed_data.columns
                )
                transformation_metadata['standardization'] = {
                    'mean': standardizer.mean_.tolist(),
                    'std': standardizer.scale_.tolist()
                }
            
            elif technique == FeatureTransformationType.LOG_TRANSFORM:
                # Log transformation
                transformed_data = np.log1p(transformed_data)
            
            elif technique == FeatureTransformationType.ONE_HOT_ENCODING:
                # One-hot encoding for categorical features
                transformed_data = pd.get_dummies(transformed_data)
        
        # Prepare results
        results = {
            'metadata': asdict(config),
            'transformed_data': transformed_data.values.tolist(),
            'feature_names': transformed_data.columns.tolist(),
            'transformation_metadata': transformation_metadata
        }
        
        # Save engineering history
        self._save_engineering_history(config.config_id, results)
        
        return results
    
    def select_features(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        target: Optional[np.ndarray] = None,
        config: Optional[FeatureEngineeringConfig] = None
    ) -> Dict[str, Any]:
        """
        Select most important features
        
        :param data: Input features
        :param target: Target variable
        :param config: Feature selection configuration
        :return: Selected features and importance scores
        """
        # Create default configuration
        if not config:
            config = FeatureEngineeringConfig(
                selection_strategy=FeatureSelectionStrategy.STATISTICAL
            )
        
        # Convert to DataFrame and NumPy if needed
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns
            data = data.values
        else:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        # Feature selection results
        feature_importance = {}
        selected_features_mask = None
        
        if config.selection_strategy == FeatureSelectionStrategy.STATISTICAL:
            # Univariate feature selection
            selector = sklearn.feature_selection.SelectKBest(
                score_func=sklearn.feature_selection.f_classif, 
                k=config.parameters.get('k', min(10, data.shape[1]))
            )
            selector.fit(data, target)
            selected_features_mask = selector.get_support()
            
            feature_importance = {
                feature_names[i]: selector.scores_[i] 
                for i in range(len(feature_names)) 
                if selected_features_mask[i]
            }
        
        elif config.selection_strategy == FeatureSelectionStrategy.CORRELATION:
            # Correlation-based feature selection
            correlation_matrix = np.abs(np.corrcoef(data.T))
            feature_importance = {
                feature_names[i]: np.mean(correlation_matrix[i]) 
                for i in range(len(feature_names))
            }
            
            # Select top features
            k = config.parameters.get('k', min(10, data.shape[1]))
            selected_features_mask = np.zeros(data.shape[1], dtype=bool)
            top_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
            for feature, _ in top_features:
                selected_features_mask[feature_names.tolist().index(feature)] = True
        
        elif config.selection_strategy == FeatureSelectionStrategy.TREE_BASED:
            # Random Forest feature importance
            rf_selector = sklearn.ensemble.RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
            rf_selector.fit(data, target)
            
            feature_importance = {
                feature_names[i]: rf_selector.feature_importances_[i] 
                for i in range(len(feature_names))
            }
            
            # Select top features
            k = config.parameters.get('k', min(10, data.shape[1]))
            selected_features_mask = np.zeros(data.shape[1], dtype=bool)
            top_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
            for feature, _ in top_features:
                selected_features_mask[feature_names.tolist().index(feature)] = True
        
        elif config.selection_strategy == FeatureSelectionStrategy.SHAP_BASED:
            # SHAP (SHapley Additive exPlanations) feature importance
            rf_model = sklearn.ensemble.RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
            rf_model.fit(data, target)
            
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(data)
            
            feature_importance = {
                feature_names[i]: np.abs(shap_values[1][:, i]).mean() 
                for i in range(len(feature_names))
            }
            
            # Select top features
            k = config.parameters.get('k', min(10, data.shape[1]))
            selected_features_mask = np.zeros(data.shape[1], dtype=bool)
            top_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k]
            
            for feature, _ in top_features:
                selected_features_mask[feature_names.tolist().index(feature)] = True
        
        # Prepare results
        results = {
            'metadata': asdict(config),
            'feature_importance': feature_importance,
            'selected_features': [
                feature_names[i] 
                for i in range(len(feature_names)) 
                if selected_features_mask[i]
            ],
            'selected_data': data[:, selected_features_mask].tolist() if selected_features_mask is not None else data.tolist()
        }
        
        # Save engineering history
        self._save_engineering_history(config.config_id, results)
        
        return results

def main():
    """Demonstration of feature engineering system"""
    # Initialize feature selector
    feature_selector = FeatureSelector()
    
    # Generate synthetic dataset
    np.random.seed(42)
    X = np.random.rand(1000, 20)
    y = (X[:, 0] + X[:, 5] + np.random.normal(0, 0.1, 1000) > 1.5).astype(int)
    
    # Feature transformation
    transformation_config = FeatureEngineeringConfig(
        transformation_techniques=[
            FeatureTransformationType.SCALING,
            FeatureTransformationType.NORMALIZATION
        ]
    )
    
    transformed_data = feature_selector.transform_features(
        X, 
        config=transformation_config
    )
    
    print("Feature Transformation:")
    print(json.dumps({
        'transformed_data_shape': [
            len(transformed_data['transformed_data']),
            len(transformed_data['transformed_data'][0])
        ]
    }, indent=2))
    
    # Feature selection
    selection_config = FeatureEngineeringConfig(
        selection_strategy=FeatureSelectionStrategy.SHAP_BASED,
        parameters={'k': 5}
    )
    
    feature_selection = feature_selector.select_features(
        X, 
        target=y, 
        config=selection_config
    )
    
    print("\nFeature Selection:")
    print(json.dumps({
        'selected_features': feature_selection['selected_features'],
        'feature_importance': feature_selection['feature_importance']
    }, indent=2))

if __name__ == '__main__':
    main()
