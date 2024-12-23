import os
import numpy as np
import pandas as pd
import sqlite3
import logging
import joblib
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

class MLPerformanceOptimizer:
    """
    Advanced Machine Learning-driven Performance Optimization System
    
    Features:
    - Predictive server performance modeling
    - Resource utilization forecasting
    - Proactive performance optimization recommendations
    - Anomaly detection
    - Adaptive configuration suggestions
    """
    
    def __init__(self, server_monitor_db: str, model_dir: str = '/opt/mlsp/ml_models'):
        """
        Initialize ML Performance Optimizer
        
        :param server_monitor_db: Path to server monitoring SQLite database
        :param model_dir: Directory to store trained ML models
        """
        self.server_monitor_db = server_monitor_db
        self.model_dir = model_dir
        
        # Create model directory if not exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('MLSPPerformanceOptimizer')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_metrics_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Load server metrics from SQLite database
        
        :param hours: Number of historical hours to retrieve
        :return: Pandas DataFrame with server metrics
        """
        try:
            with sqlite3.connect(self.server_monitor_db) as conn:
                # Calculate timestamp cutoff
                cutoff_time = datetime.now().timestamp() - (hours * 3600)
                
                query = f'''
                    SELECT * FROM server_metrics 
                    WHERE timestamp >= {cutoff_time} 
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(query, conn)
                
                # Preprocess and feature engineering
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                return df
        
        except Exception as e:
            self.logger.error(f"Error loading metrics data: {e}")
            raise
    
    def train_performance_prediction_model(self):
        """
        Train machine learning models for performance prediction
        
        Models:
        1. Resource Utilization Prediction (Regression)
        2. Performance Anomaly Detection (Classification)
        3. Time Series Performance Forecasting (LSTM)
        """
        try:
            # Load metrics data
            df = self._load_metrics_data(hours=168)  # 1 week of data
            
            # Feature selection and preprocessing
            features = [
                'cpu_usage', 'memory_usage', 'disk_usage', 
                'network_io_in', 'network_io_out', 'player_count', 
                'active_chunks', 'entity_count'
            ]
            
            X = df[features]
            y_regression = df['tps']  # Regression target: Server TPS
            y_classification = (df['tps'] < 18).astype(int)  # Classification target: Performance anomaly
            
            # Split data
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X, y_regression, test_size=0.2, random_state=42
            )
            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
                X, y_classification, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_reg_scaled = scaler.fit_transform(X_train_reg)
            X_test_reg_scaled = scaler.transform(X_test_reg)
            
            # 1. Random Forest Regression Model (TPS Prediction)
            rf_regressor = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=10
            )
            rf_regressor.fit(X_train_reg_scaled, y_train_reg)
            
            # Evaluate regression model
            y_pred_reg = rf_regressor.predict(X_test_reg_scaled)
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            self.logger.info(f"Regression Model MSE: {mse}")
            
            # 2. Random Forest Classification Model (Anomaly Detection)
            rf_classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10
            )
            rf_classifier.fit(X_train_cls, y_train_cls)
            
            # Evaluate classification model
            y_pred_cls = rf_classifier.predict(X_test_cls)
            self.logger.info("Classification Report:\n" + 
                             classification_report(y_test_cls, y_pred_cls))
            
            # 3. LSTM Time Series Forecasting
            def create_sequences(data, seq_length=12):
                """Create sequence data for LSTM"""
                sequences = []
                targets = []
                for i in range(len(data) - seq_length):
                    seq = data[i:i+seq_length]
                    target = data[i+seq_length]
                    sequences.append(seq)
                    targets.append(target)
                return np.array(sequences), np.array(targets)
            
            # Prepare LSTM data
            lstm_data = X[['tps']].values
            X_lstm, y_lstm = create_sequences(lstm_data)
            
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_lstm, y_lstm, test_size=0.2, random_state=42
            )
            
            # LSTM Model
            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            
            # Train LSTM
            lstm_model.fit(
                X_train_lstm, y_train_lstm, 
                epochs=50, 
                batch_size=32, 
                validation_split=0.2,
                verbose=0
            )
            
            # Save models and scaler
            joblib.dump(rf_regressor, os.path.join(self.model_dir, 'rf_regressor.joblib'))
            joblib.dump(rf_classifier, os.path.join(self.model_dir, 'rf_classifier.joblib'))
            joblib.dump(scaler, os.path.join(self.model_dir, 'feature_scaler.joblib'))
            lstm_model.save(os.path.join(self.model_dir, 'lstm_forecaster.h5'))
            
            self.logger.info("Performance prediction models trained successfully")
        
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise
    
    def predict_performance(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict server performance based on current metrics
        
        :param current_metrics: Current server metrics
        :return: Performance prediction results
        """
        try:
            # Load saved models and scaler
            regressor = joblib.load(os.path.join(self.model_dir, 'rf_regressor.joblib'))
            classifier = joblib.load(os.path.join(self.model_dir, 'rf_classifier.joblib'))
            scaler = joblib.load(os.path.join(self.model_dir, 'feature_scaler.joblib'))
            
            # Prepare input data
            features = [
                'cpu_usage', 'memory_usage', 'disk_usage', 
                'network_io_in', 'network_io_out', 'player_count', 
                'active_chunks', 'entity_count'
            ]
            
            # Convert input to numpy array
            input_data = np.array([current_metrics.get(f, 0) for f in features]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            # Predictions
            tps_prediction = regressor.predict(input_scaled)[0]
            anomaly_prediction = classifier.predict(input_scaled)[0]
            
            return {
                'predicted_tps': tps_prediction,
                'is_performance_anomaly': bool(anomaly_prediction),
                'performance_recommendations': self._generate_recommendations(
                    tps_prediction, 
                    current_metrics
                )
            }
        
        except Exception as e:
            self.logger.error(f"Performance prediction error: {e}")
            raise
    
    def _generate_recommendations(self, predicted_tps: float, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate performance optimization recommendations
        
        :param predicted_tps: Predicted Ticks Per Second
        :param metrics: Current server metrics
        :return: List of performance recommendations
        """
        recommendations = []
        
        # TPS-based recommendations
        if predicted_tps < 18:
            recommendations.append("Server performance is degrading. Consider reducing active chunks.")
        
        # CPU usage recommendations
        if metrics.get('cpu_usage', 0) > 75:
            recommendations.append("High CPU usage detected. Optimize server-side plugins.")
        
        # Memory usage recommendations
        if metrics.get('memory_usage', 0) > 80:
            recommendations.append("High memory consumption. Increase allocated memory or reduce loaded chunks.")
        
        # Entity and chunk management
        if metrics.get('entity_count', 0) > 500:
            recommendations.append("High entity count may impact performance. Consider entity cramming settings.")
        
        if metrics.get('active_chunks', 0) > 1000:
            recommendations.append("Reduce view and simulation distance to improve performance.")
        
        return recommendations

def main():
    """Example usage of ML Performance Optimizer"""
    # Assuming server_metrics.sqlite exists from previous monitoring
    optimizer = MLPerformanceOptimizer('/path/to/server_metrics.sqlite')
    
    try:
        # Train performance prediction models
        optimizer.train_performance_prediction_model()
        
        # Example current metrics
        current_metrics = {
            'cpu_usage': 65.5,
            'memory_usage': 75.2,
            'disk_usage': 40.1,
            'network_io_in': 1024,
            'network_io_out': 512,
            'player_count': 25,
            'active_chunks': 800,
            'entity_count': 350
        }
        
        # Predict performance
        performance_prediction = optimizer.predict_performance(current_metrics)
        
        print("Performance Prediction:")
        print(f"Predicted TPS: {performance_prediction['predicted_tps']}")
        print(f"Performance Anomaly: {performance_prediction['is_performance_anomaly']}")
        print("Recommendations:")
        for rec in performance_prediction['performance_recommendations']:
            print(f"- {rec}")
    
    except Exception as e:
        print(f"Performance optimization error: {e}")

if __name__ == '__main__':
    main()
