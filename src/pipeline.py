import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import joblib
from datetime import datetime, timedelta

from src.data.preprocessing import HealthcareDataProcessor
from src.data.clustering import ClientSegmentation
from src.models.lstm_cluster import ClusterLSTMEnsemble
from src.evaluation.metrics import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareForecastingPipeline:
    """
    Production-ready healthcare reimbursement forecasting pipeline.
    
    Combines client segmentation with cluster-specific LSTM models for
    enhanced prediction accuracy across different behavioral patterns.
    
    Features:
    - Automated data preprocessing and feature engineering
    - K-means client segmentation with behavioral profiling
    - Cluster-specific LSTM models optimized per segment
    - Comprehensive evaluation and uncertainty quantification
    - Production deployment utilities
    """
    
    def __init__(self, sequence_length: int = 12, prediction_horizon: int = 6,
                 n_clusters: int = 3, model_config: Optional[Dict] = None):
        """
        Initialize the forecasting pipeline.
        
        Args:
            sequence_length: Number of historical months for LSTM input
            prediction_horizon: Number of months to forecast
            n_clusters: Number of client clusters for segmentation
            model_config: Custom LSTM configuration parameters
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.n_clusters = n_clusters
        
        # Default model configuration
        self.model_config = model_config or {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'epochs': 100,
            'batch_size': 64
        }
        
        # Pipeline components
        self.data_processor = HealthcareDataProcessor()
        self.segmentation = ClientSegmentation(n_clusters=n_clusters)
        self.model_ensemble = None
        self.evaluator = ModelEvaluator()
        
        # Pipeline state
        self.is_trained = False
        self.training_history = {}
        self.cluster_profiles = {}
        
        logger.info(f"Healthcare Forecasting Pipeline initialized: "
                   f"{sequence_length}→{prediction_horizon} months, {n_clusters} clusters")
    
    def fit(self, data_path: str = "data/raw", validation_split: float = 0.2,
            save_artifacts: bool = True) -> Dict:
        """
        Train the complete forecasting pipeline.
        
        Args:
            data_path: Path to raw CSV data files
            validation_split: Fraction of data for validation
            save_artifacts: Whether to save trained models and artifacts
            
        Returns:
            Dictionary containing training results and performance metrics
        """
        logger.info("Starting pipeline training...")
        start_time = datetime.now()
        
        # Step 1: Data preprocessing
        logger.info("Step 1/4: Data preprocessing and feature engineering")
        self.data_processor.data_path = Path(data_path)
        preprocessing_results = self.data_processor.process_pipeline(
            self.sequence_length, self.prediction_horizon
        )
        
        # Step 2: Client segmentation
        logger.info("Step 2/4: Client segmentation and clustering")
        clustering_features = self.segmentation.prepare_clustering_features(
            preprocessing_results['feature_data']
        )
        features_with_clusters = self.segmentation.fit_clustering(clustering_features)
        
        # Create visualizations
        self.segmentation.visualize_clusters(features_with_clusters)
        
        # Assign clusters to sequences
        sequence_clusters = self.segmentation.assign_clusters_to_sequences(
            preprocessing_results['person_ids'], features_with_clusters
        )
        
        # Step 3: Model training
        logger.info("Step 3/4: Training cluster-specific LSTM models")
        self.model_ensemble = ClusterLSTMEnsemble(
            n_clusters=self.n_clusters,
            input_size=preprocessing_results['n_features'],
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout'],
            prediction_horizon=self.prediction_horizon
        )
        
        training_results = self.model_ensemble.train_all_clusters(
            preprocessing_results['X'],
            preprocessing_results['y'],
            sequence_clusters,
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size']
        )
        
        # Step 4: Model evaluation
        logger.info("Step 4/4: Model evaluation and validation")
        
        # Generate predictions for evaluation
        predictions = self.model_ensemble.predict(
            preprocessing_results['X'], sequence_clusters
        )
        
        # Evaluate predictions
        evaluation_results = self.evaluator.evaluate_predictions(
            preprocessing_results['y'], predictions, sequence_clusters
        )
        
        # Generate evaluation report
        report_path = self.evaluator.create_evaluation_report(evaluation_results)
        
        # Store pipeline state
        self.is_trained = True
        self.training_history = training_results
        self.cluster_profiles = self.segmentation.cluster_profiles
        
        # Save artifacts
        if save_artifacts:
            self._save_pipeline_artifacts()
        
        # Compile results
        training_duration = datetime.now() - start_time
        
        pipeline_results = {
            'training_duration': str(training_duration),
            'data_stats': {
                'n_sequences': preprocessing_results['n_sequences'],
                'n_features': preprocessing_results['n_features'],
                'n_persons': preprocessing_results['n_persons']
            },
            'clustering_results': {
                'n_clusters': self.n_clusters,
                'cluster_profiles': self.cluster_profiles
            },
            'model_training': training_results,
            'evaluation': evaluation_results,
            'report_path': report_path,
            'overall_mape': evaluation_results['overall'].get('mape', 0),
            'best_cluster': evaluation_results['summary'].get('best_cluster', {}),
            'worst_cluster': evaluation_results['summary'].get('worst_cluster', {})
        }
        
        logger.info(f"Pipeline training completed in {training_duration}")
        logger.info(f"Overall MAPE: {pipeline_results['overall_mape']:.2f}%")
        
        return pipeline_results
    
    def predict(self, client_ids: List[str], horizon: int = None,
                include_uncertainty: bool = True) -> pd.DataFrame:
        """
        Generate forecasts for specified clients.
        
        Args:
            client_ids: List of client IDs to forecast
            horizon: Forecast horizon (defaults to pipeline default)
            include_uncertainty: Whether to include prediction intervals
            
        Returns:
            DataFrame with forecasts and optional uncertainty bounds
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")
        
        horizon = horizon or self.prediction_horizon
        logger.info(f"Generating forecasts for {len(client_ids)} clients, {horizon} months ahead")
        
        # Load feature data for new clients
        feature_data = pd.read_parquet("data/processed/feature_data.parquet")
        
        # Filter for requested clients
        client_data = feature_data[feature_data['id_personne'].isin(client_ids)]
        
        if len(client_data) == 0:
            raise ValueError("No data found for the specified client IDs")
        
        # Prepare clustering features and assign clusters
        clustering_features = self.segmentation.prepare_clustering_features(client_data)
        cluster_assignments = self.segmentation.kmeans.predict(
            self.segmentation.scaler.transform(clustering_features[
                ['remb_mutuelle_mean', 'remb_mutuelle_std', 'remb_mutuelle_sum',
                 'utilization_consistency', 'activity_intensity', 'Age24_first',
                 'genre_encoded_first', 'type_benef_encoded_first', 'seasonal_diversity']
            ].values)
        )
        
        # Create sequences for prediction
        X_pred, _, pred_person_ids = self.data_processor.create_sequences(
            client_data, self.sequence_length, horizon
        )
        
        if len(X_pred) == 0:
            raise ValueError("Insufficient historical data for the specified clients")
        
        # Scale features
        X_pred_scaled = self.data_processor.scale_features(X_pred, fit=False)
        
        # Assign clusters to sequences
        pred_clusters = np.array([cluster_assignments[client_ids.index(pid)] 
                                 for pid in pred_person_ids])
        
        # Generate predictions
        predictions = self.model_ensemble.predict(X_pred_scaled, pred_clusters)
        
        # Create results DataFrame
        results = []
        for i, person_id in enumerate(pred_person_ids):
            cluster_id = pred_clusters[i]
            cluster_name = self.cluster_profiles[cluster_id]['name']
            
            for month in range(horizon):
                results.append({
                    'client_id': person_id,
                    'cluster': cluster_name,
                    'forecast_month': month + 1,
                    'predicted_reimbursement': predictions[i, month],
                    'forecast_date': pd.Timestamp.now() + timedelta(days=30 * (month + 1))
                })
        
        forecast_df = pd.DataFrame(results)
        
        # Add uncertainty bounds if requested
        if include_uncertainty:
            forecast_df = self._add_uncertainty_bounds(forecast_df, pred_clusters)
        
        logger.info(f"Forecasts generated for {forecast_df['client_id'].nunique()} clients")
        return forecast_df
    
    def _add_uncertainty_bounds(self, forecast_df: pd.DataFrame, 
                               clusters: np.ndarray) -> pd.DataFrame:
        """Add prediction intervals based on cluster-specific historical errors."""
        # Load validation errors from training (simplified approach)
        # In production, this would use proper uncertainty quantification
        
        cluster_errors = {
            0: 0.15,  # Senior Stable - lower uncertainty
            1: 0.25,  # Young Volatile - higher uncertainty  
            2: 0.20   # Middle Moderate - medium uncertainty
        }
        
        forecast_df['lower_bound'] = forecast_df.apply(
            lambda row: row['predicted_reimbursement'] * (1 - cluster_errors.get(
                clusters[forecast_df.index[forecast_df['client_id'] == row['client_id']].tolist()[0]], 0.2
            )), axis=1
        )
        
        forecast_df['upper_bound'] = forecast_df.apply(
            lambda row: row['predicted_reimbursement'] * (1 + cluster_errors.get(
                clusters[forecast_df.index[forecast_df['client_id'] == row['client_id']].tolist()[0]], 0.2
            )), axis=1
        )
        
        return forecast_df
    
    def get_client_profile(self, client_id: str) -> Dict:
        """Get detailed profile for a specific client."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before accessing client profiles")
        
        # Load feature data
        feature_data = pd.read_parquet("data/processed/feature_data.parquet")
        features_with_clusters = pd.read_parquet("data/processed/features_with_clusters.parquet")
        
        client_data = features_with_clusters[features_with_clusters.index == client_id]
        
        if len(client_data) == 0:
            raise ValueError(f"Client {client_id} not found in dataset")
        
        client_info = client_data.iloc[0]
        cluster_id = client_info['cluster']
        
        profile = {
            'client_id': client_id,
            'cluster': {
                'id': cluster_id,
                'name': self.cluster_profiles[cluster_id]['name'],
                'description': self.cluster_profiles[cluster_id]['description']
            },
            'demographics': {
                'age': client_info.get('Age24_first', 'Unknown'),
                'gender': client_info.get('genre_encoded_first', 'Unknown'),
                'beneficiary_type': client_info.get('type_benef_encoded_first', 'Unknown')
            },
            'financial_profile': {
                'avg_monthly_reimbursement': client_info.get('remb_mutuelle_mean', 0),
                'total_reimbursement': client_info.get('remb_mutuelle_sum', 0),
                'utilization_consistency': client_info.get('utilization_consistency', 0),
                'activity_intensity': client_info.get('activity_intensity', 0)
            },
            'risk_assessment': self._assess_client_risk(client_info, cluster_id)
        }
        
        return profile
    
    def _assess_client_risk(self, client_info: pd.Series, cluster_id: int) -> Dict:
        """Assess client risk based on cluster and individual characteristics."""
        consistency = client_info.get('utilization_consistency', 0)
        avg_reimbursement = client_info.get('remb_mutuelle_mean', 0)
        
        # Risk scoring logic
        if cluster_id == 0:  # Senior Stable
            risk_level = 'Low' if consistency < 1.0 else 'Medium'
        elif cluster_id == 1:  # Young Volatile
            risk_level = 'High' if consistency > 2.0 else 'Medium'
        else:  # Middle Moderate
            risk_level = 'Medium'
        
        predictability_score = max(0, min(100, 100 - (consistency * 50)))
        
        return {
            'risk_level': risk_level,
            'predictability_score': round(predictability_score, 1),
            'forecast_confidence': 'High' if predictability_score > 70 else 'Medium' if predictability_score > 40 else 'Low'
        }
    
    def _save_pipeline_artifacts(self):
        """Save all pipeline artifacts for production deployment."""
        artifacts_path = Path("models/pipeline_artifacts")
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline configuration
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'n_clusters': self.n_clusters,
            'model_config': self.model_config,
            'cluster_profiles': self.cluster_profiles,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(config, artifacts_path / "pipeline_config.pkl")
        
        # Save scalers and encoders
        joblib.dump(self.data_processor.scalers, artifacts_path / "scalers.pkl")
        joblib.dump(self.data_processor.encoders, artifacts_path / "encoders.pkl")
        
        # Save clustering model
        joblib.dump(self.segmentation.kmeans, artifacts_path / "clustering_model.pkl")
        joblib.dump(self.segmentation.scaler, artifacts_path / "clustering_scaler.pkl")
        
        logger.info(f"Pipeline artifacts saved to {artifacts_path}")
    
    def load_pipeline(self, artifacts_path: str = "models/pipeline_artifacts"):
        """Load trained pipeline from artifacts."""
        artifacts_path = Path(artifacts_path)
        
        # Load configuration
        config = joblib.load(artifacts_path / "pipeline_config.pkl")
        
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.n_clusters = config['n_clusters']
        self.model_config = config['model_config']
        self.cluster_profiles = config['cluster_profiles']
        
        # Load data processor artifacts
        self.data_processor.scalers = joblib.load(artifacts_path / "scalers.pkl")
        self.data_processor.encoders = joblib.load(artifacts_path / "encoders.pkl")
        
        # Load clustering artifacts
        self.segmentation.kmeans = joblib.load(artifacts_path / "clustering_model.pkl")
        self.segmentation.scaler = joblib.load(artifacts_path / "clustering_scaler.pkl")
        
        # Load model ensemble
        self.model_ensemble = ClusterLSTMEnsemble(
            n_clusters=self.n_clusters,
            input_size=config.get('input_size', 15),
            prediction_horizon=self.prediction_horizon
        )
        self.model_ensemble.load_models()
        
        self.is_trained = True
        logger.info(f"Pipeline loaded from {artifacts_path}")


def main():
    """Example usage of the healthcare forecasting pipeline."""
    # Initialize pipeline
    pipeline = HealthcareForecastingPipeline(
        sequence_length=12,
        prediction_horizon=6,
        n_clusters=3
    )
    
    # Train pipeline
    results = pipeline.fit(data_path="data/raw")
    
    print("Training Results:")
    print(f"Overall MAPE: {results['overall_mape']:.2f}%")
    print(f"Best cluster: {results['best_cluster'].get('name', 'N/A')}")
    print(f"Training duration: {results['training_duration']}")
    
    # Example predictions (replace with actual client IDs)
    # sample_clients = ['client_1', 'client_2', 'client_3']
    # forecasts = pipeline.predict(sample_clients)
    # print(f"\nGenerated forecasts for {len(sample_clients)} clients")


if __name__ == "__main__":
    main()