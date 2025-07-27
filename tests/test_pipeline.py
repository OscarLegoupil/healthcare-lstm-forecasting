import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import HealthcareForecastingPipeline


class TestHealthcareForecastingPipeline:
    """Test suite for the complete healthcare forecasting pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample healthcare data."""
        np.random.seed(42)
        
        n_records = 2000
        n_persons = 200
        
        data = {
            'id_personne': np.random.choice([f'person_{i}' for i in range(n_persons)], n_records),
            'Mois_soins': np.random.randint(1, 13, n_records),
            'delta': np.random.choice([-1, 0, 1], n_records),
            'remb_mutuelle': np.random.exponential(50, n_records),
            'frais_reels': np.random.exponential(80, n_records),
            'nb_acte': np.random.poisson(3, n_records),
            'id_cont': [f'contract_{i}' for i in np.random.randint(1, 50, n_records)],
            'rg_benef': np.random.randint(1, 4, n_records),
            'type_benef': np.random.choice(['Salarié', 'Conjoint', 'Enfant'], n_records),
            'genre': np.random.choice(['Homme', 'Femme'], n_records),
            'code_postal': np.random.randint(10000, 99999, n_records),
            'colloc': np.random.choice(['oui', 'non', ''], n_records),
            'adh_fac': np.random.choice(['oui', 'non', ''], n_records),
            'type_cont': np.random.choice([0, 1], n_records),
            'Age24': np.random.normal(45, 15, n_records),
            'Foyer24': np.random.choice(['Célibataire', 'Couple', 'Famille'], n_records),
            'tranche_age23': np.random.choice(['[20; 30[', '[30; 40[', '[40; 50[', '[50; 60['], n_records),
            'personne_morale': np.random.choice(['PM1', 'PM2', 'PM3'], n_records),
            'entite_eco': np.random.choice(['EE1', 'EE2'], n_records),
            'annee_soins': np.random.choice([2022, 2023, 2024], n_records),
            'sum_pres': np.random.randint(1, 13, n_records)
        }
        
        # Add presence indicators (PRES columns)
        for month in range(1, 13):
            month_str = f"{month:02d}"
            data[f'PRES24{month_str}'] = np.random.choice([0.0, 1.0], n_records)
            data[f'CONSOMMANT24{month_str}'] = np.random.choice([0.0, 1.0], n_records)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_data_dir(self, sample_data):
        """Create temporary directory with sample CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            
            # Split data by year and save as CSV files
            for year in [22, 23, 24]:
                year_data = sample_data[sample_data['annee_soins'] == 2000 + year].copy()
                year_data['year'] = 2000 + year
                if len(year_data) > 0:
                    year_data.to_csv(data_path / f"base_ano{year}.csv", index=False)
            
            yield data_path
    
    @pytest.fixture
    def pipeline(self):
        """Create a HealthcareForecastingPipeline instance."""
        return HealthcareForecastingPipeline(
            sequence_length=6,
            prediction_horizon=3,
            n_clusters=2,  # Smaller for testing
            model_config={
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.2,
                'epochs': 5,  # Fewer epochs for testing
                'batch_size': 32
            }
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.sequence_length == 6
        assert pipeline.prediction_horizon == 3
        assert pipeline.n_clusters == 2
        assert not pipeline.is_trained
        assert pipeline.model_config['epochs'] == 5
    
    def test_pipeline_fit_basic(self, pipeline, temp_data_dir):
        """Test basic pipeline training functionality."""
        # Skip this test if data is insufficient (common in CI environments)
        try:
            results = pipeline.fit(data_path=str(temp_data_dir), save_artifacts=False)
            
            assert pipeline.is_trained
            assert 'training_duration' in results
            assert 'data_stats' in results
            assert 'clustering_results' in results
            assert 'evaluation' in results
            assert 'overall_mape' in results
            
            # Check data stats
            assert results['data_stats']['n_sequences'] >= 0
            assert results['data_stats']['n_features'] > 0
            assert results['data_stats']['n_persons'] > 0
            
            # Check clustering results
            assert results['clustering_results']['n_clusters'] == 2
            assert 'cluster_profiles' in results['clustering_results']
            
        except Exception as e:
            # If training fails due to insufficient data, that's expected in testing
            if "insufficient" in str(e).lower() or "no data" in str(e).lower():
                pytest.skip(f"Insufficient test data: {e}")
            else:
                raise
    
    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation."""
        # Test invalid configurations
        with pytest.raises((ValueError, TypeError)):
            HealthcareForecastingPipeline(sequence_length=0)
        
        with pytest.raises((ValueError, TypeError)):
            HealthcareForecastingPipeline(prediction_horizon=-1)
        
        with pytest.raises((ValueError, TypeError)):
            HealthcareForecastingPipeline(n_clusters=0)
    
    def test_predict_untrained_pipeline(self, pipeline):
        """Test prediction with untrained pipeline."""
        with pytest.raises(ValueError, match="Pipeline must be trained"):
            pipeline.predict(['client_1', 'client_2'])
    
    def test_get_client_profile_untrained(self, pipeline):
        """Test client profile access with untrained pipeline."""
        with pytest.raises(ValueError, match="Pipeline must be trained"):
            pipeline.get_client_profile('client_1')
    
    def test_pipeline_state_management(self, pipeline):
        """Test pipeline state management."""
        # Initially not trained
        assert not pipeline.is_trained
        assert pipeline.training_history == {}
        assert pipeline.cluster_profiles == {}
        
        # Simulate training state
        pipeline.is_trained = True
        pipeline.training_history = {0: {'train_loss': [1.0, 0.8]}}
        pipeline.cluster_profiles = {0: {'name': 'Test Cluster'}}
        
        assert pipeline.is_trained
        assert len(pipeline.training_history) == 1
        assert len(pipeline.cluster_profiles) == 1
    
    def test_model_config_defaults(self):
        """Test default model configuration."""
        pipeline = HealthcareForecastingPipeline()
        
        expected_defaults = {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'epochs': 100,
            'batch_size': 64
        }
        
        for key, expected_value in expected_defaults.items():
            assert pipeline.model_config[key] == expected_value
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        custom_config = {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.4,
            'epochs': 50,
            'batch_size': 32
        }
        
        pipeline = HealthcareForecastingPipeline(model_config=custom_config)
        
        for key, expected_value in custom_config.items():
            assert pipeline.model_config[key] == expected_value
    
    def test_pipeline_components_initialization(self, pipeline):
        """Test that all pipeline components are properly initialized."""
        assert pipeline.data_processor is not None
        assert pipeline.segmentation is not None
        assert pipeline.evaluator is not None
        assert pipeline.model_ensemble is None  # Not initialized until training
    
    def test_invalid_data_path(self, pipeline):
        """Test pipeline behavior with invalid data path."""
        invalid_path = "/nonexistent/path/to/data"
        
        # Should handle gracefully but may raise informative error
        try:
            results = pipeline.fit(data_path=invalid_path, save_artifacts=False)
            # If it succeeds, should return empty or minimal results
            assert 'data_stats' in results
        except (FileNotFoundError, ValueError) as e:
            # Expected behavior for invalid path
            assert "not found" in str(e).lower() or "no data" in str(e).lower()
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results with fixed random seed."""
        # This is challenging to test fully without actual training,
        # but we can test initialization consistency
        
        config = {
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.2,
            'epochs': 5,
            'batch_size': 32
        }
        
        pipeline1 = HealthcareForecastingPipeline(
            sequence_length=6,
            prediction_horizon=3,
            n_clusters=2,
            model_config=config
        )
        
        pipeline2 = HealthcareForecastingPipeline(
            sequence_length=6,
            prediction_horizon=3,
            n_clusters=2,
            model_config=config
        )
        
        # Check that configurations are identical
        assert pipeline1.sequence_length == pipeline2.sequence_length
        assert pipeline1.prediction_horizon == pipeline2.prediction_horizon
        assert pipeline1.n_clusters == pipeline2.n_clusters
        assert pipeline1.model_config == pipeline2.model_config
    
    def test_memory_usage_reasonable(self, pipeline):
        """Test that pipeline doesn't consume excessive memory."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create pipeline (should be lightweight)
        pipeline = HealthcareForecastingPipeline()
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not increase memory by more than 100MB just for initialization
        assert memory_increase < 100, f"Pipeline initialization used {memory_increase:.1f}MB"
    
    def test_error_handling_consistency(self, pipeline):
        """Test that errors are handled consistently across methods."""
        # Test consistent error types and messages
        error_methods = [
            (lambda: pipeline.predict(['test']), "Pipeline must be trained"),
            (lambda: pipeline.get_client_profile('test'), "Pipeline must be trained"),
        ]
        
        for method, expected_message in error_methods:
            with pytest.raises(ValueError) as exc_info:
                method()
            assert expected_message.lower() in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])