import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import HealthcareDataProcessor


class TestHealthcareDataProcessor:
    """Test suite for healthcare data preprocessing pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample healthcare data for testing."""
        np.random.seed(42)
        
        # Create realistic sample data
        n_records = 1000
        n_persons = 100
        
        data = {
            'id_personne': np.random.choice([f'person_{i}' for i in range(n_persons)], n_records),
            'Mois_soins': np.random.randint(1, 13, n_records),
            'remb_mutuelle': np.random.exponential(50, n_records),
            'frais_reels': np.random.exponential(80, n_records),
            'nb_acte': np.random.poisson(3, n_records),
            'genre': np.random.choice(['Homme', 'Femme'], n_records),
            'Age24': np.random.normal(45, 15, n_records),
            'type_benef': np.random.choice(['Salarié', 'Conjoint', 'Enfant'], n_records),
            'type_cont': np.random.choice([0, 1], n_records),
            'tranche_age23': np.random.choice(['[20; 30[', '[30; 40[', '[40; 50['], n_records),
            'code_postal': np.random.randint(10000, 99999, n_records),
            'year': np.random.choice([2022, 2023, 2024], n_records)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def processor(self):
        """Create a HealthcareDataProcessor instance."""
        return HealthcareDataProcessor()
    
    @pytest.fixture
    def temp_data_dir(self, sample_data):
        """Create temporary directory with sample CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            
            # Split data by year and save as CSV files
            for year in [22, 23, 24]:
                year_data = sample_data[sample_data['year'] == 2000 + year]
                year_data.to_csv(data_path / f"base_ano{year}.csv", index=False)
            
            yield data_path
    
    def test_load_raw_data(self, processor, temp_data_dir):
        """Test loading and combining raw data files."""
        processor.data_path = temp_data_dir
        df = processor.load_raw_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'year' in df.columns
        assert set(df['year'].unique()) == {2022, 2023, 2024}
    
    def test_clean_data(self, processor, sample_data):
        """Test data cleaning functionality."""
        # Add some dirty data
        dirty_data = sample_data.copy()
        dirty_data.loc[:10, 'remb_mutuelle'] = -1  # Negative values
        dirty_data.loc[10:20, 'id_personne'] = None  # Missing IDs
        
        cleaned_data = processor.clean_data(dirty_data)
        
        assert len(cleaned_data) < len(dirty_data)  # Some records should be removed
        assert (cleaned_data['remb_mutuelle'] >= 0).all()
        assert cleaned_data['id_personne'].notna().all()
        assert 'date' in cleaned_data.columns
    
    def test_engineer_features(self, processor, sample_data):
        """Test feature engineering pipeline."""
        # Prepare data
        clean_data = processor.clean_data(sample_data)
        feature_data = processor.engineer_features(clean_data)
        
        # Check new features exist
        expected_features = [
            'cost_ratio', 'avg_cost_per_act', 'month', 'quarter', 'is_year_end',
            'genre_encoded', 'type_benef_encoded'
        ]
        
        for feature in expected_features:
            assert feature in feature_data.columns
        
        # Check person-level features
        person_features = [col for col in feature_data.columns if col.startswith('person_')]
        assert len(person_features) > 0
        
        # Check data types and ranges
        assert feature_data['cost_ratio'].between(0, 1).all() or feature_data['cost_ratio'].isna().all()
        assert feature_data['month'].between(1, 12).all()
        assert feature_data['quarter'].between(1, 4).all()
    
    def test_create_sequences(self, processor, sample_data):
        """Test time series sequence creation."""
        # Prepare data
        clean_data = processor.clean_data(sample_data)
        feature_data = processor.engineer_features(clean_data)
        
        sequence_length = 6
        prediction_horizon = 3
        
        X, y, person_ids = processor.create_sequences(
            feature_data, sequence_length, prediction_horizon
        )
        
        assert len(X) == len(y) == len(person_ids)
        assert X.shape[1] == sequence_length
        assert X.shape[2] > 0  # Should have features
        assert y.shape[1] == prediction_horizon
        assert all(isinstance(pid, str) for pid in person_ids)
    
    def test_scale_features(self, processor):
        """Test feature scaling functionality."""
        # Create sample feature array
        X = np.random.randn(100, 10, 5)  # (samples, sequence_length, features)
        
        # Fit scaling
        X_scaled = processor.scale_features(X, fit=True)
        
        assert X_scaled.shape == X.shape
        assert 'features' in processor.scalers
        
        # Test transform only
        X_new = np.random.randn(50, 10, 5)
        X_new_scaled = processor.scale_features(X_new, fit=False)
        
        assert X_new_scaled.shape == X_new.shape
    
    def test_process_pipeline_integration(self, processor, temp_data_dir):
        """Test the complete preprocessing pipeline."""
        processor.data_path = temp_data_dir
        
        results = processor.process_pipeline(sequence_length=6, prediction_horizon=3)
        
        # Check all expected outputs
        assert 'X' in results
        assert 'y' in results
        assert 'person_ids' in results
        assert 'feature_data' in results
        assert 'n_sequences' in results
        assert 'n_features' in results
        assert 'n_persons' in results
        
        # Check data consistency
        assert len(results['X']) == len(results['y']) == len(results['person_ids'])
        assert results['n_sequences'] == len(results['X'])
        assert results['n_features'] == results['X'].shape[-1]
        assert results['n_persons'] == len(set(results['person_ids']))
    
    def test_empty_data_handling(self, processor):
        """Test handling of empty or invalid data."""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            processor.clean_data(empty_df)
    
    def test_missing_files(self, processor):
        """Test handling of missing data files."""
        processor.data_path = Path("/nonexistent/path")
        
        df = processor.load_raw_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0  # Should return empty DataFrame
    
    def test_sequence_edge_cases(self, processor, sample_data):
        """Test sequence creation with edge cases."""
        # Prepare minimal data
        minimal_data = sample_data.head(10).copy()
        minimal_data['id_personne'] = 'single_person'
        
        clean_data = processor.clean_data(minimal_data)
        feature_data = processor.engineer_features(clean_data)
        
        # Test with sequence length longer than available data
        X, y, person_ids = processor.create_sequences(
            feature_data, sequence_length=20, prediction_horizon=5
        )
        
        # Should return empty arrays when insufficient data
        assert len(X) == 0
        assert len(y) == 0
        assert len(person_ids) == 0
    
    def test_data_types_consistency(self, processor, sample_data):
        """Test that data types are handled consistently."""
        clean_data = processor.clean_data(sample_data)
        feature_data = processor.engineer_features(clean_data)
        
        # Check that numerical columns are numeric
        numeric_cols = ['remb_mutuelle', 'frais_reels', 'nb_acte', 'Age24']
        for col in numeric_cols:
            if col in feature_data.columns:
                assert pd.api.types.is_numeric_dtype(feature_data[col])
        
        # Check that encoded columns are integers
        encoded_cols = [col for col in feature_data.columns if col.endswith('_encoded')]
        for col in encoded_cols:
            assert pd.api.types.is_integer_dtype(feature_data[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])