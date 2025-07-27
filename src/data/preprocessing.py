import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDataProcessor:
    """
    Professional healthcare reimbursement data preprocessing pipeline.
    Handles feature engineering, cleaning, and time series preparation.
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.scalers = {}
        self.encoders = {}
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine all yearly data files."""
        logger.info("Loading raw healthcare data...")
        
        dfs = []
        for year in [22, 23, 24]:
            file_path = self.data_path / f"base_ano{year}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['year'] = 2000 + year
                dfs.append(df)
                logger.info(f"Loaded {len(df):,} records from {file_path.name}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df):,} total records")
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        logger.info("Cleaning data...")
        initial_size = len(df)
        
        # Remove records with zero or negative reimbursements and missing key fields
        df = df[
            (df['remb_mutuelle'] >= 0) & 
            (df['frais_reels'] >= 0) &
            (df['nb_acte'] >= 0) &
            (df['Mois_soins'].notna()) &
            (df['id_personne'].notna())
        ].copy()
        
        # Handle missing values
        df['code_postal'] = df['code_postal'].fillna(99999)
        df['Age24'] = df['Age24'].fillna(df['Age24'].median())
        
        # Create date features
        df['month'] = df['Mois_soins']
        df['date'] = pd.to_datetime({'year': df['year'], 'month': df['Mois_soins'], 'day': 1})
        df = df.sort_values(['id_personne', 'date'])
        
        logger.info(f"Cleaned data: {len(df):,} records ({initial_size - len(df):,} removed)")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for modeling."""
        logger.info("Engineering features...")
        
        # Basic financial features
        df['cost_ratio'] = np.where(df['frais_reels'] > 0, 
                                   df['remb_mutuelle'] / df['frais_reels'], 0)
        df['avg_cost_per_act'] = np.where(df['nb_acte'] > 0,
                                         df['frais_reels'] / df['nb_acte'], 0)
        
        # Time-based features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_year_end'] = (df['month'].isin([11, 12])).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['genre', 'type_benef', 'type_cont', 'tranche_age23']
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].fillna('Unknown'))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].fillna('Unknown'))
        
        # Aggregate person-level features
        person_stats = df.groupby('id_personne').agg({
            'remb_mutuelle': ['mean', 'std', 'sum'],
            'frais_reels': ['mean', 'std', 'sum'],
            'nb_acte': ['mean', 'sum'],
            'cost_ratio': 'mean',
            'Age24': 'first'
        }).round(2)
        
        person_stats.columns = ['_'.join(col).strip() for col in person_stats.columns]
        person_stats = person_stats.add_prefix('person_')
        
        df = df.merge(person_stats, left_on='id_personne', right_index=True)
        
        logger.info(f"Feature engineering complete: {df.shape[1]} features")
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 12, 
                        prediction_horizon: int = 6) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create time series sequences for LSTM training."""
        logger.info(f"Creating sequences (length={sequence_length}, horizon={prediction_horizon})...")
        
        # Key features for modeling
        feature_cols = [
            'remb_mutuelle', 'frais_reels', 'nb_acte', 'cost_ratio',
            'avg_cost_per_act', 'month', 'quarter', 'is_year_end',
            'genre_encoded', 'type_benef_encoded', 'Age24',
            'person_remb_mutuelle_mean', 'person_remb_mutuelle_std',
            'person_frais_reels_mean', 'person_cost_ratio_mean'
        ]
        
        X_sequences = []
        y_sequences = []
        person_ids = []
        
        for person_id in df['id_personne'].unique():
            person_data = df[df['id_personne'] == person_id].sort_values('date')
            
            if len(person_data) >= sequence_length + prediction_horizon:
                for i in range(len(person_data) - sequence_length - prediction_horizon + 1):
                    # Input sequence
                    X_seq = person_data.iloc[i:i+sequence_length][feature_cols].values
                    
                    # Target sequence (reimbursement amounts)
                    y_seq = person_data.iloc[i+sequence_length:i+sequence_length+prediction_horizon]['remb_mutuelle'].values
                    
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
                    person_ids.append(person_id)
        
        logger.info(f"Created {len(X_sequences):,} sequences from {df['id_personne'].nunique():,} unique persons")
        return np.array(X_sequences), np.array(y_sequences), person_ids
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale numerical features for neural network training."""
        if fit:
            self.scalers['features'] = StandardScaler()
            # Reshape for scaling
            original_shape = X.shape
            X_scaled = self.scalers['features'].fit_transform(X.reshape(-1, X.shape[-1]))
            return X_scaled.reshape(original_shape)
        else:
            original_shape = X.shape
            X_scaled = self.scalers['features'].transform(X.reshape(-1, X.shape[-1]))
            return X_scaled.reshape(original_shape)
    
    def process_pipeline(self, sequence_length: int = 12, prediction_horizon: int = 6) -> Dict:
        """Complete preprocessing pipeline."""
        logger.info("Starting full preprocessing pipeline...")
        
        # Load and process data
        raw_data = self.load_raw_data()
        clean_data = self.clean_data(raw_data)
        feature_data = self.engineer_features(clean_data)
        
        # Create sequences
        X, y, person_ids = self.create_sequences(feature_data, sequence_length, prediction_horizon)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Save processed data
        output_path = Path("data/processed")
        output_path.mkdir(exist_ok=True)
        
        np.save(output_path / "X_sequences.npy", X_scaled)
        np.save(output_path / "y_sequences.npy", y)
        
        with open(output_path / "person_ids.txt", "w") as f:
            for pid in person_ids:
                f.write(f"{pid}\n")
        
        # Save feature data for clustering
        feature_data.to_parquet(output_path / "feature_data.parquet")
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return {
            'X': X_scaled,
            'y': y,
            'person_ids': person_ids,
            'feature_data': feature_data,
            'n_sequences': len(X_scaled),
            'n_features': X_scaled.shape[-1],
            'n_persons': len(set(person_ids))
        }


if __name__ == "__main__":
    processor = HealthcareDataProcessor()
    results = processor.process_pipeline()
    print(f"Processing complete: {results['n_sequences']:,} sequences, {results['n_features']} features")