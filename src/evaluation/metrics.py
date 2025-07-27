import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingMetrics:
    """
    Comprehensive forecasting evaluation metrics for healthcare reimbursement prediction.
    Includes industry-standard metrics and healthcare-specific performance measures.
    """
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    @staticmethod
    def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error (MASE)."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Calculate naive forecast MAE on training data
        naive_forecast_errors = np.abs(np.diff(y_train))
        scale = np.mean(naive_forecast_errors)
        
        if scale == 0:
            return np.inf
        
        return np.mean(np.abs(y_true - y_pred)) / scale
    
    @staticmethod
    def prediction_interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_lower: np.ndarray, y_upper: np.ndarray) -> float:
        """Calculate prediction interval coverage probability."""
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        return coverage * 100
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (trend prediction)."""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        return np.mean(true_direction == pred_direction) * 100


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for healthcare forecasting.
    Provides detailed performance analysis by cluster and time horizon.
    """
    
    def __init__(self, cluster_names: Optional[Dict[int, str]] = None):
        self.cluster_names = cluster_names or {0: 'Senior Stable', 1: 'Young Volatile', 2: 'Middle Moderate'}
        self.metrics = ForecastingMetrics()
        
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           clusters: np.ndarray, y_train: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive evaluation of model predictions."""
        logger.info("Evaluating model predictions...")
        
        results = {
            'overall': {},
            'by_cluster': {},
            'by_horizon': {},
            'summary': {}
        }
        
        # Overall metrics
        results['overall'] = self._calculate_all_metrics(y_true, y_pred, y_train)
        
        # Metrics by cluster
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            if np.sum(mask) > 0:
                cluster_true = y_true[mask]
                cluster_pred = y_pred[mask]
                cluster_train = y_train[mask] if y_train is not None else None
                
                results['by_cluster'][cluster_id] = self._calculate_all_metrics(
                    cluster_true, cluster_pred, cluster_train
                )
                results['by_cluster'][cluster_id]['name'] = self.cluster_names.get(cluster_id, f'Cluster {cluster_id}')
                results['by_cluster'][cluster_id]['n_samples'] = np.sum(mask)
        
        # Metrics by forecast horizon
        if y_true.ndim > 1:
            for horizon in range(y_true.shape[1]):
                horizon_true = y_true[:, horizon]
                horizon_pred = y_pred[:, horizon]
                horizon_train = y_train[:, horizon] if y_train is not None and y_train.ndim > 1 else None
                
                results['by_horizon'][horizon + 1] = self._calculate_all_metrics(
                    horizon_true, horizon_pred, horizon_train
                )
        
        # Summary statistics
        results['summary'] = self._create_summary(results)
        
        logger.info("Model evaluation completed")
        return results
    
    def _calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_train: Optional[np.ndarray] = None) -> Dict:
        """Calculate all available metrics for given predictions."""
        # Flatten arrays if multidimensional
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[valid_mask]
        y_pred_clean = y_pred_flat[valid_mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid predictions'}
        
        metrics = {
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'mse': mean_squared_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'mape': self.metrics.mean_absolute_percentage_error(y_true_clean, y_pred_clean),
            'smape': self.metrics.symmetric_mean_absolute_percentage_error(y_true_clean, y_pred_clean),
            'r2': r2_score(y_true_clean, y_pred_clean),
            'directional_accuracy': self.metrics.directional_accuracy(y_true_clean, y_pred_clean),
            'n_samples': len(y_true_clean)
        }
        
        # Add MASE if training data is available
        if y_train is not None:
            y_train_flat = y_train.flatten()
            train_valid_mask = np.isfinite(y_train_flat)
            y_train_clean = y_train_flat[train_valid_mask]
            
            if len(y_train_clean) > 1:
                metrics['mase'] = self.metrics.mean_absolute_scaled_error(
                    y_true_clean, y_pred_clean, y_train_clean
                )
        
        return metrics
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create performance summary across all evaluations."""
        summary = {
            'best_cluster': None,
            'worst_cluster': None,
            'performance_variance': 0.0,
            'horizon_degradation': 0.0
        }
        
        # Find best and worst performing clusters by MAPE
        if 'by_cluster' in results and results['by_cluster']:
            cluster_mapes = {cid: metrics.get('mape', float('inf')) 
                           for cid, metrics in results['by_cluster'].items()}
            
            best_cluster_id = min(cluster_mapes, key=cluster_mapes.get)
            worst_cluster_id = max(cluster_mapes, key=cluster_mapes.get)
            
            summary['best_cluster'] = {
                'id': best_cluster_id,
                'name': self.cluster_names.get(best_cluster_id, f'Cluster {best_cluster_id}'),
                'mape': cluster_mapes[best_cluster_id]
            }
            
            summary['worst_cluster'] = {
                'id': worst_cluster_id,
                'name': self.cluster_names.get(worst_cluster_id, f'Cluster {worst_cluster_id}'),
                'mape': cluster_mapes[worst_cluster_id]
            }
            
            # Calculate performance variance
            mape_values = [mape for mape in cluster_mapes.values() if mape != float('inf')]
            if mape_values:
                summary['performance_variance'] = np.std(mape_values)
        
        # Calculate horizon degradation
        if 'by_horizon' in results and results['by_horizon']:
            horizon_mapes = [metrics.get('mape', 0) for metrics in results['by_horizon'].values()]
            if len(horizon_mapes) > 1:
                # Linear trend of MAPE increase over horizons
                horizons = list(range(1, len(horizon_mapes) + 1))
                summary['horizon_degradation'] = np.polyfit(horizons, horizon_mapes, 1)[0]
        
        return summary
    
    def create_evaluation_report(self, results: Dict, save_path: str = "results/reports") -> str:
        """Generate comprehensive evaluation report."""
        logger.info("Creating evaluation report...")
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        report_lines = []
        report_lines.append("# Healthcare Reimbursement Forecasting - Model Evaluation Report\n")
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall Performance
        if 'overall' in results:
            overall = results['overall']
            report_lines.append("## Overall Performance\n")
            report_lines.append(f"- **MAPE**: {overall.get('mape', 0):.2f}%")
            report_lines.append(f"- **RMSE**: {overall.get('rmse', 0):.2f}")
            report_lines.append(f"- **R²**: {overall.get('r2', 0):.3f}")
            report_lines.append(f"- **Directional Accuracy**: {overall.get('directional_accuracy', 0):.1f}%")
            report_lines.append(f"- **Samples**: {overall.get('n_samples', 0):,}\n")
        
        # Cluster Performance
        if 'by_cluster' in results and results['by_cluster']:
            report_lines.append("## Performance by Client Cluster\n")
            
            cluster_df = pd.DataFrame({
                cid: {
                    'Cluster Name': metrics.get('name', f'Cluster {cid}'),
                    'MAPE (%)': f"{metrics.get('mape', 0):.2f}",
                    'RMSE': f"{metrics.get('rmse', 0):.2f}",
                    'R²': f"{metrics.get('r2', 0):.3f}",
                    'Samples': f"{metrics.get('n_samples', 0):,}"
                }
                for cid, metrics in results['by_cluster'].items()
            }).T
            
            report_lines.append(cluster_df.to_markdown())
            report_lines.append("")
        
        # Horizon Analysis
        if 'by_horizon' in results and results['by_horizon']:
            report_lines.append("## Performance by Forecast Horizon\n")
            
            horizon_df = pd.DataFrame({
                f'Month {horizon}': {
                    'MAPE (%)': f"{metrics.get('mape', 0):.2f}",
                    'RMSE': f"{metrics.get('rmse', 0):.2f}",
                    'R²': f"{metrics.get('r2', 0):.3f}"
                }
                for horizon, metrics in results['by_horizon'].items()
            }).T
            
            report_lines.append(horizon_df.to_markdown())
            report_lines.append("")
        
        # Key Insights
        if 'summary' in results:
            summary = results['summary']
            report_lines.append("## Key Insights\n")
            
            if summary.get('best_cluster'):
                best = summary['best_cluster']
                report_lines.append(f"- **Best Performing Cluster**: {best['name']} (MAPE: {best['mape']:.2f}%)")
            
            if summary.get('worst_cluster'):
                worst = summary['worst_cluster']
                report_lines.append(f"- **Most Challenging Cluster**: {worst['name']} (MAPE: {worst['mape']:.2f}%)")
            
            report_lines.append(f"- **Performance Variance**: {summary.get('performance_variance', 0):.2f}% MAPE std")
            report_lines.append(f"- **Horizon Degradation**: {summary.get('horizon_degradation', 0):.2f}% MAPE per month")
        
        # Business Impact
        report_lines.append("\n## Business Impact Assessment\n")
        overall_mape = results.get('overall', {}).get('mape', 0)
        
        if overall_mape < 10:
            impact = "Excellent - Suitable for production deployment"
        elif overall_mape < 15:
            impact = "Good - Requires monitoring but production-ready"
        elif overall_mape < 20:
            impact = "Moderate - Additional validation recommended"
        else:
            impact = "Poor - Requires model improvement"
        
        report_lines.append(f"**Model Quality**: {impact}")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = Path(save_path) / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of model evaluation."""
    # Load test data (this would be real predictions in practice)
    y_true = np.random.exponential(50, (1000, 6))  # 6-month forecasts
    y_pred = y_true + np.random.normal(0, 5, y_true.shape)  # Add some prediction error
    clusters = np.random.randint(0, 3, 1000)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_predictions(y_true, y_pred, clusters)
    
    # Generate report
    report_path = evaluator.create_evaluation_report(results)
    
    print(f"Evaluation completed! Report saved to: {report_path}")
    print(f"Overall MAPE: {results['overall']['mape']:.2f}%")


if __name__ == "__main__":
    main()