#!/usr/bin/env python3
"""
Automated ML Fraud Detection Pipeline

Runs: Feature Aggregation ? ML Fraud Detection (single-split or k-fold)
- Date-stamped directories
- Direct report instantiation (no GUI dependency)
- Multiple analysis configurations
- Configurable feature generation and batch sizes
- K-fold cross-validation support for overfitting assessment

Usage:
    python automated_ml_pipeline.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.append('src')

from src.reports.MLFeatureAggregationReport import MLFeatureAggregationReport
from src.reports.MLFraudDetectionReport import MLFraudDetectionReport
from src.reports.MLKFoldValidator import MLKFoldValidator

# Configuration: Define your ML experiment runs
ANALYSIS_CONFIGS = [
    {
        "enabled": False,
        "tag": "test 150 kfold",  # Optional - adds custom tag to directory name
        "csv_dir": r"C:\Users\Neal Lesh\Coverage\data\fake and real",
        "min_visits": 120,
        "max_visits": 150,
        "batch_size": None,  # None = use all visits up to max_visits (not implemented yet)
        "features": "all",  # "all" or list like ["muac", "age", "gender"] (not implemented yet)
        "use_kfold": True,  # NEW: Use k-fold cross-validation instead of single split
        "n_folds": 10,  # Number of folds (only used if use_kfold=True)
        "test_split": 0.2,  # Only used if use_kfold=False
        "models": ["rf", "lr"],  # "rf" = Random Forest, "lr" = Logistic Regression
        "balance_method": "class_weight",
        "random_seed": 42
    },
    {
        "enabled": True,
        "tag": "v4 feature test 150 kfold",  # Optional - adds custom tag to directory name
        "csv_dir": r"C:\Users\Neal Lesh\Coverage\data\fake and real",
        "min_visits": 120,
        "max_visits": 150,
        "batch_size": None,  # None = use all visits up to max_visits (not implemented yet)
	"features": "all",  # <-- For aggregation (always "all")
        "features_for_ml": ["pct_female",
        "age_imbalance_6month", 
        "age_imbalance_12month",
        "muac_features_passed",
        "muac_peak_concentration",
        "muac_increasing_to_peak",
        "muac_decreasing_from_peak",
        "muac_no_skipped_bins",
        "muac_no_plateau",
        "muac_bins_with_data",
        "muac_bins_sufficient",
        "muac_peak_reasonable",
        "muac_completion_rate"],
        "use_kfold": True,  # NEW: Use k-fold cross-validation instead of single split
        "n_folds": 10,  # Number of folds (only used if use_kfold=True)
        "test_split": 0.2,  # Only used if use_kfold=False
        "models": ["rf", "lr"],  # "rf" = Random Forest, "lr" = Logistic Regression
        "balance_method": "class_weight",
        "random_seed": 42
    }
]

class AutomatedMLPipeline:
    """Automated pipeline for ML Feature Aggregation and Fraud Detection"""
    
    def __init__(self, base_output_dir=r"C:\Users\Neal Lesh\Coverage\automated_ml_output"):
        """Initialize pipeline with base output directory"""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create today's directory
        today = datetime.now().strftime("%Y_%m_%d")
        self.today_dir = self.base_output_dir / today
        self.today_dir.mkdir(exist_ok=True)
        
        print(f"?? Working directory: {self.today_dir}")

    def run_pipeline(self, analysis_configs):
        """Run the complete pipeline for all enabled configurations"""
        print(f"\n?? Starting Automated ML Pipeline")
        
        # Filter enabled configs
        enabled_configs = [c for c in analysis_configs if c.get("enabled", True)]
        
        print(f"?? Processing {len(enabled_configs)} enabled configurations (skipping {len(analysis_configs) - len(enabled_configs)})")
        print("=" * 60)
        
        results = []
        for i, config in enumerate(enabled_configs, 1):
            config_name = self._generate_config_name(config)
            print(f"\n?? Analysis {i}/{len(enabled_configs)}: {config_name}")
            print("-" * 40)
            
            try:
                result = self._run_single_analysis(config, config_name)
                results.append({
                    'config': config,
                    'config_name': config_name,
                    'status': 'success',
                    'output_dir': result['output_dir'],
                    'files': result['files']
                })
                print(f"? Success: {len(result['files'])} files generated")
                
            except Exception as e:
                print(f"? Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'config': config,
                    'config_name': config_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        self._print_summary(results)
        return results

    def _generate_config_name(self, config):
        """Generate descriptive name from config parameters"""
        parts = []
        
        # Optional tag prefix
        tag = config.get('tag')
        if tag:
            parts.append(tag)
        
        # Visits range
        parts.append(f"v{config['min_visits']}-{config['max_visits']}")
        
        # Batch size
        if config.get('batch_size') is None:
            parts.append("nobatch")
        else:
            parts.append(f"b{config['batch_size']}")
        
        # Features
        features = config.get('features', 'all')
        if features == 'all':
            parts.append("allfeats")
        elif isinstance(features, list):
            parts.append("-".join(features))
        
        # Models
        models = config.get('models', [])
        if models:
            parts.append("-".join(models))
        
        # K-fold or single split
        if config.get('use_kfold', False):
            n_folds = config.get('n_folds', 10)
            parts.append(f"kfold{n_folds}")
        else:
            parts.append("single")
        
        # ML features (if not "all") - NEVER list individual features to avoid long paths
        features_for_ml = config.get('features_for_ml', 'all')
        if features_for_ml != 'all' and isinstance(features_for_ml, list):
            # Create short descriptor based on feature types
            has_muac = any('muac' in f.lower() for f in features_for_ml)
            has_age = any('age' in f.lower() for f in features_for_ml)
            has_gender = any('gender' in f.lower() or 'female' in f.lower() for f in features_for_ml)
            
            if has_muac and not has_age and not has_gender:
                parts.append("muaconly")
            elif has_age and not has_muac and not has_gender:
                parts.append("ageonly")
            elif has_muac and not has_age:
                parts.append("muac-gender")
            elif has_muac and has_age and not has_gender:
                parts.append("muac-age")
            else:
                parts.append(f"{len(features_for_ml)}feats")
        
        return "_".join(parts)

    def _run_single_analysis(self, config, config_name):
        """Run Feature Aggregation ? Fraud Detection for a single configuration"""
        
        # Create analysis directory
        analysis_dir = self.today_dir / config_name
        analysis_dir.mkdir(exist_ok=True)
        
        print(f"?? Analysis directory: {analysis_dir.name}")
        
        # Validate CSV directory
        csv_dir = Path(config['csv_dir'])
        if not csv_dir.exists():
            raise ValueError(f"CSV directory not found: {csv_dir}")
        
        # Step 1: Feature Aggregation
        print("?? Step 1: Aggregating FLW features...")
        features_dir = analysis_dir / "01_features"
        features_dir.mkdir(exist_ok=True)
        
        def log_func(message):
            print(f"    {message}")
        
        aggregator = MLFeatureAggregationReport.create_for_automation(
            output_dir=str(features_dir),
            csv_dir=str(csv_dir),
            min_visits=config['min_visits'],
            max_visits=config['max_visits'],
            batch_size=config.get('batch_size'),
            features=config.get('features', 'all')
        )
        
        aggregator.log = log_func
        
        try:
            feature_files = aggregator.generate()
            print(f"  ? Feature aggregation complete: {len(feature_files)} files")
        except Exception as e:
            raise RuntimeError(f"Feature aggregation failed: {str(e)}")
        
        # Find the aggregated features CSV
        features_csv = None
        for f in feature_files:
            if 'ml_features_' in str(f) and str(f).endswith('.csv') and 'summary' not in str(f):
                features_csv = f
                break
        
        if not features_csv:
            raise RuntimeError("Could not find aggregated features CSV")
        
        print(f"  ?? Features CSV: {Path(features_csv).name}")
        
        # Step 2: ML Fraud Detection (single-split or k-fold)
        use_kfold = config.get('use_kfold', False)
        
        if use_kfold:
            print("?? Step 2: Running k-fold cross-validation...")
            detection_dir = analysis_dir / "02_kfold_validation"
        else:
            print("?? Step 2: Training ML fraud detection models (single split)...")
            detection_dir = analysis_dir / "02_detection"
        
        detection_dir.mkdir(exist_ok=True)
        
        # Load features
        features_df = pd.read_csv(features_csv)
        
        if use_kfold:
            # Use k-fold cross-validation
            n_folds = config.get('n_folds', 10)
            features_for_ml = config.get('features_for_ml', 'all')
            
            validator = MLKFoldValidator.create_for_automation(
                df=features_df,
                output_dir=str(detection_dir),
                n_folds=n_folds,
                include_rf='rf' in config.get('models', []),
                include_lr='lr' in config.get('models', []),
                balance_method=config.get('balance_method', 'class_weight'),
                random_state=config.get('random_seed', 42),
                features_for_ml=features_for_ml
            )
            
            validator.log = log_func
            
            try:
                detection_files = validator.run_kfold_validation()
                print(f"  ? K-fold validation complete ({n_folds} folds): {len(detection_files)} files")
            except Exception as e:
                raise RuntimeError(f"K-fold validation failed: {str(e)}")
        
        else:
            # Use traditional single train/test split
            detector = MLFraudDetectionReport.create_for_automation(
                df=features_df,
                output_dir=str(detection_dir),
                test_split=config.get('test_split', 0.2),
                include_rf='rf' in config.get('models', []),
                include_lr='lr' in config.get('models', []),
                balance_method=config.get('balance_method', 'class_weight'),
                random_state=config.get('random_seed', 42),
                use_date_subdir=False  # Skip date subdirectory - pipeline already creates dated structure
            )
            
            detector.log = log_func
            
            try:
                detection_files = detector.generate()
                print(f"  ? ML fraud detection complete: {len(detection_files)} files")
            except Exception as e:
                raise RuntimeError(f"ML fraud detection failed: {str(e)}")
        
        all_files = feature_files + detection_files
        
        return {
            'output_dir': str(analysis_dir),
            'files': all_files,
            'feature_files': feature_files,
            'detection_files': detection_files
        }

    def _print_summary(self, results):
        """Print pipeline execution summary"""
        print(f"\n?? Pipeline Summary")
        print("=" * 60)
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"? Successful: {len(successful)}")
        print(f"? Failed: {len(failed)}")
        
        if successful:
            print(f"\n?? Output directories:")
            for result in successful:
                config_name = result['config_name']
                output_dir = Path(result['output_dir']).name
                file_count = len(result['files'])
                print(f"  {config_name}: {output_dir} ({file_count} files)")
        
        if failed:
            print(f"\n?? Failed analyses:")
            for result in failed:
                config_name = result['config_name']
                error = result['error']
                print(f"  {config_name}: {error}")


def main():
    """Main entry point"""
    print("?? Automated ML Fraud Detection Pipeline")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = AutomatedMLPipeline()
        
        # Run the pipeline
        results = pipeline.run_pipeline(ANALYSIS_CONFIGS)
        
        print(f"\n?? Pipeline completed!")
        
    except Exception as e:
        print(f"\n?? Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
