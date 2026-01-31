"""
NASA C-MAPSS Turbofan Engine Remaining Useful Life (RUL) Prediction
Pre-Incident Failure Detection Model for Predictive Maintenance
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NASACMAPSSRULPredictor:
    """
    Remaining Useful Life (RUL) Prediction for NASA C-MAPSS Dataset
    Predicts when engine failure will occur based on sensor readings
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = {}
        
    def load_from_huggingface_cache(self, dataset_path=None):
        """
        Load the NASA C-MAPSS dataset from Hugging Face cache
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            # If the package is missing, try loading from common local cache locations (e.g., C:\\ .cache)
            print("'datasets' package not installed; attempting to load from local cache (C:\\.cache) instead...")
            try:
                return self.load_from_local_cache(cache_dir=os.path.join('C:\\', '.cache', 'huggingface', 'datasets'))
            except Exception:
                raise ImportError("The 'datasets' package is not installed. Install it with 'pip install datasets' or place the dataset in 'C:\\\\.cache\\huggingface\\datasets'.") from e

        print("Loading NASA C-MAPSS dataset from cache...")
        
        try:
            # Load from Hugging Face (will use cache if available)
            dataset = load_dataset("penikmatrumput/nasa-cmapss-rul", split='train')
            self.df_train = dataset.to_pandas()
            
            print(f"✅ Training data loaded: {self.df_train.shape}")
            print(f"Columns: {list(self.df_train.columns)}")
            
            return self.df_train
        except Exception as e:
            # Re-raise with more context so the calling script can handle it
            raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}") from e
    
    def load_from_parquet(self, filepath):
        """
        Load data from a parquet file directly
        """
        self.df_train = pd.read_parquet(filepath)
        print(f"✅ Data loaded from parquet: {self.df_train.shape}")
        return self.df_train
    
    def load_from_csv(self, filepath):
        """
        Load data from CSV file
        """
        self.df_train = pd.read_csv(filepath)
        print(f"✅ Data loaded from CSV: {self.df_train.shape}")
        return self.df_train

    def load_from_local_cache(self, cache_dir=None):
        """
        Load dataset from a local Hugging Face cache directory (searches parquet and CSV files).
        If cache_dir is None, this will try common locations including:
        - C:\\.cache\\huggingface\\hub (snapshots)
        - C:\\.cache\\huggingface\\datasets
        - The user's home cache (~/.cache/huggingface)
        - Current working directory
        """
        search_paths = []
        if cache_dir:
            search_paths.append(cache_dir)

        # Common cache and hub locations (include 'hub' snapshots where datasets often live)
        home = os.path.expanduser('~')
        search_paths.extend([
            os.path.join('C:\\', '.cache', 'huggingface', 'hub'),
            os.path.join(home, '.cache', 'huggingface', 'hub'),
            os.path.join('C:\\', '.cache', 'huggingface', 'datasets'),
            os.path.join(home, '.cache', 'huggingface', 'datasets'),
            os.getcwd()
        ])

        tried_files = []
        candidate_files = []
        # Collect candidate parquet and csv files across search paths
        for base in search_paths:
            if not base or not os.path.isdir(base):
                continue

            parquet_files = glob.glob(os.path.join(base, '**', '*.parquet'), recursive=True)
            csv_files = glob.glob(os.path.join(base, '**', '*.csv'), recursive=True)

            candidate_files.extend(parquet_files)
            candidate_files.extend(csv_files)

        # Deduplicate while preserving order
        seen = set()
        candidate_files = [x for x in candidate_files if not (x in seen or seen.add(x))]

        if not candidate_files:
            # No candidates found - provide helpful guidance
            raise FileNotFoundError(
                f"No parquet or CSV dataset found in cache locations: {search_paths}.\n"
                f"If you have the dataset snapshot (e.g., C:\\Users\\<USER>\\.cache\\huggingface\\hub\\datasets--<repo>\\snapshots\\<id>), "
                f"you can pass that path as cache_dir or place the parquet/csv file under one of the searched locations.")

        # Score candidates by filename/path to prefer likely C-MAPSS files
        def score_path(p):
            lp = p.lower()
            s = 0
            if 'fd' in lp or 'fd00' in lp:
                s += 5
            if 'rul' in lp:
                s += 5
            if 'train' in lp:
                s += 3
            if 'test' in lp:
                s += 1
            if 'cmapss' in lp:
                s += 3
            if 'datasets--' in lp or 'snapshots' in lp or 'penikmatrumput' in lp:
                s += 2
            return s

        candidate_files.sort(key=lambda p: score_path(p), reverse=True)

        # Try to read candidates in priority order and validate structure
        for p in candidate_files:
            try:
                if p.lower().endswith('.parquet'):
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
            except Exception as e:
                tried_files.append((p, f"read error: {e}"))
                continue

            cols = [c.lower() for c in df.columns]
            has_required = ('unit_number' in cols and 'time_in_cycles' in cols)
            non_target_cols = [c for c in cols if 'rul' not in c]
            has_features = len(non_target_cols) > 0 and len(df.columns) > 1

            if has_required or has_features:
                self.df_train = df
                print(f"✅ Loaded dataset from: {p}")
                return self.df_train
            else:
                tried_files.append((p, 'file contains only target/no feature columns'))
                continue

        # If we reached here, we found candidate files but none were suitable
        raise RuntimeError(f"Found candidate files but none matched expected dataset structure. Tried: {tried_files}")
    
    def explore_data(self):
        """
        Perform exploratory data analysis on NASA C-MAPSS data
        """
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS - NASA C-MAPSS TURBOFAN ENGINE DATA")
        print("="*70)
        
        print("\n📊 Dataset Shape:", self.df_train.shape)
        print("\n📋 Column Names:")
        print(self.df_train.columns.tolist())
        
        print("\n📈 Data Types:")
        print(self.df_train.dtypes)
        
        print("\n🔢 Statistical Summary:")
        print(self.df_train.describe())
        
        print("\n❓ Missing Values:")
        missing = self.df_train.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found! ✅")
        
        print("\n🔧 Unique Units (Engines):")
        if 'unit_number' in self.df_train.columns:
            print(f"Number of unique engines: {self.df_train['unit_number'].nunique()}")
        
        print("\n⏱️ Time Cycles:")
        if 'time_in_cycles' in self.df_train.columns:
            print(f"Max cycles: {self.df_train['time_in_cycles'].max()}")
            print(f"Min cycles: {self.df_train['time_in_cycles'].min()}")
        
        # Check for RUL column
        rul_columns = [col for col in self.df_train.columns if 'rul' in col.lower()]
        if rul_columns:
            print(f"\n🎯 Target column found: {rul_columns}")
        else:
            print("\n⚠️  No RUL column found - will need to calculate it")
        
        return self.df_train.head(10)
    
    def calculate_rul(self):
        """
        Calculate Remaining Useful Life (RUL) for each engine unit
        RUL = max_cycles_for_unit - current_cycle
        """
        print("\n" + "="*70)
        print("CALCULATING REMAINING USEFUL LIFE (RUL)")
        print("="*70)
        
        # Check if RUL already exists
        if 'RUL' in self.df_train.columns or 'rul' in self.df_train.columns:
            print("✅ RUL column already exists in dataset")
            self.target_column = 'RUL' if 'RUL' in self.df_train.columns else 'rul'
            return self.df_train
        
        # Calculate RUL for each engine unit
        print("Calculating RUL for each engine unit...")
        
        # Group by unit and find max cycle for each unit
        max_cycles = self.df_train.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycles']
        
        # Merge back to get max cycles for each row
        self.df_train = self.df_train.merge(max_cycles, on='unit_number', how='left')
        
        # Calculate RUL
        self.df_train['RUL'] = self.df_train['max_cycles'] - self.df_train['time_in_cycles']
        
        # Drop the temporary max_cycles column
        self.df_train = self.df_train.drop('max_cycles', axis=1)
        
        print(f"✅ RUL calculated successfully!")
        print(f"RUL Statistics:")
        print(self.df_train['RUL'].describe())
        
        self.target_column = 'RUL'
        
        return self.df_train
    
    def prepare_features(self):
        """
        Prepare features for modeling
        """
        print("\n" + "="*70)
        print("FEATURE PREPARATION")
        print("="*70)
        
        # Identify feature columns (exclude metadata and target)
        exclude_cols = ['unit_number', 'time_in_cycles', 'RUL', 'rul']
        self.feature_names = [col for col in self.df_train.columns if col not in exclude_cols]
        
        print(f"\n📋 Feature columns ({len(self.feature_names)}):")
        print(self.feature_names)
        
        # Separate features and target
        X = self.df_train[self.feature_names]
        y = self.df_train[self.target_column]
        
        print(f"\n✅ Features shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train/test and scale features
        """
        print("\n" + "="*70)
        print("DATA SPLITTING AND SCALING")
        print("="*70)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"\n📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        print(f"📊 Split ratio: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        # Scale features
        print("\n🔄 Scaling features using StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        print("✅ Data split and scaled successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_multiple_models(self):
        """
        Train and compare multiple regression models
        """
        print("\n" + "="*70)
        print("TRAINING MULTIPLE MODELS")
        print("="*70)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            ),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*70}")
            print(f"Training: {name}")
            print(f"{'='*70}")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
            print(f"📊 Train RMSE: {train_rmse:.2f} cycles")
            print(f"📊 Test RMSE:  {test_rmse:.2f} cycles")
            print(f"📊 Train MAE:  {train_mae:.2f} cycles")
            print(f"📊 Test MAE:   {test_mae:.2f} cycles")
            print(f"📊 Train R²:   {train_r2:.4f}")
            print(f"📊 Test R²:    {test_r2:.4f}")
        
        self.results = results
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   Test RMSE: {results[best_model_name]['test_rmse']:.2f} cycles")
        print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
        print(f"{'='*70}")
        
        return results
    
    def evaluate_and_compare(self):
        """
        Create comprehensive evaluation report
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON REPORT")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Train RMSE': f"{metrics['train_rmse']:.2f}",
                'Test RMSE': f"{metrics['test_rmse']:.2f}",
                'Train MAE': f"{metrics['train_mae']:.2f}",
                'Test MAE': f"{metrics['test_mae']:.2f}",
                'Train R²': f"{metrics['train_r2']:.4f}",
                'Test R²': f"{metrics['test_r2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Save to CSV in the user's home directory (Windows-friendly)
        home = os.path.expanduser('~')
        comp_path = os.path.join(home, 'rul_model_comparison.csv')
        comparison_df.to_csv(comp_path, index=False)
        print(f"\n💾 Model comparison saved to: {comp_path}")
        
        return comparison_df
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance for tree-based models
        """
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            print("\n" + "="*70)
            print(f"FEATURE IMPORTANCE - {self.best_model_name}")
            print("="*70)
            
            # Get feature importance
            importance = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop {top_n} Most Important Features:")
            print(feature_importance_df.head(top_n).to_string(index=False))
            
            # Create plot
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}\nNASA C-MAPSS RUL Prediction', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            home = os.path.expanduser('~')
            feat_path = os.path.join(home, 'feature_importance_rul.png')
            plt.savefig(feat_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Feature importance plot saved: {feat_path}")
            
            return feature_importance_df
    
    def plot_predictions(self):
        """
        Plot actual vs predicted RUL
        """
        print("\n" + "="*70)
        print("GENERATING PREDICTION PLOTS")
        print("="*70)
        
        # Get predictions
        y_pred = self.best_model.predict(self.X_test)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual RUL (cycles)', fontsize=10)
        axes[0, 0].set_ylabel('Predicted RUL (cycles)', fontsize=10)
        axes[0, 0].set_title('Actual vs Predicted RUL', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted RUL (cycles)', fontsize=10)
        axes[0, 1].set_ylabel('Residuals', fontsize=10)
        axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residual distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Prediction samples
        sample_indices = np.random.choice(len(y_pred), size=min(100, len(y_pred)), replace=False)
        axes[1, 1].scatter(range(len(sample_indices)), 
                          self.y_test.iloc[sample_indices], 
                          label='Actual', alpha=0.7, s=30)
        axes[1, 1].scatter(range(len(sample_indices)), 
                          y_pred[sample_indices], 
                          label='Predicted', alpha=0.7, s=30)
        axes[1, 1].set_xlabel('Sample Index', fontsize=10)
        axes[1, 1].set_ylabel('RUL (cycles)', fontsize=10)
        axes[1, 1].set_title(f'Sample Predictions (n={len(sample_indices)})', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'NASA C-MAPSS RUL Prediction Analysis\nModel: {self.best_model_name}', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        home = os.path.expanduser('~')
        analysis_path = os.path.join(home, 'rul_prediction_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        print(f"📊 Prediction analysis plot saved: {analysis_path}")
        
    def save_model(self, filename='nasa_cmapss_rul_model.pkl'):
        """
        Save the trained model and preprocessing objects
        """
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'metrics': self.results[self.best_model_name],
            'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        home = os.path.expanduser('~')
        filepath = os.path.join(home, filename)
        joblib.dump(model_package, filepath)
        print(f"💾 Model saved successfully: {filename}")
        print(f"📍 Location: {filepath}")
        
        return filepath
    
    def predict_rul(self, new_data):
        """
        Predict RUL for new sensor data
        
        Args:
            new_data: DataFrame with same features as training data
        
        Returns:
            Array of predicted RUL values
        """
        # Ensure correct columns
        new_data = new_data[self.feature_names]
        
        # Scale features
        new_data_scaled = self.scaler.transform(new_data)
        
        # Predict
        predictions = self.best_model.predict(new_data_scaled)
        
        return predictions


def main():
    """
    Main execution pipeline for NASA C-MAPSS RUL prediction
    """
    print("="*70)
    print("NASA C-MAPSS TURBOFAN ENGINE")
    print("REMAINING USEFUL LIFE (RUL) PREDICTION")
    print("Pre-Incident Failure Detection System")
    print("="*70)
    
    # Initialize predictor
    predictor = NASACMAPSSRULPredictor()
    
    # Step 1: Load data from Hugging Face cache
    print("\n🔄 STEP 1: Loading data from Hugging Face cache...")
    df = predictor.load_from_huggingface_cache()
    
    if df is None:
        print("\n⚠️  Could not load from Hugging Face.")
        print("Please provide the data file path:")
        print("Example: predictor.load_from_parquet('/path/to/data.parquet')")
        print("Or: predictor.load_from_csv('/path/to/data.csv')")
        return predictor
    
    # Step 2: Explore data
    print("\n🔍 STEP 2: Exploring data...")
    predictor.explore_data()
    
    # Step 3: Calculate RUL
    print("\n🎯 STEP 3: Calculating RUL...")
    predictor.calculate_rul()
    
    # Step 4: Prepare features
    print("\n⚙️  STEP 4: Preparing features...")
    X, y = predictor.prepare_features()
    
    # Step 5: Split and scale
    print("\n📊 STEP 5: Splitting and scaling data...")
    X_train, X_test, y_train, y_test = predictor.split_and_scale_data(X, y)
    
    # Step 6: Train models
    print("\n🤖 STEP 6: Training models...")
    results = predictor.train_multiple_models()
    
    # Step 7: Evaluate
    print("\n📈 STEP 7: Evaluating models...")
    predictor.evaluate_and_compare()
    
    # Step 8: Feature importance
    print("\n🔍 STEP 8: Analyzing feature importance...")
    predictor.plot_feature_importance()
    
    # Step 9: Plot predictions
    print("\n📊 STEP 9: Generating prediction plots...")
    predictor.plot_predictions()
    
    # Step 10: Save model
    print("\n💾 STEP 10: Saving best model...")
    predictor.save_model()
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. rul_model_comparison.csv - Model performance comparison")
    print("  2. feature_importance_rul.png - Feature importance visualization")
    print("  3. rul_prediction_analysis.png - Prediction analysis plots")
    print("  4. nasa_cmapss_rul_model.pkl - Trained model (ready for deployment)")
    print("\nYour model is now ready for pre-incident failure detection!")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
