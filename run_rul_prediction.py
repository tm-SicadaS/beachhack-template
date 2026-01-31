"""
Quick Start Script for NASA C-MAPSS RUL Prediction
Run this script to train your failure prediction model
"""

# Import the predictor
from nasa_cmapss_rul_predictor import NASACMAPSSRULPredictor
import os
import glob
import sys
import getpass

# Initialize the predictor
predictor = NASACMAPSSRULPredictor()

print("="*70)
print("QUICK START: NASA C-MAPSS RUL PREDICTION")
print("="*70)

# OPTION 1: Load from Hugging Face cache (automatic if you've already downloaded)
print("\n🔄 Attempting to load from Hugging Face cache...")
loaded = False
try:
    predictor.load_from_huggingface_cache()
    print("✅ Data loaded from cache successfully!")
    loaded = True
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n📝 Attempting alternative automatic loading (searching for parquet/csv files)...")

    # Search current directory for parquet or csv files
    cwd = os.getcwd()
    parquet_candidates = glob.glob(os.path.join(cwd, '**', '*.parquet'), recursive=True)
    csv_candidates = glob.glob(os.path.join(cwd, '**', '*.csv'), recursive=True)

    # Try parquet first
    for p in parquet_candidates:
        try:
            predictor.load_from_parquet(p)
            print(f"✅ Data loaded from parquet: {p}")
            loaded = True
            break
        except Exception as e2:
            # Keep trying other files
            continue

    # Try CSV files if parquet didn't work
    if not loaded:
        for p in csv_candidates:
            try:
                predictor.load_from_csv(p)
                print(f"✅ Data loaded from CSV: {p}")
                loaded = True
                break
            except Exception as e2:
                continue

    # Check typical Hugging Face cache location under the user's home directory
    if not loaded:
        home = os.path.expanduser('~')
        cache_dir = os.path.join(home, '.cache', 'huggingface', 'datasets')
        if os.path.isdir(cache_dir):
            parquet_candidates = glob.glob(os.path.join(cache_dir, '**', '*.parquet'), recursive=True)
            csv_candidates = glob.glob(os.path.join(cache_dir, '**', '*.csv'), recursive=True)

            for p in parquet_candidates:
                try:
                    predictor.load_from_parquet(p)
                    print(f"✅ Data loaded from cache parquet: {p}")
                    loaded = True
                    break
                except Exception:
                    continue

            if not loaded:
                for p in csv_candidates:
                    try:
                        predictor.load_from_csv(p)
                        print(f"✅ Data loaded from cache CSV: {p}")
                        loaded = True
                        break
                    except Exception:
                        continue

    if not loaded:
        print("\n⚠️  Could not auto-load data from parquet/csv in current folder or cache.")
        print("You can:")
        print("1. Load from parquet file:")
        print("   predictor.load_from_parquet('/path/to/cache/file.parquet')")
        print("\n2. Load from CSV:")
        print("   predictor.load_from_csv('/path/to/data.csv')")
        print("\n3. Your cache location is typically:")
        user = getpass.getuser()
        print(f"   Windows: C:\\Users\\{user}\\.cache\\huggingface\\datasets")
        print("   Linux/Mac: ~/.cache/huggingface/datasets")
        sys.exit(1)

# If loaded is True we can continue, otherwise the script will have exited above

# Explore the data
print("\n🔍 Exploring data...")
predictor.explore_data()

# Calculate RUL (Remaining Useful Life)
print("\n🎯 Calculating RUL...")
predictor.calculate_rul()

# Prepare features
print("\n⚙️ Preparing features...")
X, y = predictor.prepare_features()

# Split and scale data
print("\n📊 Splitting and scaling data...")
X_train, X_test, y_train, y_test = predictor.split_and_scale_data(X, y, test_size=0.2)

# Train multiple models and compare
print("\n🤖 Training models (this may take a few minutes)...")
results = predictor.train_multiple_models()

# Evaluate all models
print("\n📈 Evaluating models...")
comparison = predictor.evaluate_and_compare()

# Analyze feature importance
print("\n🔍 Analyzing feature importance...")
predictor.plot_feature_importance(top_n=15)

# Generate prediction plots
print("\n📊 Generating visualizations...")
predictor.plot_predictions()

# Save the best model
print("\n💾 Saving the best model...")
predictor.save_model('nasa_rul_failure_prediction_model.pkl')

print("\n" + "="*70)
print("✅ SUCCESS! Your Pre-Incident Failure Detection Model is Ready!")
print("="*70)
print("\n📂 Output Files Generated:")
print("  1. rul_model_comparison.csv")
print("  2. feature_importance_rul.png")
print("  3. rul_prediction_analysis.png")
print("  4. nasa_rul_failure_prediction_model.pkl")
print("\n🎯 Best Model:", predictor.best_model_name)
print(f"📊 Test RMSE: {predictor.results[predictor.best_model_name]['test_rmse']:.2f} cycles")
print(f"📊 Test R²: {predictor.results[predictor.best_model_name]['test_r2']:.4f}")
print("\n💡 You can now use this model to predict engine failures before they occur!")
