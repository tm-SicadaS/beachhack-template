"""
How to Use Your Trained RUL Model for Failure Prediction
After training, use this script to make predictions on new engine data
"""

import joblib
import pandas as pd
import numpy as np

def load_trained_model(model_path='nasa_rul_failure_prediction_model.pkl'):
    """
    Load a previously trained RUL prediction model
    """
    print("Loading trained model...")
    model_package = joblib.load(model_path)
    
    print(f"✅ Model loaded: {model_package['model_name']}")
    print(f"📅 Trained on: {model_package['trained_date']}")
    print(f"📊 Test RMSE: {model_package['metrics']['test_rmse']:.2f} cycles")
    print(f"📊 Test R²: {model_package['metrics']['test_r2']:.4f}")
    
    return model_package

def predict_failure(model_package, new_sensor_data):
    """
    Predict Remaining Useful Life for new sensor readings
    
    Args:
        model_package: Loaded model package
        new_sensor_data: DataFrame with sensor readings (same features as training)
    
    Returns:
        predictions: Array of predicted RUL values (in cycles)
    """
    # Extract components
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # Ensure correct feature order
    new_sensor_data = new_sensor_data[feature_names]
    
    # Scale the data
    scaled_data = scaler.transform(new_sensor_data)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    
    return predictions

def interpret_prediction(rul_cycles, cycle_duration_hours=1.0):
    """
    Interpret RUL prediction and provide maintenance recommendations
    
    Args:
        rul_cycles: Predicted remaining useful life in cycles
        cycle_duration_hours: Duration of one operational cycle in hours
    
    Returns:
        dict with interpretation and recommendations
    """
    rul_hours = rul_cycles * cycle_duration_hours
    rul_days = rul_hours / 24
    
    interpretation = {
        'rul_cycles': rul_cycles,
        'rul_hours': rul_hours,
        'rul_days': rul_days,
        'status': '',
        'recommendation': '',
        'priority': ''
    }
    
    if rul_cycles < 20:
        interpretation['status'] = '🔴 CRITICAL - Immediate Attention Required'
        interpretation['recommendation'] = 'Schedule immediate maintenance. Engine failure imminent.'
        interpretation['priority'] = 'CRITICAL'
    elif rul_cycles < 50:
        interpretation['status'] = '🟠 WARNING - Maintenance Needed Soon'
        interpretation['recommendation'] = 'Schedule maintenance within the next few days.'
        interpretation['priority'] = 'HIGH'
    elif rul_cycles < 100:
        interpretation['status'] = '🟡 CAUTION - Monitor Closely'
        interpretation['recommendation'] = 'Plan maintenance within 1-2 weeks. Monitor continuously.'
        interpretation['priority'] = 'MEDIUM'
    else:
        interpretation['status'] = '🟢 HEALTHY - Normal Operation'
        interpretation['recommendation'] = 'Continue normal operations. Schedule routine inspection.'
        interpretation['priority'] = 'LOW'
    
    return interpretation


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("PREDICTION EXAMPLE: Engine Failure Detection")
    print("="*70)
    
    # Load the trained model
    model_package = load_trained_model('nasa_rul_failure_prediction_model.pkl')
    
    # Example: Create sample sensor readings for prediction
    # Replace this with your actual sensor data
    print("\n📊 Creating example sensor data...")
    
    # Get feature names from model
    feature_names = model_package['feature_names']
    
    # Create sample data (replace with actual sensor readings)
    sample_data = pd.DataFrame({
        feature: [np.random.random()] for feature in feature_names
    })
    
    print(f"Features: {feature_names}")
    print("\n🔍 Sample sensor readings:")
    print(sample_data)
    
    # Make prediction
    print("\n🤖 Making prediction...")
    rul_prediction = predict_failure(model_package, sample_data)
    
    print(f"\n{'='*70}")
    print(f"PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"\n📈 Predicted Remaining Useful Life (RUL): {rul_prediction[0]:.1f} cycles")
    
    # Interpret the prediction
    interpretation = interpret_prediction(rul_prediction[0])
    
    print(f"\n{interpretation['status']}")
    print(f"\n⏰ Time Remaining:")
    print(f"   - Cycles: {interpretation['rul_cycles']:.1f}")
    print(f"   - Hours: {interpretation['rul_hours']:.1f}")
    print(f"   - Days: {interpretation['rul_days']:.1f}")
    print(f"\n⚠️  Priority: {interpretation['priority']}")
    print(f"\n💡 Recommendation:")
    print(f"   {interpretation['recommendation']}")
    
    print("\n" + "="*70)
    print("BATCH PREDICTION EXAMPLE")
    print("="*70)
    
    # Example: Predict for multiple engines
    n_engines = 5
    batch_data = pd.DataFrame({
        feature: np.random.random(n_engines) for feature in feature_names
    })
    
    print(f"\n🔍 Predicting for {n_engines} engines...")
    batch_predictions = predict_failure(model_package, batch_data)
    
    print("\n📊 Results:")
    print("-" * 70)
    print(f"{'Engine':<10} {'RUL (cycles)':<15} {'Status':<30} {'Priority':<10}")
    print("-" * 70)
    
    for i, rul in enumerate(batch_predictions, 1):
        interp = interpret_prediction(rul)
        status_icon = interp['status'].split()[0]  # Get emoji
        print(f"Engine {i:<3} {rul:<15.1f} {status_icon} {interp['priority']:<20} {interp['priority']:<10}")
    
    print("-" * 70)
    
    print("\n" + "="*70)
    print("✅ Prediction Complete!")
    print("="*70)
    print("\n💡 To use with your own data:")
    print("1. Load your sensor readings into a DataFrame")
    print("2. Ensure columns match the training features")
    print("3. Call predict_failure(model_package, your_data)")
    print("4. Interpret results using interpret_prediction()")
