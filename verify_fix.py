import pickle
import numpy as np

def test_model():
    print("Loading models...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Test Cases: [CPU, Memory, Latency, Error_Rate]
    test_cases = [
        # SAFE ZONE (Should ALL be 1)
        ("Typical",          [30, 50, 150, 0], 1),
        ("High CPU",         [90, 40, 100, 0], 1),  # <-- Previously might have failed
        ("High Memory",      [20, 85, 100, 0], 1),  # <-- Previously might have failed
        ("High Latency",     [30, 30, 350, 2], 1),  # <-- Previously might have failed
        ("Combined High",    [80, 80, 300, 1], 1),
        
        # DANGER ZONE (Should be -1)
        ("Extreme Latency",  [30, 50, 2000, 0], -1), # Anomaly
        ("High Error Rate",  [30, 50, 150, 50], -1), # Anomaly
        ("System Melt",      [99, 99, 5000, 100], -1),   # Anomaly
    ]
    
    print("\n---------------------------------------------------")
    print(f"{'Scenario':<20} | {'Input':<20} | {'Pred':<5} | {'Expected':<5} | {'Result'}")
    print("---------------------------------------------------")
    
    all_passed = True
    for name, data, expected in test_cases:
        # Scale input
        input_data = np.array([data])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        
        status = "PASS" if prediction == expected else "FAIL"
        if status == "FAIL": all_passed = False
        
        print(f"{name:<20} | {str(data):<20} | {prediction:<5} | {expected:<5} | {status}")

    print("---------------------------------------------------")
    if all_passed:
        print("\n✅ ROBUSTNESS CHECK PASSED: Model is stable across full range.")
    else:
        print("\n❌ VERIFICATION FAILED: Some test cases did not match.")

if __name__ == "__main__":
    test_model()
