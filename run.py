
import os
import subprocess
import sys

def main():
    print("Starting AI Monitor Dashboard...")
    try:
        # Run streamlit as a subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    