#!/usr/bin/env python3
"""
Quick execution script for the Network Intrusion Detection System
"""

from intrusion_detection_system import NetworkIntrusionDetector
import sys
import os

def main():
    """
    Main execution function with error handling
    """
    print("=" * 70)
    print("NETWORK INTRUSION DETECTION SYSTEM")
    print("Machine Learning-Based Anomaly Detection")
    print("=" * 70)
    
    # Check if dataset directory exists
    if not os.path.exists("Datasets"):
        print("ERROR: Datasets directory not found!")
        print("Please ensure the CICIDS2017 dataset files are in the 'Datasets' folder")
        sys.exit(1)
    
    # Check if dataset files exist
    csv_files = [f for f in os.listdir("Datasets") if f.endswith('.csv')]
    if not csv_files:
        print("ERROR: No CSV files found in Datasets directory!")
        print("Please ensure the CICIDS2017 dataset files are present")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} dataset files:")
    for file in csv_files:
        print(f"  - {file}")
    
    try:
        # Initialize and run the intrusion detection system
        detector = NetworkIntrusionDetector()
        detector.run_full_analysis()
        
        print("\nSuccess! Analysis completed successfully.")
        print("Check the generated files:")
        print("  - intrusion_detection_results_[timestamp].csv")
        print("  - model_performance_comparison.png")
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        print("Please check the dataset files and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()