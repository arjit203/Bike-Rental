import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_and_clean_data, prepare_features

def main():
    print("="*60)
    print("BIKE RENTAL PREDICTION PROJECT")
    print("="*60)
    
    # Load and clean the data
    print("\n📊 Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    #  Prepare features for machine learning
    print("\n🔧 Preparing features...")
    X, y = prepare_features(df)
    print(f"Features prepared! X shape: {X.shape}, y shape: {y.shape}")
    
if __name__ == "__main__":
    main()