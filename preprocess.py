"""
Data Preprocessing Script
Loads raw data, performs cleaning and feature engineering, and splits into train/test sets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import yaml

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data(input_path, output_dir, params):
    """
    Load and preprocess the dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    print("Handling missing values...")
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical variables (except target)
    print("Encoding categorical variables...")
    target_col = params['preprocess']['target_column']
    
    le_dict = {}
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    
    # Encode target variable if it's categorical
    if target_col in categorical_cols:
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])
        print(f"Target classes: {le_target.classes_}")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into train and test sets
    test_size = params['preprocess']['test_size']
    random_state = params['preprocess']['random_state']
    
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Combine features and target
    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    
    # Save preprocessed data
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training set saved to {train_path} (shape: {train_df.shape})")
    print(f"Test set saved to {test_path} (shape: {test_df.shape})")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    params = load_params()
    
    input_path = params['preprocess']['input_path']
    output_dir = params['preprocess']['output_dir']
    
    preprocess_data(input_path, output_dir, params)