"""
Model Training Script
Trains a baseline model and logs metrics with MLflow
"""
from dotenv import load_dotenv 
load_dotenv()

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import yaml
import os

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model(train_path, model_dir, params):
    """
    Train a Random Forest classifier and save the model
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    target_col = params['preprocess']['target_column']
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Get model parameters
    model_params = params['train']['model_params']
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Start MLflow run
    with mlflow.start_run(run_name="baseline_training"):
        print("Training Random Forest model...")
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            random_state=params['preprocess']['random_state'],
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions on training data
        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        print(f"\nTraining Metrics:")
        print(f"Accuracy: {train_accuracy:.4f}")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall: {train_recall:.4f}")
        print(f"F1-Score: {train_f1:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", model_params['n_estimators'])
        mlflow.log_param("max_depth", model_params['max_depth'])
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10))
        
        # Save the model
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save feature importance
        feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path)
        
        print("Training completed successfully!")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    params = load_params()
    
    train_path = os.path.join(params['preprocess']['output_dir'], 'train.csv')
    model_dir = params['train']['model_dir']
    
    train_model(train_path, model_dir, params)