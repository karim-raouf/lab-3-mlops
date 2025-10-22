"""
Hyperparameter Tuning Script
Uses MLflow nested runs to test different hyperparameter combinations
"""
from dotenv import load_dotenv 
load_dotenv()
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import os
import itertools

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def evaluate_model(model, X, y, dataset_name="test"):
    """
    Evaluate model and return metrics
    """
    y_pred = model.predict(X)
    
    metrics = {
        f'{dataset_name}_accuracy': accuracy_score(y, y_pred),
        f'{dataset_name}_precision': precision_score(y, y_pred, average='weighted', zero_division=0),
        f'{dataset_name}_recall': recall_score(y, y_pred, average='weighted', zero_division=0),
        f'{dataset_name}_f1': f1_score(y, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics

def tune_hyperparameters(train_path, test_path, model_dir, params):
    """
    Perform hyperparameter tuning using MLflow nested runs
    """
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = params['preprocess']['target_column']
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Get hyperparameter grid
    tune_params = params['tune']
    n_estimators_list = tune_params['n_estimators']
    max_depth_list = tune_params['max_depth']
    
    # Create all combinations
    param_combinations = list(itertools.product(n_estimators_list, max_depth_list))
    
    print(f"\nTesting {len(param_combinations)} hyperparameter combinations:")
    print(f"n_estimators: {n_estimators_list}")
    print(f"max_depth: {max_depth_list}")
    
    # Start parent run
    with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
        mlflow.log_param("tuning_strategy", "grid_search")
        mlflow.log_param("n_combinations", len(param_combinations))
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Test each combination in a nested run
        for i, (n_est, max_d) in enumerate(param_combinations, 1):
            print(f"\n{'='*60}")
            print(f"Combination {i}/{len(param_combinations)}")
            print(f"n_estimators={n_est}, max_depth={max_d}")
            print(f"{'='*60}")
            
            with mlflow.start_run(run_name=f"n_est={n_est}_depth={max_d}", nested=True):
                # Train model
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    random_state=params['preprocess']['random_state'],
                    n_jobs=-1
                )
                
                print("Training model...")
                model.fit(X_train, y_train)
                
                # Evaluate on train set
                train_metrics = evaluate_model(model, X_train, y_train, "train")
                
                # Evaluate on test set
                test_metrics = evaluate_model(model, X_test, y_test, "test")
                
                # Log parameters
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", max_d)
                mlflow.log_param("model_type", "RandomForestClassifier")
                
                # Log all metrics
                for metric_name, metric_value in {**train_metrics, **test_metrics}.items():
                    mlflow.log_metric(metric_name, metric_value)
                    print(f"{metric_name}: {metric_value:.4f}")
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Track best model based on test F1 score
                current_score = test_metrics['test_f1']
                if current_score > best_score:
                    best_score = current_score
                    best_params = {'n_estimators': n_est, 'max_depth': max_d}
                    best_model = model
                    print(f"★ New best model! Test F1: {best_score:.4f}")
        
        # Log best results to parent run
        print(f"\n{'='*60}")
        print("TUNING SUMMARY")
        print(f"{'='*60}")
        print(f"Best hyperparameters:")
        print(f"  n_estimators: {best_params['n_estimators']}")
        print(f"  max_depth: {best_params['max_depth']}")
        print(f"Best test F1 score: {best_score:.4f}")
        
        mlflow.log_param("best_n_estimators", best_params['n_estimators'])
        mlflow.log_param("best_max_depth", best_params['max_depth'])
        mlflow.log_metric("best_test_f1", best_score)
        
        # Evaluate best model
        best_train_metrics = evaluate_model(best_model, X_train, y_train, "best_train")
        best_test_metrics = evaluate_model(best_model, X_test, y_test, "best_test")
        
        for metric_name, metric_value in {**best_train_metrics, **best_test_metrics}.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Save best model
        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        
        import joblib
        joblib.dump(best_model, best_model_path)
        print(f"\nBest model saved to {best_model_path}")
        
        # Log best model
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Save tuning results
        results_path = os.path.join(model_dir, 'tuning_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Best Hyperparameters:\n")
            f.write(f"n_estimators: {best_params['n_estimators']}\n")
            f.write(f"max_depth: {best_params['max_depth']}\n")
            f.write(f"\nBest Test F1 Score: {best_score:.4f}\n")
        
        mlflow.log_artifact(results_path)
        
        print(f"\nHyperparameter tuning completed successfully!")
        print(f"Parent MLflow run ID: {parent_run.info.run_id}")

if __name__ == "__main__":
    params = load_params()
    
    data_dir = params['preprocess']['output_dir']
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    model_dir = params['train']['model_dir']
    
    tune_hyperparameters(train_path, test_path, model_dir, params)