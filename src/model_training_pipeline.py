import pandas as pd
import numpy as np
import pickle
import json
import os
from glob import glob
from datetime import datetime
from xgboost import XGBClassifier
from data_loader import LoadDataset
from feature_engineering import FeatureExtraction
from hyperparameter_tunning import HyperparameterTunning
from sklearn.metrics import f1_score, average_precision_score
from utils import create_timestamped_filename
from log_generator import make_logger
from load_and_save_model import save_model
from typing import Dict, Any
import optuna

METRIC_REGISTRY = {
    "macro_f1": lambda y_true, y_pred, y_prob: f1_score(y_true, y_pred, average='macro'),
    "pr_auc": lambda y_true, y_pred, y_prob: average_precision_score(y_true, y_prob)
}

class ModelTrainingPipeline:
    """
    ML Model Training pipeline: Data Loading -> Preprocessing -> Tuning(if required) -> Training.
    """

    def __init__(self, data_folder: str, target_col: str):
        self.data_folder = data_folder
        self.target_col = target_col
        now = datetime.now()
        self.timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_file = create_timestamped_filename(
            base_name="../logs/model_training",
            extension="log",
            timestamp=self.timestamp,
        )
        # Initialize the logger at the top level
        self.logger = make_logger("ModelTrainingPipeline", log_file)

    def load_and_clean_data(self, file_name: str) -> pd.DataFrame:
        """Handles Data Loading and Initial Cleaning."""
        self.logger.info("\nSTEP 1: Loading and Initial Data Cleaning")

        loader = LoadDataset(data_folder=self.data_folder)
        df = loader.load_excel(file_name)

        X, y = loader.clean_data(df, self.target_col)

        
        self.logger.info(
            f"[SUCCESS] Data loaded and cleaned successfully : {X.shape, y.shape}"
        )
        return X,y

    def perform_feature_engineering(self, raw_data_df: pd.DataFrame):
        """Feature Engineering with fit and transform. Also, saves the state."""
        self.logger.info("\nSTEP 2: Feature Engineering")

        fe = FeatureExtraction()
        X_processed, preprocessor_state_obj = fe.fit_transform(raw_data_df)

        # Save the fitted model
        preprocessor_state_obj.save(f"../models/preprocessor_{self.timestamp}.pkl")
        self.logger.info("[SUCCESS] Preprocessor State saved successfully.")
        return X_processed, preprocessor_state_obj

    def run_workflow(
        self,
        file_name: str,
        run_tuning: bool = True,
        parameter_search_space: Dict[str, Any] = None,
    ):
        """
        The main method to execute the entire ML pipeline.
        """
        if parameter_search_space is None:
            self.logger.error(
                "Parameter search space MUST be provided during training."
            )
            return

        # 1. Load Data
        X,y = self.load_and_clean_data(file_name)
        # 2. Extract Features
        X_processed, preprocessor_state_obj = self.perform_feature_engineering(
            X
        )
        y = pd.DataFrame(y)
        #3. Hyperparameters Tuning ---
        best_params: dict
        best_n_estimators: int

        if run_tuning:
            self.logger.info("Running Hyperparameter tunning pipeline")

            # Instantiate and use the Tuning Object
            tuner = HyperparameterTunning(
                features=X_processed,
                target=y
            )

            try:
                tuner_results = tuner.extract_best_parameters(
                    parameter_search_space=parameter_search_space,scoring_func=METRIC_REGISTRY["macro_f1"]
                )
                best_params = tuner_results["best_parameters"]
                best_n_estimators = tuner_results["optimal_n_estimators"]
            except Exception as e:
                self.logger.critical(f"Workflow terminated due to tuning failure: {e}")
                return

        else:
            # Logic for using existing parameters (Load from file or use defaults)
            self.logger.warning(
                "\nUsing Existing Hyperparameters/Defaults Mode"
            )
            best_params, best_n_estimators = self.load_or_default_params()

        # 4. Train the final model and save it
        self.logger.info("Step 3: Starting model training")
        try:
            xgb_model = XGBClassifier(**best_params, n_estimators=best_n_estimators, use_label_encoder=False)
            xgb_model.fit(X_processed, y)

            # Save the model using the utility class
            save_model(xgb_model, f"../models/xgboost_model_{self.timestamp}.pkl") 
            self.logger.info("ML training pipeline completed!")

        except Exception as e:
            self.logger.error(f"The final model training failed: {e}")

    def load_or_default_params(self):
        """Handles loading parameters from file or using defaults."""
        try:
            # Find latest hyperparameters fiel fro model folder
            search_path = os.path.join("../models/", "hyperparameters_*.json")
            all_files = []
            try:
                all_files = glob.glob(search_path)
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to search directory {"../models/"}: {e}")

            # Check if any files were found
            if not all_files:
                return self.load_default_params()

            try:
                hyperparameters_filename = max(all_files, key=os.path.getmtime)
            except Exception as e:
                self.logger.error(f"[ERROR] Could not determine the latest file by timestamp: {e}")
                return self.load_default_params()

            # Attempt to load latest parameters
            with open(hyperparameters_filename, "r") as f:
                loaded_data = json.load(f)
                self.logger.info(
                    "[SUCCESS] Loaded hyperparameters successfully from disk."
                )
                return (loaded_data["best_params"], loaded_data["optimal_n_estimators"])
        except FileNotFoundError:
            self.logger.error(f"[ERROR] Failed to process parameters in {hyperparameters_filename}: {e}")
            return self.load_default_params()
    
    def load_default_params(self):
        """Helper function to return default hyperparameters."""
        default_params = {
            "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, 
            "subsample": 0.8, "colsample_bytree": 0.8, 
            "reg_alpha": 1.0, "reg_lambda": 2.0, "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        self.logger.warning("[FALLBACK] Using default hyperparameter values.")
        return (default_params, 300)



def main():
    """
    Main function to run the pipeline
    """
    print("===== Machine Learning Pipeline =====")
    
    # CONFIGURATION SECTION
    DATA_FOLDER = "../data/"
    TRAINING_FILE = "train_dataset.xlsx"
    TARGET_COLUMN = "Churn"

    # Define required parameters for tuning/defaulting
    parameter_search_space = {
        "objective": optuna.distributions.CategoricalDistribution(["binary:logistic"]), 
        "eval_metric": optuna.distributions.CategoricalDistribution(["logloss"]),
        "scale_pos_weight":optuna.distributions.CategoricalDistribution([1.7]), # sqrt(negatives/positives)
        "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
        "max_depth": optuna.distributions.IntDistribution(4, 10),
        "min_child_weight": optuna.distributions.IntDistribution(3, 8),
        "subsample": optuna.distributions.FloatDistribution(0.7, 1.0),
        "colsample_bytree": optuna.distributions.FloatDistribution(0.7, 1.0),
        "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 3.0, log=True),
        "reg_lambda": optuna.distributions.FloatDistribution(1, 3.0, log=True),
    }


    orchestrator = ModelTrainingPipeline(data_folder=DATA_FOLDER, target_col=TARGET_COLUMN)

    try:
        # run_workflow handles loading data, running the tuner, and training the model.
        orchestrator.run_workflow(
            file_name=TRAINING_FILE, 
            run_tuning=True,
            parameter_search_space=parameter_search_space
        )
    except Exception as e:
        print(f"[Error] ML Workflow Failed\n{e}")

if __name__ == "__main__":
    main()
