import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import shap
from xgboost import XGBClassifier
from data_loader import LoadDataset
from feature_engineering import FeatureExtraction
from hyperparameter_tuning import HyperparameterTunning
from sklearn.metrics import (
    classification_report,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from utils import create_timestamped_filename, load_search_space, load_yaml_config
from log_generator import make_logger
from save_and_load_model import save_json, save_model
from typing import Dict, Any
import optuna
from shap import Explainer
import matplotlib.pyplot as plt
import seaborn as sns

METRIC_REGISTRY = {
    "macro_f1": lambda y_true, y_pred, y_prob: f1_score(
        y_true, y_pred, average="macro"
    ),
    "pr_auc": lambda y_true, y_pred, y_prob: average_precision_score(y_true, y_prob),
}


class ModelTrainingPipeline:
    """
    ML Model Training pipeline: Data Loading -> Preprocessing -> Tuning(if required) -> Training.
    """

    def __init__(self, data_folder: str, target_col: str, optional_params: dict = {}):
        self.data_folder = data_folder
        self.target_col = target_col

        now = datetime.now(timezone.utc)
        self.timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Resolve the root_folder
        script_dir = Path(__file__).resolve().parent
        self.project_root = script_dir.parent

        # Use optional_params dictionary to set all necessary paths
        self.model_folder = self.project_root / optional_params.get(
            "output_model_dir", "models"
        )
        self.artifact_folder = self.project_root / optional_params.get(
            "output_artifact_dir", "artifacts"
        )

        logs_folder = self.project_root / optional_params.get("output_logs_dir", "logs")
        log_file = create_timestamped_filename(
            base_name=f"{logs_folder}/model_training",
            extension="log",
            timestamp=self.timestamp,
        )

        # Create directory if missing
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.artifact_folder, exist_ok=True)
        os.makedirs(logs_folder, exist_ok=True)
        # Initialize the logger at the top level
        self.logger = make_logger("ModelTrainingPipeline", log_file)

        self.default_hyperparameters = None

    def load_and_clean_data(self, file_name: str) -> pd.DataFrame:
        """Handles Data Loading and Initial Cleaning."""
        self.logger.info("\nSTEP 1: Loading and Initial Data Cleaning")

        loader = LoadDataset(data_folder=self.data_folder, logger=self.logger)
        df = loader.load_excel(file_name)

        X, y = loader.clean_data(df, self.target_col)

        self.logger.info(
            f"[SUCCESS] Data loaded and cleaned successfully : {X.shape, y.shape}"
        )
        return X, y

    def perform_feature_engineering(self, raw_data_df: pd.DataFrame):
        """Feature Engineering with fit and transform. Also, saves the state."""
        self.logger.info("\nSTEP 2: Feature Engineering")

        fe = FeatureExtraction(logger=self.logger)
        X_processed, preprocessor_state_obj = fe.fit_transform(raw_data_df)

        # Save the fitted model
        preprocessor_state_obj.save(
            f"{self.model_folder}/preprocessor_{self.timestamp}.pkl"
        )
        self.logger.info("[SUCCESS] Preprocessor State saved successfully.")
        return X_processed, preprocessor_state_obj

    def run_workflow(
        self,
        file_name: str,
        test_file_name: str = None,
        run_tuning: bool = True,
        parameter_search_space: Dict[str, Any] = None,
        default_hyperparameters: Dict[str, float] = {},
        scoring_func: callable = None,
        optional_tuning_params: dict = {},
    ):
        """
        The main method to execute the entire ML pipeline.
        """
        if run_tuning and parameter_search_space is None:
            self.logger.error(
                "parameter_search_space MUST be provided during training."
            )
            return

        self.default_hyperparameters = default_hyperparameters

        self.n_estimators = optional_tuning_params.get("max_n_estimators", 300)

        # 1. Load Data
        X, y = self.load_and_clean_data(file_name)
        # 2. Extract Features
        X_processed, preprocessor_state_obj = self.perform_feature_engineering(X)
        y = pd.DataFrame(y)
        # 3. Hyperparameters Tuning
        best_params = {}
        best_n_estimators = 0

        if run_tuning:
            self.logger.info("Running Hyperparameter tunning pipeline")

            # Instantiate and use the Tuning Object
            tuner = HyperparameterTunning(
                features=X_processed, target=y, logger=self.logger
            )

            try:
                tuner_results = tuner.extract_best_parameters(
                    parameter_search_space=parameter_search_space,
                    n_trials=optional_tuning_params.get("n_trials", 20),
                    scoring_func=scoring_func,
                    optional_tuning_params=optional_tuning_params,
                )
                best_params = tuner_results["best_parameters"]
                best_n_estimators = tuner_results["optimal_n_estimators"]
            except Exception as e:
                self.logger.critical(f"Workflow terminated due to tuning failure: {e}")
                return

        else:
            # Logic for using existing parameters (Load from file or use defaults)
            self.logger.warning("\nUsing Existing Hyperparameters/Defaults Mode")
            best_params, best_n_estimators = self.load_or_default_params()
            self.logger.info(f"Training Hyperparameters: {best_params}")
            self.logger.info(f"Training n_estimators: {best_n_estimators}")

        # 4. Train the final model and save it
        self.logger.info("\nStep 3: Starting model training")
        try:
            xgb_model = XGBClassifier(**best_params, n_estimators=best_n_estimators)
            xgb_model.fit(X_processed, y)

            # Save the model using the utility class
            save_model(
                xgb_model, f"{self.model_folder}/xgboost_model_{self.timestamp}.pkl"
            )
            self.logger.info("ML training pipeline completed!")

        except Exception as e:
            self.logger.error(f"The final model training failed: {e}")

        # 5. Model Training Performance calculation and monitoring
        performance_metrics = self.evaluate_model(model=xgb_model, X=X_processed, y=y)
        self.logger.info(f"{performance_metrics}")

        metrics_file = (
            self.artifact_folder / f"train_performance_metrics_{self.timestamp}.json"
        )
        save_json(performance_metrics, metrics_file)
        self.logger.info(f"[SUCCESS] Train Performance metrics saved to {metrics_file}")

        # 6. Extract Feature Importance
        feature_importance = self.get_feature_importance(xgb_model, X_processed)
        importance_file = (
            self.artifact_folder / f"feature_importances_{self.timestamp}.json"
        )
        save_json(feature_importance, importance_file)
        self.logger.info(f"[SUCCESS] Feature importance saved to {importance_file}")

        # 7. Calculate test performance
        if test_file_name:
            X_test, y_test = self.load_and_clean_data(test_file_name)
            fe = FeatureExtraction(
                preprocessor_state_obj=preprocessor_state_obj, logger=self.logger
            )
            X_test_processed = fe.transform(X_test)
            y_test = pd.DataFrame(y_test)

            test_performance_metrics = self.evaluate_model(
                model=xgb_model, X=X_test_processed, y=y_test
            )
            self.logger.info(f"{test_performance_metrics}")

            metrics_file = (
                self.artifact_folder / f"test_performance_metrics_{self.timestamp}.json"
            )
            save_json(test_performance_metrics, metrics_file)
            self.logger.info(f"[SUCCESS] Performance metrics saved to {metrics_file}")
            self.explain_model(xgb_model, X_test_processed)
        else:
            self.explain_model(xgb_model, X_processed)

    def load_or_default_params(self):
        """Handles loading parameters from file or using defaults."""
        default_params = (self.default_hyperparameters, self.n_estimators)
        # print(f"[DEBUG]{self.model_folder}")
        try:
            all_files = [
                f
                for f in self.model_folder.iterdir()
                if f.name.startswith("hyperparameters_") and f.name.endswith(".json")
            ]

            # Check if any files were found
            if not all_files:
                self.logger.error(f"[ERROR] Could not read files")
                return default_params

            try:
                hyperparameters_filename = max(all_files, key=os.path.getmtime)
            except Exception as e:
                self.logger.error(
                    f"[ERROR] Could not determine the latest file by timestamp: {e}"
                )
                return default_params

            # Attempt to load latest parameters
            with open(hyperparameters_filename, "r") as f:
                loaded_data = json.load(f)
                self.logger.info(
                    "[SUCCESS] Loaded hyperparameters successfully from disk."
                )
                return (
                    loaded_data["best_parameters"],
                    loaded_data["optimal_n_estimators"],
                )
        except FileNotFoundError as e:
            self.logger.error(
                f"[ERROR] Failed to process parameters in {hyperparameters_filename}: {e}"
            )
            return default_params
        except Exception as e:
            self.logger.error(
                f"[ERROR] Failed to process parameters in {hyperparameters_filename}: {e}"
            )
            return default_params

    def evaluate_model(self, model: XGBClassifier, X: pd.DataFrame, y: pd.Series):
        """
        Evaluates the trained model on a test set and returns performance metrics.
        Saves these metrics to both logs and an artifact file.
        """
        self.logger.info("\nEvaluating Model Performance...")

        # Predict probabilities (for AUC)
        y_prob = model.predict_proba(X)[:, 1]

        # Predict classes (for F1 and Classification Report)
        y_pred = model.predict(X)

        # Calculate Metrics Dictionary
        metrics = {
            "F1_score_macro": f1_score(y, y_pred, average="macro"),
            "AUC_ROC_score": roc_auc_score(y, y_prob),
            "Classification_Report": classification_report(y, y_pred, output_dict=True),
        }

        self.logger.info(f"[SUCCESS] Evaluation complete.")
        return metrics

    def explain_model(self, model: XGBClassifier, X: pd.DataFrame):
        """Function for creating and storing shap importance plot"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)

            # Generate and save the Summary Plot
            plt.figure(figsize=(14, 8))
            shap.summary_plot(shap_values, X, show=False)

            # Define a stable path for the image artifact
            image_path = self.artifact_folder / f"shap_summary_{self.timestamp}.png"
            plt.savefig(str(image_path))
            plt.close()

            self.logger.info(f"[SUCCESS] SHAP Summary Plot saved to {image_path}")

        except Exception as e:
            self.logger.warning(f"Could not generate SHAP plot. Error: {e}")

    def get_feature_importance(self, model: XGBClassifier, X: pd.DataFrame):
        """Simple funtion for feature importance extraction"""
        self.logger.info("\n Generating Feature Importance")
        feature_importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        importance_dict = feature_importances.to_dict()
        importance_dict = {
            key: f"{value:.3f}" for key, value in importance_dict.items()
        }
        return importance_dict


def main():
    """
    Main function to run the pipeline
    """
    print("===== Machine Learning Pipeline =====")

    # 1. Load Training Configuration
    training_config = load_yaml_config("config/training.yaml")

    if not training_config:
        print("[FATAL] Failed to load core training configuration. Exiting.")
        return

    # Extracting necessary values from training config
    DATA_FOLDER = training_config.get("dataset_dir")
    TRAINING_FILE = training_config.get("train_data_file")
    TESTING_FILE = training_config.get("test_data_file")
    TARGET_COLUMN = training_config.get("target_column")
    OPTIONAL_TRAINING_PARAMS = training_config.get("optional_training_params", {})
    HYPER_TUNING_REQUIRED = training_config.get("hyper_tuning_flag")

    # 2. Load Hyperparameters Configuration
    hyperparams_config = load_yaml_config("config/hyperparmeter_tuning.yaml")

    if not hyperparams_config:
        print("[FATAL] Failed to load hyperparameter configuration. Exiting.")
        return

    # if tuning is required then read the hyperparameters config
    if HYPER_TUNING_REQUIRED:
        # Extracting necessary values from hyperparameters config
        SCORING_FUNC = hyperparams_config.get("scoring_func", "macro_f1")
        DEFAULT_HYPERPARAMETERS = hyperparams_config.get("default_params", {})
        OPTIONAL_TUNING_PARAMS = hyperparams_config.get("optional_tuning_params", {})
        # Read and reconstruct Search Space
        raw_search_space_data = hyperparams_config.get("parameter_search_space", {})
        if raw_search_space_data:
            PARAMETER_SEARCH_SPACE = load_search_space(raw_search_space_data)
        else:
            print("[WARNING] Parameter search space was empty in the YAML file.")
            PARAMETER_SEARCH_SPACE = {}
    else:
        SCORING_FUNC = "macro_f1"
        DEFAULT_HYPERPARAMETERS = hyperparams_config.get("default_params", {})
        PARAMETER_SEARCH_SPACE = {}
        OPTIONAL_TUNING_PARAMS = {}

    # parameter_search_space = {
    #     "objective": optuna.distributions.CategoricalDistribution(["binary:logistic"]),
    #     "eval_metric": optuna.distributions.CategoricalDistribution(["logloss"]),
    #     "scale_pos_weight": optuna.distributions.CategoricalDistribution(
    #         [1.7]
    #     ),  # sqrt(negatives/positives)
    #     "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
    #     "max_depth": optuna.distributions.IntDistribution(4, 10),
    #     "min_child_weight": optuna.distributions.IntDistribution(3, 8),
    #     "subsample": optuna.distributions.FloatDistribution(0.7, 1.0),
    #     "colsample_bytree": optuna.distributions.FloatDistribution(0.7, 1.0),
    #     "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 3.0, log=True),
    #     "reg_lambda": optuna.distributions.FloatDistribution(1, 3.0, log=True),
    # }

    orchestrator = ModelTrainingPipeline(
        data_folder=DATA_FOLDER,
        target_col=TARGET_COLUMN,
        optional_params=OPTIONAL_TRAINING_PARAMS,
    )
    print(training_config)
    print(hyperparams_config)

    try:
        # run_workflow handles loading data, running the tuner, and training the model.
        orchestrator.run_workflow(
            file_name=TRAINING_FILE,
            test_file_name=TESTING_FILE,
            run_tuning=HYPER_TUNING_REQUIRED,
            parameter_search_space=PARAMETER_SEARCH_SPACE,
            default_hyperparameters=DEFAULT_HYPERPARAMETERS,
            scoring_func=METRIC_REGISTRY[SCORING_FUNC],  # Lookup the function object
            optional_tuning_params=OPTIONAL_TUNING_PARAMS,
        )
    except Exception as e:
        # Instead of just printing 'e', print the full traceback information
        print("\n" + "=" * 50)
        print("[CRITICAL ERROR] ML Workflow Failed!")
        import traceback  # <-- Import traceback library

        traceback.print_exc()  # <-- Print the full stack trace!
        print("=" * 50 + "\n")
        # print(f"[Error] ML Workflow Failed\n{e}")


if __name__ == "__main__":
    main()
