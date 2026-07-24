# Pipeline for hyperparameter tuning using optuna
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
import json
import logging
from typing import Dict, Any, Callable
from log_generator import make_logger
from utils import create_timestamped_filename
from pathlib import Path 
class HyperparameterTunning:
    def __init__(self, features:pd.DataFrame, target:pd.DataFrame,logger: Callable=None):
        "Initializes the dataset"
        self.X = features
        self.y = target
        # Resolve root folder
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        if(logger):
            self.logger = logger
        else:

            log_file = f"{project_root}/logs/hyperparameter_tunning.log"
            self.logger = make_logger("HyperparameterTunning",log_file)

        self.model_folder = project_root / "models"



    def create_objective(self, n_splits:int, parameter_search_space:Dict[str,Any], scoring_func:Callable) -> float:
        """Wrapper function for objective to pass the required datasets and other parameters for hyperparameter tuning."""
        self.logger.info(f"n_splits: {n_splits}")
        self.logger.info(f"parameter_search_space: {parameter_search_space}")
        self.logger.info(f"scoring_func: {scoring_func.__name__}")
                
        def tune_hyperparameters_objective(trial):
            """ Objective function for tuning the parameters with optuna with early stopping."""

            ## Basic parameters dict
            params = {}

            # Parse and add params from parameter_search_space
            for name, distribution in parameter_search_space.items():
                params[name] = trial._suggest(name, distribution)


            # Stratified Cross Validation
            cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 30)
            kfold_scores = []
            kfold_best_n_estimator = []


            # Loop through the k folds and keep each one as validation
            for train_idx,val_idx in cv.split(self.X, self.y):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
                # Early stopping on validation set
                xgb_model = XGBClassifier(**params, n_estimators = 300, early_stopping_rounds=50)
                xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # Extract predictions
                y_preds = xgb_model.predict(X_val)
                y_probs = xgb_model.predict_proba(X_val)

                # Extract score 
                score = scoring_func(y_val, y_preds, y_probs)

                # Extract and store best iteration (number of trees)
                kfold_best_n_estimator.append(xgb_model.best_iteration)

                # Add scores to list
                kfold_scores.append(score)

            # Extract and store the best iteration in memory
            trial.set_user_attr("n_estimators", int(np.mean(kfold_best_n_estimator)))
            return np.mean(kfold_scores)

        return tune_hyperparameters_objective


    def extract_best_parameters(self, 
             parameter_search_space: Dict[str, Any], 
             n_trials: int = 20,
             scoring_func:Callable=None)->Dict[str, Any]:
        "Function for performing hyperparameter tunning and extracting the best hyperparameters"

        # Create the objective function
        self.logger.info(f"Creating Objective Function")
        custom_objective = self.create_objective(
            n_splits=5, 
            parameter_search_space=parameter_search_space,
            scoring_func= scoring_func
        )

        # Use Optuna for hyperparameter tuning
        self.logger.info(f"Creating study in optuna")

        # Completely silence Optuna's own terminal prints
        optuna_logger = optuna.logging.get_logger("optuna")

        # Remove Optuna's default terminal handler
        for handler in optuna_logger.handlers[:]:
            optuna_logger.removeHandler(handler)

        # Send Optuna logs to logger's handlers
        for handler in self.logger.handlers:
            optuna_logger.addHandler(handler)

        # Create study
        study = optuna.create_study(direction="maximize")
        self.logger.info(f"Extracting Best Hyperparameters")

        study.optimize(custom_objective, n_trials=n_trials)


        best_parameters = study.best_params
        self.logger.info(f"Best Hyperparamters : {best_parameters}")
        try:
            best_n_estimators = int(np.mean(study.best_trial.user_attrs["n_estimators"]))
        except KeyError:
             best_n_estimators = 300 
        self.logger.info(f"Ideal n_estimators: {best_n_estimators}")

        results = {
            "best_parameters": best_parameters,
            "optimal_n_estimators": best_n_estimators,
        }

        results_file = create_timestamped_filename(base_name= f"{self.model_folder}/hyperparameters",extension = "json")
        
        try:
            with open(results_file, "w") as f:
                json.dump(results, f)
        except Exception as e:
            self.logger.info(f"Unable to save file to {results_file} : {e}")

        self.logger.info(f"Best Hyperparameters saved to {results_file}")

        return results








