from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path
import pickle
from dataclasses import dataclass
import logging
from log_generator import make_logger

@dataclass
class PreprocessorState:
    """
    Container to hold the fitted model and necessary metadata for inference.
    """
    transformer: Any = None
    target_col: str = None
    logger: Any = None

    def save(self,file_path:str):
        """Saves the state to a pickle file."""
        self.logger.info(f"Saving Preprocessor")
        if(file_path):
            try:
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'wb') as f:
                    pickle.dump(self, f)
                
                self.logger.info("[LOG] Preprocessor saved successfully at file_path: {file_path}")
                return True
            except Exception as e:
                self.logger.exception(f"[ERROR] Failed to save the preprocessor at file_path: {file_path} with Error: {e}")
                raise
        else:
            self.logger.exception(f"[ERROR] Filepath is incorrect: {file_path}")

    def load(self, file_path: str):
        """Loads the state from a pickle file."""
        self.logger.info(f"Loading Preprocessor")
        try:
            with open(file_path, 'rb') as f:
                self.__init__(transformer=None) # Reset first
                loaded_state = pickle.load(f)
            
            # Unpack attributes safely
            self.transformer = loaded_state.transformer
            self.target_col = loaded_state.target_col
            
            self.logger.info(f"[LOG] Model loaded successfully from: {file_path}")
            return True
        except Exception as e:
            self.logger.exception(f"[ERROR] Failed to load model: {e}")
            raise


class FeatureExtraction:
    """Class for extracting the features from the data. This includes scaling, null value imputer and categorical encoding"""
    def __init__(self, preprocessor_state_obj = None, log_file: str = "../logs/feature_extraction.log"):
        """Initializes the data folder
        """
        self.preprocessor_state_obj = preprocessor_state_obj
        self.logger = make_logger("FeatureExtraction",log_file)

    def extract_column_types(self,dataset:pd.DataFrame,col_type:str):
        col_list = dataset.select_dtypes(include=[col_type]).columns
        self.logger.info(f"Columns of type {col_type}: {col_list.tolist()}")
        return col_list
    
    def sanitize_names(self, raw_names):
        """Converts sklearn names to clean UPPER_SNAKE_CASE."""
        return [str(name).replace("remainder__", "").replace("-", "_").replace(" ", "_").upper() for name in raw_names]
    
    
    def fit_transform(self,dataset:pd.DataFrame)->pd.DataFrame:
        """Function to fit the feature engineering preprocessor on train set"""
        self.logger.info(f"Fit & Transform Started")
        self.logger.info(f"Loaded Dataset Shape {dataset.shape}")

        try:
            X = dataset.copy()
            # Extract Numerical Features
            num_cols = self.extract_column_types(dataset, col_type="number")
            # Extract categorical columns
            cat_cols = self.extract_column_types(dataset, col_type="object")
            
            self.logger.info(f"Numerical cols: {num_cols}")
            self.logger.info(f"Categorical cols: {cat_cols}")
            
            # Build Transformer
            transformers = []

            if len(num_cols)>0:
                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler())
                ])
                transformers.append(("num", num_pipeline, num_cols))
                
                
            if len(cat_cols)>0:
                transformers.append(("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols))

            # Initialize Column Transformer
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop",sparse_threshold=0)
            

            # Fit Preprocessor
            X_transformed = preprocessor.fit_transform(X)
            self.logger.info(f"Transformed dataframe : {X_transformed.shape}")
            # Reconstruct DataFrame
            feature_names = self.sanitize_names(preprocessor.get_feature_names_out())
            self.logger.info(f"Feature Names {feature_names}")
            
            X_processed = pd.DataFrame(data=X_transformed, columns=feature_names, index=X.index)
            self.logger.info(f"Shape of processed dataframe : {X_processed.shape}")

            preprocessor_state_obj = PreprocessorState(
                transformer=preprocessor, 
                target_col= None,
                logger = self.logger
            )
            self.preprocessor_state_obj = preprocessor_state_obj
        except Exception as e:
            self.logger.exception(f"Feature Engineering Error during fit transform, {str(e)}")
            raise

        return X_processed, preprocessor_state_obj
        

    def transform(self,dataset:pd.DataFrame, target_col: str = None)->pd.DataFrame:
        """Function to fit the feature engineering preprocessor on train set"""
        self.logger.info(f"Transform Started")
        self.logger.info(f"Loaded Dataset Shape: {dataset.shape}")

        try:
            X = dataset.copy()
            # Extract Numerical Features
            num_cols = self.extract_column_types(dataset, col_type="number")
            # Extract categorical columns
            cat_cols = self.extract_column_types(dataset, col_type="object")
            
            self.logger.info(f"Numerical cols: {num_cols}")
            self.logger.info(f"Categorical cols: {cat_cols}")


            # Apply Transformation
            if self.preprocessor_state_obj.transformer is None:
                raise ValueError("Inference Mode: No pre-fitted transformer provided.")
            
            X_transformed = self.preprocessor_state_obj.transformer.transform(X)

            feature_names = self.sanitize_names(self.preprocessor_state_obj.transformer.get_feature_names_out())
            self.logger.info(f"Feature Names {feature_names}")

            X_processed = pd.DataFrame(data=X_transformed, columns=feature_names, index=X.index)
            self.logger.info(f"Shape of processed dataframe {X_processed.shape}")
            return X_processed
        
        except Exception as e:
            self.logger.exception(f"Feature Engineering Error during transform - {str(e)}")
            raise