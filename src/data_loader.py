import logging
import os
import pandas as pd
import numpy as np
from log_generator import make_logger


class LoadDataset:
    """Class for loading the data"""

    def __init__(self, data_folder: str, log_file: str = "../logs/data_loader.log"):
        """Initializes the data folder"""
        self.data_folder = data_folder
        self.logger = make_logger("DataLoader", log_file)

    def load_excel(self, file_name: str) -> pd.DataFrame:
        """Loads the excel file. Currently support .xlsx files with single sheet only."""
        self.logger.info(f"Loading Dataset")
        file_path = self.data_folder + file_name
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            self.logger.info(
                f"[LOG] Successfully loaded {len(df)} rows from {file_path}"
            )
            return df
        except Exception as e:
            self.logger.info(f"[ERROR] Failed to read the Excel file: {e}")
            raise IOError(f"Failed to read the Excel file: {e}")

    def clean_data(self, dataset: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Clean the dataset and rename any fields if required."""
        self.logger.info(f"Cleaning Dataset")
        try:
            # Separate features and target variable
            X = dataset.drop(columns=[target_col]) if target_col else dataset.copy()
            y = dataset[target_col].values if target_col else None

            # Rename Column Values
            X["PaymentMethod"] = X["PaymentMethod"].replace(
                {
                    "Bank transfer (automatic)": "Bank transfer",
                    "Credit card (automatic)": "Credit card",
                }
            )
            X = X.drop("customerID", axis=1)
            # Rename Columns to UPPERCASE
            X.columns = [col.upper() for col in X.columns.to_list()]
            if y is not None:
                self.logger.info(
                    f"Cleaning Dataset Complete. Shape of features {X.shape} & Shape of target Variable {len(y)}"
                )
            else:
                self.logger.info(
                    f"Cleaning Dataset Complete. Shape of features {X.shape}"
                )

            return X, y
        except Exception as e:
            raise ValueError(f"Failed to clean the dataset {e}")
