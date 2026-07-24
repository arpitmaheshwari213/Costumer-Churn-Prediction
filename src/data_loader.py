from pathlib import Path
import pandas as pd
import numpy as np
from log_generator import make_logger
from typing import Callable


class LoadDataset:
    """Class for loading the data"""

    def __init__(self, data_folder: str, logger: Callable = None):
        """Initializes the data folder"""
        # Resolve root folder
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        clean_folder_name = data_folder.lstrip("./")
        self.data_folder = f"{project_root}/{clean_folder_name}"

        if logger:
            self.logger = logger
        else:
            log_file = f"{project_root}/logs/data_loader.log"
            self.logger = make_logger("DataLoader", log_file)

    def load_excel(self, file_name: str) -> pd.DataFrame:
        """Loads the excel file.
        Currently support .xlsx files with single sheet only."""
        self.logger.info("Loading Dataset")
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
        self.logger.info("Cleaning Dataset")
        try:
            # Separate features and target variable
            X = dataset.drop(columns=[target_col]) if target_col else dataset.copy()
            y = dataset[target_col].values if target_col else None

            y = np.where(y == "No", 0, 1)

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
                    f"Cleaning Dataset Complete. Shape of features and target {X.shape} & {len(y)}"
                )
            else:
                self.logger.info(
                    f"Cleaning Dataset Complete. Shape of features {X.shape}"
                )

            return X, y
        except Exception as e:
            raise ValueError(f"Failed to clean the dataset {e}")
