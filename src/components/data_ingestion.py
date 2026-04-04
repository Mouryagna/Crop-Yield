import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            data_path = os.path.join(PROJECT_ROOT, "Data", "clean_df.csv")
            df = pd.read_csv(data_path)
            logging.info("Dataset read successfully")

            # Clean
            df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
            df.dropna(inplace=True)
            logging.info(f"Dataset shape after cleaning: {df.shape}")

            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Sort before split to preserve time order
            df = df.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, _, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train, y_train, X_test, y_test))