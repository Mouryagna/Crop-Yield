from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import sys

class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")

            transformation = DataTransformation()
            X_train, y_train, X_test, y_test, _, _ = transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info("Data transformation completed")

            trainer = ModelTrainer()
            r2 = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
            logging.info(f"Training completed! Best R2: {r2}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()