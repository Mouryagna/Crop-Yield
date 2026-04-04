import os
import sys
import json
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    model_report_path = os.path.join(PROJECT_ROOT, "artifacts", "model_report.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Training XGBoost with GridSearchCV")

            params = {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "n_estimators": [200, 300, 500],
                "max_depth": [5, 7, 9],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1, 3, 5]
            }

            xgb = XGBRegressor(random_state=42)

            gs = GridSearchCV(
                xgb,
                params,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            logging.info(f"Best Params: {gs.best_params_}")

            y_pred = best_model.predict(X_test)
            y_pred_actual = np.expm1(y_pred)
            y_test_actual = np.expm1(y_test)

            r2   = r2_score(y_test_actual, y_pred_actual)
            mae  = mean_absolute_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

            logging.info(f"R2:   {r2}")
            logging.info(f"MAE:  {mae}")
            logging.info(f"RMSE: {rmse}")

            evaluation_results = {
                "best_params": gs.best_params_,
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse
            }

            with open(self.model_trainer_config.model_report_path, "w") as f:
                json.dump(evaluation_results, f, indent=4)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2

        except Exception as e:
            raise CustomException(e, sys)