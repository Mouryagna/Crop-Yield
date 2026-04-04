import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    encoder_obj_file_path = os.path.join(PROJECT_ROOT, "artifacts", "encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df):
        try:
            # Fix dtypes first
            df['average_rain_fall_mm_per_year'] = pd.to_numeric(
                df['average_rain_fall_mm_per_year'], errors='coerce'
            )
            df['avg_temp'] = pd.to_numeric(df['avg_temp'], errors='coerce')
            df['pesticides_tonnes'] = pd.to_numeric(df['pesticides_tonnes'], errors='coerce')
            df['hg/ha_yield'] = pd.to_numeric(df['hg/ha_yield'], errors='coerce')
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

            df.dropna(inplace=True)

            # Feature engineering
            df['hg/ha_yield_log'] = np.log1p(df['hg/ha_yield'])
            df['pesticides_cbrt'] = np.cbrt(df['pesticides_tonnes'])
            df['temp_stress'] = (df['avg_temp'] - 20).abs()
            df['rain_temp_ratio'] = df['average_rain_fall_mm_per_year'] / (df['avg_temp'] + 1)
            df['year_trend'] = df['Year'] - 1990

            rainfall_map = {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            temp_map = {'Cold': 0, 'Mild': 1, 'Warm': 2, 'Hot': 3}

            df['rainfall_category'] = pd.cut(
                df['average_rain_fall_mm_per_year'],
                bins=[0, 500, 1000, 1500, 2000, 99999],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            ).map(rainfall_map)

            df['temp_category'] = pd.cut(
                df['avg_temp'],
                bins=[-99, 10, 20, 30, 99],
                labels=['Cold', 'Mild', 'Warm', 'Hot']
            ).map(temp_map)

            df.dropna(inplace=True)
            logging.info("Feature engineering completed")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            numeric_cols = [
                'Year', 'average_rain_fall_mm_per_year', 'pesticides_cbrt',
                'avg_temp', 'temp_stress', 'rain_temp_ratio',
                'year_trend', 'rainfall_category', 'temp_category'
            ]

            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numeric_cols)
            ], remainder='passthrough')

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded")

            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            target_column = 'hg/ha_yield_log'

            feature_cols = [
                'Area', 'Item', 'Year', 'average_rain_fall_mm_per_year',
                'pesticides_cbrt', 'avg_temp', 'temp_stress',
                'rain_temp_ratio', 'year_trend',
                'rainfall_category', 'temp_category'
            ]

            X_train = train_df[feature_cols]
            y_train = train_df[target_column]

            X_test = test_df[feature_cols]
            y_test = test_df[target_column]

            # Target Encoding
            encoder = ce.TargetEncoder(cols=['Area', 'Item'])
            X_train = encoder.fit_transform(X_train, y_train)
            X_test = encoder.transform(X_test)

            # Scaling
            preprocessor = self.get_data_transformer_object()
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logging.info("Saving preprocessing objects")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            save_object(
                file_path=self.data_transformation_config.encoder_obj_file_path,
                obj=encoder
            )

            logging.info("Data transformation completed")

            return (
                X_train,
                y_train,
                X_test,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.encoder_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)