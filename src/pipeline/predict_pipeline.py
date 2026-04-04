import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            encoder_path = "artifacts/encoder.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            encoder = load_object(encoder_path)

            # Target encode Area and Item
            features_encoded = encoder.transform(features)

            # Scale
            features_scaled = preprocessor.transform(features_encoded)

            # Predict (log scale)
            pred_log = model.predict(features_scaled)

            # Reverse log transform
            prediction = np.expm1(pred_log)

            logging.info(f"Prediction: {prediction}")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Area,
        Item,
        Year,
        average_rain_fall_mm_per_year,
        pesticides_tonnes,
        avg_temp
    ):
        self.Area = Area
        self.Item = Item
        self.Year = Year
        self.average_rain_fall_mm_per_year = average_rain_fall_mm_per_year
        self.pesticides_tonnes = pesticides_tonnes
        self.avg_temp = avg_temp

    def get_data_as_data_frame(self):
        try:
            # Feature engineering on input
            pesticides_cbrt = np.cbrt(self.pesticides_tonnes)
            temp_stress = abs(self.avg_temp - 20)
            rain_temp_ratio = self.average_rain_fall_mm_per_year / (self.avg_temp + 1)
            year_trend = self.Year - 1990

            rainfall_map = {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            temp_map = {'Cold': 0, 'Mild': 1, 'Warm': 2, 'Hot': 3}

            if self.average_rain_fall_mm_per_year <= 500:
                rainfall_category = rainfall_map['Very Low']
            elif self.average_rain_fall_mm_per_year <= 1000:
                rainfall_category = rainfall_map['Low']
            elif self.average_rain_fall_mm_per_year <= 1500:
                rainfall_category = rainfall_map['Medium']
            elif self.average_rain_fall_mm_per_year <= 2000:
                rainfall_category = rainfall_map['High']
            else:
                rainfall_category = rainfall_map['Very High']

            if self.avg_temp <= 10:
                temp_category = temp_map['Cold']
            elif self.avg_temp <= 20:
                temp_category = temp_map['Mild']
            elif self.avg_temp <= 30:
                temp_category = temp_map['Warm']
            else:
                temp_category = temp_map['Hot']

            custom_data_input_dict = {
                'Area': [self.Area],
                'Item': [self.Item],
                'Year': [self.Year],
                'average_rain_fall_mm_per_year': [self.average_rain_fall_mm_per_year],
                'pesticides_cbrt': [pesticides_cbrt],
                'avg_temp': [self.avg_temp],
                'temp_stress': [temp_stress],
                'rain_temp_ratio': [rain_temp_ratio],
                'year_trend': [year_trend],
                'rainfall_category': [rainfall_category],
                'temp_category': [temp_category]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)