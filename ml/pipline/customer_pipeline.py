import os
import sys
from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from django.conf import settings


@dataclass
class DataIngestionConfig:
    trained_model_path: str = settings.BASE_DIR / "ml/models/customer.pkl"


class PredictionPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ingestion_config = DataIngestionConfig()

    def load_trained_model(self):
        try:
            model = load_object(self.ingestion_config.trained_model_path)
            logging.info("Trained model loaded successfully")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def predict_and_evaluate(self):
        try:
            new_data = pd.read_csv(self.data_path)
            logging.info("Read the new data for prediction")

            model = self.load_trained_model()
            new_data = new_data.dropna()

            new_data = new_data.drop("CUST_ID", axis=1)

            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(new_data)

            pca = PCA(n_components=2)

            new_data_pca = pca.fit_transform(df_scaled)

            new_data["cluster_pca"] = model.predict(new_data_pca)

            return new_data

        except Exception as e:
            raise CustomException(e, sys)


def predict(data_path):
    pipeline = PredictionPipeline(data_path)
    new_data = pipeline.predict_and_evaluate()
    return new_data


if __name__ == "__main__":
    data_path = "notebook/data/segmented_customers.csv"
    pipeline = PredictionPipeline(data_path)
    new_data = pipeline.predict_and_evaluate()

    print(new_data.to_string(index=True))
