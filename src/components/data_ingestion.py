import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
from src.utils import save_object

from django.conf import settings


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    trained_model_path: str = settings.BASE_DIR / "artifacts/trained_model.pkl"


class DataIngestion:
    def __init__(self, data_path):
        self.ingestion_config = DataIngestionConfig()
        self.data_path = data_path

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(
                self.data_path,
                parse_dates=["data"],
                index_col="data",
                usecols=["data", "venda"],
            )
            logging.info("Read the dataset as a dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            test_size = 30
            train = df.iloc[:-test_size]
            test = df.iloc[-test_size:]

            train.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            model = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=7
            )
            model_fit = model.fit()

            save_object(
                file_path=self.ingestion_config.trained_model_path, obj=model_fit
            )

            predictions = model_fit.forecast(len(test))

            mse = mean_squared_error(test, predictions)
            rmse = np.sqrt(mse)

            return mse, rmse

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_path = "notebook/data/mock_kaggle.csv"
    obj = DataIngestion(data_path)
    print(obj.initiate_data_ingestion())
