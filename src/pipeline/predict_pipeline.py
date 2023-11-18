import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_ingestion import DataIngestionConfig
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import sys


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
            new_data = pd.read_csv(
                self.data_path,
                parse_dates=["data"],
                index_col="data",
                usecols=["data", "venda"],
            )
            logging.info("Read the new data for prediction")

            model = self.load_trained_model()

            predictions = model.forecast(len(new_data))

            new_data.index = predictions.index

            mse = mean_squared_error(new_data["venda"], predictions)
            rmse = np.sqrt(mse)
            logging.info(f"Root Mean Squared Error (RMSE): {rmse}")

            result_df = pd.DataFrame(
                {"Actual": new_data["venda"], "Predicted": predictions}
            )
            # result_df.to_csv(os.path.join('artifacts', 'prediction_results.csv'), index=True, header=True)

            # new_data.reset_index(drop=True, inplace=True)
            # results_df["date"]=new_data['data']

            return rmse, result_df, new_data  # Include new_data in the return

        except Exception as e:
            raise CustomException(e, sys)


def predict(data_path):
    pipeline = PredictionPipeline(data_path)
    (
        rmse,
        results_df,
        new_data,
    ) = pipeline.predict_and_evaluate()  # Receive new_data in the results
    print(f"RMSE: {rmse}")
    df_subset = pd.read_csv(data_path)
    results_df.reset_index(drop=True, inplace=True)
    results_df["date"] = df_subset["data"]
    return results_df


if __name__ == "__main__":
    data_path = "artifacts/last_30_rows.csv"
    pipeline = PredictionPipeline(data_path)
    (
        rmse,
        results_df,
        new_data,
    ) = pipeline.predict_and_evaluate()  # Receive new_data in the results
    print(f"RMSE: {rmse}")
    df_subset = pd.read_csv(data_path)
    results_df.reset_index(drop=True, inplace=True)
    results_df["date"] = df_subset["data"]
    # Add the 'data' column to result_df

    print("Prediction results:")
    print(results_df.to_string(index=True))
