import sys
import os
from pathlib import Path

path = Path(os.path.split(__file__)[0])
sys.path.insert(1, str(path.parent))

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from model.train_models import frequency_train, cost_train
from pipeline.preprocessing import preprocessing
from model.model_evaluate import evaluate_model_cost, evaluate_model_freq
from data.data_load import load_csv
import mlflow
import os


def log_to_mlflow(
    remote_server_uri, mlflow_experiment_name, run_name, n_estimator, max_depth
):

    data = load_csv()
    data = preprocessing(data)

    PRIME_AVG = sum(np.expm1(data["total_cost"])) / len(data["total_cost"])


    # Set up MLFlow context
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

        # For each hyperparameter combination we trained the model with, we log a run in MLflow
        #run_name = f"run_freq {idx}"

    params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    with mlflow.start_run(run_name=run_name):
        reg_pipeline = cost_train(
            df=data,
            **params,
            test_path="test_cost.csv",
        )

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})

        # Log fit metrics a ajouter
        test_data = load_csv('test_cost.csv', ',')
        x_test = test_data.drop(['target', 'Unnamed: 0'], axis=1) 
        y_test = test_data['target']
        mse = evaluate_model_cost(reg_pipeline, x_test, y_test)
        mlflow.log_metric("Mean square error", round(mse, 3))
        # Log model as an artifact
        mlflow.sklearn.log_model(reg_pipeline, "reg_model")


if __name__ == "__main__":

    #if len(sys.argv) > 1:
    remote_server_uri = str(sys.argv[1])
    experiment_name = str(sys.argv[2])
    run_name = str(sys.argv[3])
    n_estimators = int(sys.argv[4])
    max_depth = int(sys.argv[5])
    #reg = bool(sys.argv[9])
    #freq = bool(sys.argv[10])


    log_to_mlflow(
        remote_server_uri,
        experiment_name,
        run_name,
        n_estimators,
        max_depth)