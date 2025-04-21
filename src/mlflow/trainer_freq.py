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
    remote_server_uri, mlflow_experiment_name, run_name, kernel, degree, class_weight
):

    data = load_csv()
    data = preprocessing(data)

    PRIME_AVG = sum(np.expm1(data["total_cost"])) / len(data["total_cost"])


    # Set up MLFlow context
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    if True:
        # For each hyperparameter combination we trained the model with, we log a run in MLflow
        #run_name = f"run_freq {idx}"

        params = {'kernel': kernel, 'degree': degree, 'class_weight': class_weight}
        with mlflow.start_run(run_name=run_name):
            freq_pipeline = frequency_train(
                df=data, **params
            )

            mlflow.log_params(
                {"kernel": kernel, "degree": degree, "class_weight": class_weight}
            )

            # Log fit metrics a ajouter
            test_data = load_csv('test_freq.csv', ',')
            x_test = test_data.drop(['frequence_claims', 'Unnamed: 0'], axis=1) 
            y_test = test_data['frequence_claims'] 
            confusion_matrix, report = evaluate_model_freq(freq_pipeline, x_test, y_test)
            # Log model as an artifact
            mlflow.sklearn.log_model(freq_pipeline, "freq_model")
            # metric
            for key in ["accuracy", "macro avg", "weighted avg"]:
                report.pop(key, None)
            for class_or_avg, metrics_dict in report.items():
                for metric, value in metrics_dict.items():
                    mlflow.log_metric(class_or_avg + '_' + metric, round(value, 3))
            # confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix)
            disp.plot(cmap='Blues')
            plt.savefig("confusion-matrix.png")
            mlflow.log_artifact("confusion-matrix.png")
            os.remove("confusion-matrix.png")


if __name__ == "__main__":

    #if len(sys.argv) > 1:
    remote_server_uri = str(sys.argv[1])
    experiment_name = str(sys.argv[2])
    run_name = str(sys.argv[3])
    kernel = str(sys.argv[4])
    degree = int(sys.argv[5])
    class_weight = str(sys.argv[6])




    log_to_mlflow(
        remote_server_uri,
        experiment_name,
        run_name,
        kernel,
        degree,
        class_weight)
            