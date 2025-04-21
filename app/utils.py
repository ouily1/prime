"""
Utils.
"""
import mlflow
import random
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



def get_model(
    model_name: str, model_version: str
):
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.
    Returns:
        model (mlflow.pyfunc.PyFuncModel): The loaded machine learning model.
    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

    try:
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error


class ModelEnsemble(BaseEstimator, TransformerMixin):
    """ Class to estimate the premium value for a given client.
    It first predict the ocurrence of a insurance claims, then it proceed to calculate the premium.
    This class is not trainable, it receives models that have been already trained.
    """
    def __init__(self, model1, model2, prime_avg=86.76, n0=42294, n1=5751):
        self.model1 = model1
        self.model2 = model2
        self.prime_avg = prime_avg
        self.n0 = n0
        self.n1 = n1
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Method to calculate the premium value for a given set of parameters X.
        PARAMETERS
        ----------
        X: DataFrame
           Dataframe containing the features
        OUTPUT
        ------
        premium: float
               premium value predicted
        proba: list
               probabilities of the possible outcomes               
        """
        model1_output = self.model1.predict(X)
        proba = self.model1.predict_proba(X)[0]
        proba = [round(proba[0], 2), round(proba[1], 2)]
        if model1_output == 0:
            prime = self.prime_avg
        else:
            X["frequence_claims"] = model1_output
            prime = self.prime_avg + self.n1 * np.expm1(self.model2.predict(X)[0]) / self.n0
        return round(prime, 2), proba
