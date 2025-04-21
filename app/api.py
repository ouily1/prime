"""An API to expose our trained pipeline for prime prediction."""
import sys
import os
from pathlib import Path
import logging

PATH = str(Path(os.path.split(__file__)[0]).parent)
sys.path.insert(1, PATH + "/src")
sys.path.insert(2, PATH)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.utils import get_model, ModelEnsemble


PATH_LOG = str(Path(os.path.split(__file__)[0])) # Location to save log_file

logging.basicConfig(
    filename = PATH_LOG + "/log_file.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

# creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_freq
    global model_reg
    global model

    model_freq_name: str = os.getenv("MLFLOW_MODEL_FREQ_NAME")
    model_freq_version: str = os.getenv("MLFLOW_MODEL_FREQ_VERSION")
    model_reg_name: str = os.getenv("MLFLOW_MODEL_REG_NAME")
    model_reg_version: str = os.getenv("MLFLOW_MODEL_REG_VERSION")
    # Load the ML model
    model_freq = get_model(model_freq_name, model_freq_version)
    model_reg = get_model(model_reg_name, model_reg_version)
    model = ModelEnsemble(model_freq, model_reg)
    yield

app = FastAPI(lifespan=lifespan, title="Prédiction prime", description="test")

@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """
    return {
        "Message": "API de prédiction de prime",
        "Model_name": "Prime ML",
        "Model_version": "0.1",
    }

@app.get("/predict", tags=["Predict"])
async def predict(
    Type: str = "A",
    Occupation: str = "Employed",
    Age: float = 30,
    Group1: int = 3,
    Bonus: int = 23,
    Poldur: int = 45,
    Value: float = 1000.0,
    Adind: int = 1,
    Density: float = 100.0,
    Exppdays: float = 1,
) -> float:
    """
    Predict function of the API.
    """

    df = pd.DataFrame(
        {
            "Type": [Type],
            "Occupation": [Occupation],
            "Age": [Age],
            "Group1": [Group1],
            "Bonus": [Bonus],
            "Poldur": [Poldur],
            "Value": [Value],
            "Adind": [Adind],
            "Density": [Density],
            "Exppdays": [Exppdays],
        }
    )

    prediction, probabability = model.transform(df)
    p0, p1 = probabability[0], probabability[1]
    logger.info("prediction: %.2f probability: %.2f, %.2f" % (prediction, p0, p1))
    
    return prediction
