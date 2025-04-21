import sys
import os
from pathlib import Path

path = Path(os.path.split(__file__)[0])
sys.path.insert(1, str(path.parent))
sys.path.insert(2, str(path.parent.parent))

import joblib
import pandas as pd
import numpy as np
from pipeline import preprocessing
from model.train_models import frequency_train, cost_train
from model.model_evaluate import evaluate_model_cost, evaluate_model_freq
from pipeline.preprocessing import preprocessing
from data.data_load import load_csv
from app.utils import ModelEnsemble


#---------------------------------------------------------------#
#                           Preprocessing                       #
#---------------------------------------------------------------#

# Loading data
data = load_csv()
# Preprocessing
data = preprocessing(data)

PRIME_AVG = sum(np.expm1(data['total_cost'])) / len(data['total_cost'])
N_0 = len(data[data['total_cost'] == 0])
N_1 = len(data['total_cost']) - N_0

#---------------------------------------------------------------#
#                 Frequency prediction training                 #
#---------------------------------------------------------------#

freq_pipeline = frequency_train(data)

#---------------------------------------------------------------#
#                     Cost prediction training                  #
#---------------------------------------------------------------#

nn_pipeline = cost_train(data)

#---------------------------------------------------------------#
#                            Merging all                        #
#---------------------------------------------------------------#

ensemble_model = ModelEnsemble(freq_pipeline, nn_pipeline, PRIME_AVG, N_0, N_1)
# Saving the model
joblib.dump(ensemble_model, 'ensemble_model.joblib')