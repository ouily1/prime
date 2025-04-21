from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_pipeline(num_variables, cat_variables, model):
    """ Create a pipeline for preprocessing and model definition
    """
    scaler = StandardScaler()
    numeric_tranformation = Pipeline(
        steps=[('scaler', scaler)]
    )

    onehot = OneHotEncoder()
    categorical_tranformation = Pipeline(
        steps=[('onehot', onehot)]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numerical", numeric_tranformation, num_variables),
                      ("categorical", categorical_tranformation, cat_variables)]
    )

    pipe = Pipeline(
        steps=[('preprocessor', preprocessor),
               ("model", model)]
    )
    return pipe
