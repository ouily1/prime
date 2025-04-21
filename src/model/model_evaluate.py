from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error


def evaluate_model_freq(pipe, x_test, y_test):
    """ Function to evaluate the model to predict the ocurrence of insurance claims.
        PARAMETERS
        ----------
        pipe: Pipeline object
              fitted pipeline
        x_test: DataFrame
                DataFrame containing the features
        y_test: Series
                Series object containing the target
        OUTPUT
        ------
        matrix: ndarray object
                Confusion matrix
        report: dict
                Text summary of precision, recall and F1 score for each class
    """
    matrix = confusion_matrix(y_test, pipe.predict(x_test))
    report = classification_report(y_test, pipe.predict(x_test), output_dict=True)
    return matrix, report 

def evaluate_model_cost(pipe, x_test, y_test):
    """ Function to evaluate the model to predict the ocurrence of insurance claims.
        PARAMETERS
        ----------
        pipe: Pipeline object
              fitted pipeline
        x_test: DataFrame
                DataFrame containing the features
        y_test: Series
                Series object containing the target
        OUTPUT
        ------
        mse: float
             Mean square error
    """
    mse = mean_squared_error(y_test, pipe.predict(x_test))
    return mse
