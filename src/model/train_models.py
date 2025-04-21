import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from pipeline.build_pipeline import create_pipeline
from model.models import col_type_selector, random_sampling, removing_zero_cost
from data.data_load import send_csv

def cost_train(df, n_estimators=100, max_depth=7, test_path="test_cost.csv"):
    """ Function to train a model to predict the cost of an insurance claim.
        PARAMETERS
        ----------
        df: DataFrame
            Dataframe containing the features
        test_path: string
                   address and name to save a test file
        OUTPUT
        ------
        nn_pipeline: Pipeline object
                     Trained pipeline
    """
    X = df.drop(["total_cost"], axis=1)
    y = df["total_cost"]

    # Removing zero values from data
    X, y = removing_zero_cost(X, y)
    # Splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Saving test data
    send_csv(pd.concat([x_test, y_test], axis=1), test_path)

    # Defining a selector to get the categorical and numerical variables
    cat_variables, num_variables = col_type_selector(X)
    # Model training
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    nn_pipeline = create_pipeline(num_variables, cat_variables, model)
    nn_pipeline.fit(x_train, y_train)
    return nn_pipeline

def frequency_train(df, kernel='poly', degree=3, class_weight='balanced', test_path="test_freq.csv"):
    """ Function to train a model to predict the ocurrence of a insurance claim.
        PARAMETERS
        ----------
        df: DataFrame
            Dataframe containing the features
        test_path: string
                   address and name to save a test file
        OUTPUT
        ------
        freq_pipeline: Pipeline object
                       Trained pipeline
    """
    X = df.drop(["total_cost", "frequence_claims"], axis=1)
    y = df["frequence_claims"]
    # Defining a selector to get the categorical and numerical variables
    cat_variables, num_variables = col_type_selector(X)
    # Splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=0)
    # Saving test data
    send_csv(pd.concat([x_test, y_test], axis=1), test_path)
    # resampling the data
    x_train, y_train = random_sampling(x_train, y_train, values=[0, 1], new_sizes=[10000, 10000])
    # Splitting resampled data
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    # Model training
    svc = SVC(kernel=kernel, degree=degree, class_weight=class_weight, probability=True, random_state=0)
    freq_pipeline = create_pipeline(num_variables, cat_variables, svc)
    freq_pipeline.fit(x_train, y_train)
    return freq_pipeline
