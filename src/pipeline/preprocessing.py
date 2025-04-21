import pandas as pd
import numpy as np

# Eliminating wrong values
def idx_lowest(dff):
    """Group by the column 'Polnum' and find the index which corresponds
    ot the the minimal value in the column total_cost. This function output the index
    to eliminate, whose data entries have the same 'Polnum' value, but different 'total_cost'.
    It captures the smaller 'total_cost'.
    """
    df = dff.copy()
    df["total_cost"] = df["Indtppd"] + df["Indtpbi"]
    idx_to_drop = df.groupby(df.columns[0])[df.columns[-1]].idxmin()
    return idx_to_drop

def preprocessing(data):
    """ Function to preprocesses the dataset
    PARAMETERS
    ----------
    data: DataFrame
          Dataframe containing the whole dataset
    OUTPUT
    ------
    data: DataFrame
          preprocessed Dataframe 
    """
    polnums = []
    for i, value in data["PolNum"].value_counts().items():
        if value == 2:
            polnums.append(i)
    wrong_idx = data[data["PolNum"].apply(lambda x: True if x in polnums else False)]
    data.drop(idx_lowest(wrong_idx), inplace=True)

    # Feature engineering
    data["total_cost"] = np.log1p(data["Indtppd"] + data["Indtpbi"])
    data["frequence_claims"] = data["Numtppd"] + data["Numtpbi"]
    data["Exppdays"] = data["Exppdays"] / 365

    # Selecting one particular year
    data = data[data["CalYear"] == 2009]

    # Removing already encoded features
    cols_to_drop = ["Numtppd",	"Numtpbi", "Indtppd", "Indtpbi", "Group2", "Gender",
                    "PolNum", "CalYear", "SubGroup2", "Category"]
    data.drop(cols_to_drop, axis=1, inplace=True)
    
    # Removing outliers
    percentile_cost = np.percentile(data["total_cost"], 98)
    percentile_freq = np.percentile(data["frequence_claims"], 99)
    data = data[(data["total_cost"] < percentile_cost) & (data["frequence_claims"] < percentile_freq)]
    return data
