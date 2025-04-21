import os
import s3fs
import pandas as pd
import pyarrow.parquet as pq

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
BUCKET = "ahmed" # "danalejo" #
FILE_PATH_S3 = BUCKET + "/diffusion/"


fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL}, key = os.environ["AWS_ACCESS_KEY_ID"], secret = os.environ["AWS_SECRET_ACCESS_KEY"])

def load_csv(file="training.csv", sep=";"):
    with fs.open(FILE_PATH_S3 + file, mode="rb") as file:
        df = pd.read_csv(file, sep=sep)
    return df


def send_csv(df, name):
    with fs.open(FILE_PATH_S3 + name, "w") as file:
        df.to_csv(file)

def send_parquet(table, name):
    pq.write_to_dataset(table, FILE_PATH_S3 + name, filesystem=fs)