import os
import pandas as pd
from .utils import feature_engineering, create_dmatrix
import xgboost as xgb


def get_dataframe(gcs_path: str, prod_flag: bool) -> pd.DataFrame:
    if not prod_flag:
        # run on small sample saved locally to save time and test plumbing
        tmp_file_path = "/tmp/petfinder_serve.csv"
        if os.path.exists(tmp_file_path):
            df = pd.read_csv(tmp_file_path)
        else:
            df = pd.read_csv(gcs_path).head(100)
            df.to_csv(tmp_file_path)
    else:
        df = pd.read_csv(gcs_path)
    return df


def load_booster_from_json(model_path: str) -> xgb.Booster:
    boost = xgb.Booster()
    boost.load_model(model_path)
    return boost


def predict_from_dataframe(df: pd.DataFrame, boost: xgb.Booster) -> pd.Series:
    predict_df = feature_engineering(df)
    predict_df = predict_df[boost.feature_names]
    predictions = boost.predict(create_dmatrix(predict_df, None))
    return predictions


def export_to_csv(df: pd.DataFrame, path: str):
    desired_column_format = [
        "Type",
        "Age",
        "Breed1",
        "Gender",
        "Color1",
        "Color2",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
        "Fee",
        "PhotoAmt",
        "Adopted",
        "Adopted_prediction",
    ]
    df[desired_column_format].to_csv(path)


if __name__ == "__main__":
    gcs_path = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
    prod_flag = os.environ.get("PROD") == "true"
    df = get_dataframe(gcs_path, prod_flag)
    model_id = "v0.1"
    model_path = f"./artifacts/model/{model_id}.json"
    boost = load_booster_from_json(model_path)

    predictions = predict_from_dataframe(df, boost)
    df["Adopted_prediction"] = predictions
    path = "./output/results.csv"
    export_to_csv(df, path)
