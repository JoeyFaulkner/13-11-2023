import pandas as pd
import os
import numpy as np
from typing import Dict
import xgboost as xgb
from .utils import feature_engineering, create_dmatrix

np.random.seed(1)


def split_to_parts(df: pd.DataFrame, normalized_split_dict: Dict) -> Dict:
    if sum(normalized_split_dict.values()) != 1:
        raise ValueError("normalized_split_dict values need to add to 1")

    df = df.sample(frac=1.0)  # shuffle the dataframe

    output_dict = {}
    current_idx = 0
    for key, prop in normalized_split_dict.items():
        num_egs = int(np.floor(prop * len(df)))
        output_dict[key] = df.iloc[current_idx : current_idx + num_egs]
        current_idx = current_idx + num_egs

    return output_dict


def train_model(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, target_column: str
) -> xgb.Booster:
    train_dmatrix = create_dmatrix(train_df, target_column=target_column)
    valid_dmatrix = create_dmatrix(valid_df, target_column=target_column)
    return xgb.train(
        params={
            "objective": "reg:logistic",
            "eval_metric": "logloss",
        },
        dtrain=train_dmatrix,
        evals=[(valid_dmatrix, "valid")],
        early_stopping_rounds=10,
        num_boost_round=100,
    )


if __name__ == "__main__":
    import logging
    from sklearn.metrics import f1_score, accuracy_score, recall_score

    gcs_path = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
    prod_flag = os.environ.get("PROD") == "true"
    if not prod_flag:
        # run on small sample saved locally to save time and test plumbing
        tmp_file_path = "/tmp/petfinder.csv"
        if os.path.exists(tmp_file_path):
            df = pd.read_csv(tmp_file_path)
        else:
            df = pd.read_csv(gcs_path).head(100)
            df.to_csv(tmp_file_path)
    else:
        df = pd.read_csv(gcs_path)

    df = feature_engineering(df)
    output_dict = split_to_parts(df, {"train": 0.6, "valid": 0.2, "test": 0.2})

    model = train_model(output_dict["train"], output_dict["valid"], "Adopted")
    test_labels = output_dict["test"]["Adopted"]
    predictions = model.predict(create_dmatrix(output_dict["test"], "Adopted")) > 0.5
    f1 = f1_score(test_labels, predictions)
    acc = accuracy_score(test_labels, predictions)
    rec = recall_score(test_labels, predictions)
    logging.warning(
        f"""model trained, test set evaluations:
    f1: {f1}
    accuracy: {acc}
    recall: {rec}
"""
    )
    if prod_flag:
        model_id = "v0.1"
        model.save_model(f"artifacts/model/{model_id}.json")
        logging.warning(f"model id {model_id} saved.")
