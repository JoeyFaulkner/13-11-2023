import pandas as pd
import xgboost as xgb


def _parse_health(health: str) -> int:
    match health:
        case "Healthy":
            return 0
        case "Minor Injury":
            return 1
        case "Serious Injury":
            return 2
        case _:
            raise NotImplementedError("new health case appeared")


def _parse_maturity_size(maturity_size: str) -> int:
    match maturity_size:
        case "Small":
            return 0
        case "Medium":
            return 1
        case "Large":
            return 2
        case _:
            raise NotImplementedError("new maturity case appeared")


def _parse_fur_length(fur_length: str) -> int:
    match fur_length:
        case "Short":
            return 0
        case "Medium":
            return 1
        case "Long":
            return 2


def feature_engineering(df: pd.DataFrame):
    df["Adopted"] = df["Adopted"].map(lambda l: l == "Yes")
    df["Health"] = df["Health"].map(_parse_health)
    df["MaturitySize"] = df["MaturitySize"].map(_parse_maturity_size)
    df["FurLength"] = df["FurLength"].map(_parse_fur_length)
    float_columns = ["Age", "Fee", "PhotoAmt"]
    for column in float_columns:
        df[column] = df[column].astype(float)

    categorical_columns = ["Type", "Breed1", "Gender", "Color1", "Color2"]
    for column in categorical_columns:
        df[column] = df[column].astype("category")
    bool_columns = ["Vaccinated", "Sterilized"]
    for column in bool_columns:
        df[column] = df[column].astype(bool)
    return df


def create_dmatrix(df: pd.DataFrame, target_column: str = None):
    if target_column:
        label = df[target_column]
        data = df.drop(target_column, axis=1)
    else:
        label = None
        data = df
    return xgb.DMatrix(data, label=label, enable_categorical=True)
