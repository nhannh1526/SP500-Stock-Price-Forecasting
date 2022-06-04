import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def feature_label_split(df, target_col):
    return df.drop(columns=[target_col]), df[[target_col]]


def train_val_test_split(df, target_col, test_ratio):
    val_ration = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ration, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


def transform_data(X_train, X_val, X_test, y_train, y_val, y_test, scaling="standard"):
    scaler = get_scaler(scaling)

    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    return X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr, scaler


def load_data_into_dataloader(X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr, batch_size=64):
    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    # test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df, scaler):
    preds = np.concatenate(predictions, axis=0).ravel()
    vals = np.concatenate(values, axis=0).ravel()
    df = pd.DataFrame(data={"true": vals, "pred": preds},
                      index=df.head(len(vals)).index)
    df.sort_index(inplace=True)
    df = inverse_transform(scaler, df, [["true", "pred"]])
    return df


def get_evaluation_metrics(y_true, y_pred):
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    MAE = mean_absolute_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    return RMSE, MAE, MAPE
