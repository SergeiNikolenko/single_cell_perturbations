import gc
import json
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf


def load_json_file(path: Path) -> dict:
    """
    Args: path
    Returns: dct - dictionary
    """
    with open(path) as f:
        dct = json.load(f)
        return dct
    

def load_raw_data_file(file) -> pd.DataFrame:
    """
    Args: None
    Returns: a training data frame.
    """
    root = Path(__file__).parent
    path = load_json_file(root / 'SETTINGS.json')['RAW_DATA_DIR']
    path = root / path / file
    if str(path).endswith('csv'):
        df = pd.read_csv(path)
    elif str(path).endswith('parquet'):
        df = pd.read_parquet(path)
    return df


def load_sample_submission() -> pd.DataFrame:
    """
    Args: None
    Returns: a sample submission.
    """
    df = load_raw_data_file('sample_submission.csv')
    return df


def load_test_data() -> pd.DataFrame:
    """
    Args: None
    Returns: a test data frame.
    """
    df = load_raw_data_file('id_map.csv')
    return df


def load_train_data() -> pd.DataFrame:
    """
    Args: None
    Returns: a training data frame.
    """
    df = load_raw_data_file('de_train.parquet')
    df = df.sample(frac=1.0, random_state=42)
    return df


def load_training_data(stage: str) -> Union[np.array, np.array]:
    """
    Args:
        stage: one of simple, stage_1, stage_2
    Returns:
        x: training input
        y: training labels
    """
    root = Path(__file__).parent
    path = Path(load_json_file(root / 'SETTINGS.json')['TRAIN_DATA_CLEAN_PATH']) / stage
    x_path = path / 'x.npy'
    y_path = path / 'y.npy'
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


def load_test_x(stage: str) -> np.array:
    """
    Args:
        stage: one of simple, stage_1, stage_2
    Returns:
        x: test input
    """
    root = Path(__file__).parent
    path = root / load_json_file(root / 'SETTINGS.json')['TEST_DATA_CLEAN_PATH'] / stage
    x_path = path / 'test_x.npy'
    x = np.load(x_path)
    return x


def load_pseudolabels() -> pd.DataFrame:
    """
    Args: None
    Returns: pseudolabels, which can use as final solution too.
    """
    root = Path(__file__).parent
    path = load_json_file(root / 'SETTINGS.json')['SUBMISSION_DIR']
    path = root / path / 'stage_1_submission.csv'
    pseudolabel = pd.read_csv(path)
    test_df = load_test_data()
    pseudolabel = pd.concat([test_df[['cell_type', 'sm_name']], pseudolabel.loc[:, 'A1BG':]], axis=1)
    return pseudolabel


def save_preds(preds: np.array, name:str) -> None:
    """
    Save prediction in data directory
    Args:
        preds - a predicted numpy array
    Returns: None
    """
    columns = load_sample_submission().columns
    df = pd.DataFrame(preds, columns = columns[1:])
    df['id'] = range(len(df))
    df = df.loc[:, columns]
    root = Path(__file__).parent
    path = root / load_json_file(root / 'SETTINGS.json')['SUBMISSION_DIR']
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / name, index=False)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Union[np.array, np.array, np.array]:
    """
    Preprocess train and test data frames for neural network use.
    Args:
        train_df - a training data frame
        test_df - a test data frame
    Returns:
        x - train input
        y - train labels
        test_x - test input
    """
    x = train_df[['cell_type', 'sm_name']].values
    y = train_df.loc[:, 'A1BG':].values
    encoder = LabelEncoder()
    encoder.fit(x.flat)
    x = encoder.transform(x.flat).reshape(-1, 2)
    test_x = test_df[['cell_type', 'sm_name']].values
    test_x = encoder.transform(test_x.flat).reshape(-1, 2)
    return x, y, test_x


def clip(array: np.array) -> np.array:
    """
    Clip predictions between min and max values per columns
    Args:
        array - numpy array
    Returns:
        clipped_pred - numpy array
    """
    values = load_train_data().loc[:, 'A1BG':].values
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    clipped_pred = np.clip(array, mins, maxs)
    return clipped_pred


def fit_and_predict_embedding_nn(x: np.array, 
                                 y: np.array, 
                                 test_x: np.array, 
                                 model_constructor: callable, 
                                 best_params: dict) -> np.array:
    """
    Train neural network and predict test input.
    Args:
        x - train input
        y - train labels
        test_x - test input
        model_constructor - model constructor
        best_params - params for model creating and training environment
    Returns:
        pred - a prediction
    """
    model_params, training_params = split_params_to_training_model(best_params)
    n_dim = model_params['n_dim']
    d = TruncatedSVD(n_dim)
    y = d.fit_transform(y)
    model = model_constructor(**model_params)
    model.fit(x, y, epochs=training_params['epochs'], 
                    batch_size=training_params['bs'], 
                    verbose=0,
                    shuffle=True)
    pred = d.inverse_transform(model.predict(test_x, batch_size=1))
    return pred


def train_nn(x: np.array, 
             y: np.array, 
             model_constructor: callable, 
             best_params: dict,
             model_path:Path) -> np.array:
    model_params, training_params = split_params_to_training_model(best_params)
    n_dim = model_params['n_dim']
    d = TruncatedSVD(n_dim)
    y = d.fit_transform(y)
    model = model_constructor(**model_params)
    model.fit(x, y, epochs=training_params['epochs'], 
                    batch_size=training_params['bs'], 
                    verbose=1,
                    shuffle=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    

def predict(x: np.array,
            y: np.array, 
            x_test: np.array, 
            models: List[callable], 
            params: List[dict], 
            weights: List[float],
            reps: int) -> np.array:
    """
    Predict for multiple models and use weighted mean to get final prediction.
    Args:
        x - a train input
        y - a train labels
        test_x - a test input
        models - a list of model costructors
        params - a list of parameters
        weights - weights for weighted mean ensembling
        reps - repeats of training and prediction
    Returns:
        pred - an ensembled prediction
    """
    preds = []
    for model, param in zip(models, params):
        temp_pred = [fit_and_predict_embedding_nn(x, y, x_test, model, param) for i in range(reps)]
        temp_pred = np.median(temp_pred, axis=0)
        preds.append(temp_pred)
    pred = np.sum([w * p for w, p in zip(weights, preds)], axis=0) / sum(weights)    
    return pred


def reset_tensorflow_keras_backend() -> None:
    """
    Clears gpu after model training.
    Args: None  
    Returns: None
    """
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()


def mean_rowwise_rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    The main metric.
    Args:
        y_true - targets
        y_pred - predictions
    Returns:
        mrrmse_score - value of metric
    """
    rowwise_rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    mrrmse_score = np.mean(rowwise_rmse)
    return mrrmse_score


def custom_mean_rowwise_rmse(y_true, y_pred):
    """"
    The main loss.
    Args:
        y_true - keras tensor
        y_pred - keras tensor
    Returns:
        mean_rmse - keras tensor
    """
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    mean_rmse = tf.reduce_mean(rmse_per_row)
    return mean_rmse


def split_params_to_training_model(model_params: dict) -> Union[dict, dict]:
    """"
    Split dict of params to model and training params.
    Args:
        model_params: all training paramaters
    Returns:
        model_params - params for mode    root = Path(__file__).parent
    path = load_json_file(root / 'SETTINGS.json')['SUBMISSION_DIR']l
        training_params - params for training
    """
    model_params = model_params['params']
    training_keys = ['epochs', 'bs']
    training_params = {k: model_params[k] for k in training_keys}
    model_params = {k: model_params[k] for k in model_params.keys() if k not in training_keys}
    return model_params, training_params



