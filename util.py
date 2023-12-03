import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf


def load_train_data():
    path = Path(__file__).parent / 'data/de_train.parquet'
    df = pd.read_parquet(path)
    df = df.sample(frac=1.0, random_state=42)
    return df


def load_test_data():
    path = Path(__file__).parent / 'data/id_map.csv'
    df = pd.read_csv(path)
    return df


def load_sample_submission():
    path = Path(__file__).parent / 'data/sample_submission.csv'
    df = pd.read_csv(path)
    return df


def load_pseudolabels():
    path = Path(__file__).parent / 'data/submission_stage_1.csv'
    pseudolabel = pd.read_csv(path)
    test_df = load_test_data()
    pseudolabel = pd.concat([test_df[['cell_type', 'sm_name']], pseudolabel.loc[:, 'A1BG':]], axis=1)
    return pseudolabel


def save_preds(preds, name):
    columns = load_sample_submission().columns
    df = pd.DataFrame(preds, columns = columns[1:])
    df['id'] = range(len(df))
    df = df.loc[:, columns]
    return df.to_csv(Path(__file__).parent / 'data' / name, index=False)


def preprocess_data(train_df, test_df):
    x = train_df[['cell_type', 'sm_name']].values
    y = train_df.loc[:, 'A1BG':].values
    encoder = LabelEncoder()
    encoder.fit(x.flat)
    x = encoder.transform(x.flat).reshape(-1, 2)
    test_x = test_df[['cell_type', 'sm_name']].values
    test_x = encoder.transform(test_x.flat).reshape(-1, 2)
    return x, y, test_x

def clip(array):
    values = load_train_data().loc[:, 'A1BG':].values
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    clipped_pred = np.clip(array, mins, maxs)
    return clipped_pred

def fit_and_predict_embedding_nn(x, y, test_x, model_constructor, best_params):
    model_params, training_params = split_params_to_training_model(best_params)
    n_dim = model_params['n_dim']
    d = TruncatedSVD(n_dim)
    y = d.fit_transform(y)
    model = model_constructor(**model_params)
    model.fit(x, y, epochs=training_params['epochs'], 
                    batch_size=training_params['bs'], 
                    verbose=0,
                    shuffle=True)
    return d.inverse_transform(model.predict(test_x, batch_size=1))


def predict(x, y, x_test, models, params, weights, reps=10):
    preds = []
    for model, param in zip(models, params):
        temp_pred = [fit_and_predict_embedding_nn(x, y, x_test, model, param) for i in range(reps)]
        temp_pred = np.median(temp_pred, axis=0)
        preds.append(temp_pred)
    pred = np.sum([w * p for w, p in zip(weights, preds)], axis=0) / sum(weights)    
    return pred


def reset_tensorflow_keras_backend():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()


def mean_rowwise_rmse(y_true, y_pred):
    rowwise_rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    mrrmse_score = np.mean(rowwise_rmse)
    return mrrmse_score


def abs_error(true, pred):
    return np.abs(true - pred).mean()


def custom_mean_rowwise_rmse(y_true, y_pred):
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    mean_rmse = tf.reduce_mean(rmse_per_row)
    return mean_rmse


def split_params_to_training_model(model_params):
    model_params = model_params['params']
    training_keys = ['epochs', 'bs']
    training_params = {k: model_params[k] for k in training_keys}
    model_params = {k: model_params[k] for k in model_params.keys() if k not in training_keys}
    return model_params, training_params



