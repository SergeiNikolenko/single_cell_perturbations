import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

from util import custom_mean_rowwise_rmse


"""
Args:
    lr - a learning rate
    emb_out - an output size of the embedding layer
    n_dim - a number of dimentions in a reducted space
    dense_{i} - a number of neurons in a dense layer
    dropout_{i} - a dropout value
"""


def model_1(lr: float, 
            emb_out: int,
            n_dim: int) -> Sequential:
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(1024, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(n_dim, activation= "linear")
    ])
    model.compile(loss="mae", 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model



def model_2(lr: float, 
            emb_out: int, 
            dense_1: int, dense_2: int, 
            dropout_1: float, dropout_2: float,
            n_dim: int) -> Sequential:
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        
        Dense(dense_1),
        BatchNormalization(),
        Activation("relu"),
        
        Dropout(dropout_1),
        Dense(dense_2, activation="relu"),
        Activation("relu"),
        BatchNormalization(),
        Dropout(dropout_2),
        
        Dense(n_dim, activation= "linear")
    ])
    model.compile(loss="mae", 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_3(lr: float, 
            emb_out: int, 
            dense_1: int, dense_2: int , dense_3: int, dense_4: int,
            dropout_1: float, dropout_2: float, dropout_3: float, dropout_4: float,
            n_dim: int) -> Sequential:
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        
        Dense(dense_1),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_1),
        
        Dense(dense_2),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_2),
        
        Dense(dense_3),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_3),

        Dense(dense_4),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_4),
        
        Dense(n_dim, activation= "linear")
    ])

    model.compile(loss="mae", 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_4(lr: float, 
            emb_out: int, 
            dense_1: int, dense_2: int, dense_3: int,
            dropout_1: float, dropout_2: float, dropout_3: float,
            n_dim: int) -> Sequential:
    model = Sequential([
    Embedding(152, emb_out, input_length=2),
    Flatten(),

    Dense(dense_1), # 128 - 1024
    BatchNormalization(),
    Activation("relu"),
    Dropout(dropout_1),

    Dense(dense_2),
    BatchNormalization(),
    Activation("relu"),
    Dropout(dropout_2),
        
    Dense(dense_3),
    BatchNormalization(),
    Activation("relu"),
    Dropout(dropout_3),

    Dense(n_dim, activation= "linear")
    ])

    model.compile(loss="mae", 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_5(lr: float, 
            emb_out: int,
            n_dim: int,
            dropout_1: float,
            dropout_2: float) -> Sequential:
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_1),
        Dense(1024),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_2),
        
        Dense(n_dim, activation= "linear")
    ])
    model.compile(loss=custom_mean_rowwise_rmse, 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_6(lr: float, 
            emb_out: int, 
            dense_1: int, dense_2: int,
            n_dim: int,
            dropout_1: float,
            dropout_2: float) -> Sequential:
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        BatchNormalization(),
        Dense(dense_1),
        Activation("relu"),
        
        Dropout(dropout_2),
        Dense(dense_2),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_2),
        
        Dense(n_dim, activation= "linear")
    ])
    model.compile(loss=custom_mean_rowwise_rmse, 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_7(lr: float, 
            emb_out: int, 
            dense_1: int, dense_2: int, dense_3: int, dense_4: int,
            dropout_1: float, dropout_2: float, dropout_3: float, dropout_4: float,
            n_dim: int) -> Sequential:
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        
        Dense(dense_1),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_1),
        
        Dense(dense_2),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_2),
        
        Dense(dense_3),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_3),

        Dense(dense_4),
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_4),
        
        Dense(n_dim, activation= "linear")
    ])

    model.compile(loss=custom_mean_rowwise_rmse, 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_8(lr: float, 
            emb_out: int, 
            dense_1: int, dense_2: int, dense_3: int,
            dropout_1: float, dropout_2: float, dropout_3: float,
            n_dim: int) -> Sequential:
    model = Sequential([
    Embedding(152, emb_out, input_length=2),
    Flatten(),

    Dense(dense_1),
    BatchNormalization(),
    Activation("relu"),
    Dropout(dropout_1),

    Dense(dense_2),
    BatchNormalization(),
    Activation("relu"),
    Dropout(dropout_2),
        
    Dense(dense_3),
    BatchNormalization(),
    Activation("relu"),
    Dropout(dropout_3),

    Dense(n_dim, activation= "linear")
    ])

    model.compile(loss=custom_mean_rowwise_rmse, 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


SIMPLE_MODEL = [model_7]
SIMPLE_MODEL_IDXS = ['7']

MODELS_STAGE_1 = [model_1, model_2, model_3, model_5, model_6, model_7, model_8]
MODELS_STAGE_1_IDXS = ['1', '2', '3', '5', '6', '7', '8']

MODELS_STAGE_2 = [model_1] + [model_2] * 3 + [model_3] * 2 + [model_4] * 2 + [model_5] * 3 + [model_6] * 3 + [model_7] * 2 + [model_8] * 4
MODELS_STAGE_2_IDXS = ['1a',
                       '2a', '2b', '2c',
                       '3a', '3b',
                       '4a', '4b',
                       '5a', '5b', '5c',
                       '6a', '6b', '6c',
                       '7a', '7b',
                       '8a', '8b', '8c', '8d']
