import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

from util import custom_mean_rowwise_rmse


def model_1(lr, 
            emb_out,
            n_dim):
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



def model_2(lr, 
            emb_out, 
            dense_1, dense_2, 
            dropout_1, dropout_2,
            n_dim):
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        
        Dense(dense_1), # 64 - 512
        BatchNormalization(),
        Activation("relu"),
        
        Dropout(dropout_1), # 256 - 2048
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


def model_3(lr, 
            emb_out, 
            dense_1, dense_2, dense_3, dense_4,
            dropout_1, dropout_2, dropout_3, dropout_4,
            n_dim):
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        
        Dense(dense_1), # 128 - 1024
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_1),
        
        Dense(dense_2), # 64 - 512
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_2),
        
        Dense(dense_3), # 32 - 256
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_3),

        Dense(dense_4), # 16 - 512
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_4),
        
        Dense(n_dim, activation= "linear")
    ])

    model.compile(loss="mae", 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_4(lr, 
            emb_out, 
            dense_1, dense_2, dense_3,
            dropout_1, dropout_2, dropout_3,
            n_dim):
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


def model_5(lr, 
            emb_out,
            n_dim,
            dropout_1,
            dropout_2):
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


def model_6(lr, 
            emb_out, 
            dense_1, dense_2,
            n_dim,
            dropout_1,
            dropout_2):
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        BatchNormalization(),
        Dense(dense_1), # 64 - 512
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


def model_7(lr, 
            emb_out, 
            dense_1, dense_2, dense_3, dense_4,
            dropout_1, dropout_2, dropout_3, dropout_4,
            n_dim):
    model = Sequential([
        Embedding(152, emb_out, input_length=2),
        Flatten(),
        
        Dense(dense_1), # 128 - 1024
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_1),
        
        Dense(dense_2), # 64 - 512
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_2),
        
        Dense(dense_3), # 32 - 256
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_3),

        Dense(dense_4), # 16 - 512
        BatchNormalization(),
        Activation("relu"),
        Dropout(dropout_4),
        
        Dense(n_dim, activation= "linear")
    ])

    model.compile(loss=custom_mean_rowwise_rmse, 
                    optimizer=Adam(learning_rate=lr),
                 metrics=[custom_mean_rowwise_rmse])
    return model


def model_8(lr, 
            emb_out, 
            dense_1, dense_2, dense_3,
            dropout_1, dropout_2, dropout_3,
            n_dim):
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


MODELS_STAGE_1 = [model_1, model_2, model_3, model_5, model_6, model_7, model_8]
MODELS_STAGE_2 = [model_1] + [model_2] * 3 + [model_3] * 2 + [model_4] * 2 + [model_5] * 3 + [model_6] * 3 + [model_7] * 2 + [model_8] * 4