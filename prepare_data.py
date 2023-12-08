from pathlib import Path
import pandas as pd
from util import load_json_file, load_pseudolabels, load_test_data, load_train_data, preprocess_data

import numpy as np


def prepare_data(stage: int):
    if stage == 1:
        prepare_data_stage_1()
    elif stage == 2:
        prepare_data_stage_2()


def prepare_data_stage_1():
    train_df = load_train_data()
    test_df = load_test_data()
    x, y, test_x = preprocess_data(train_df, test_df)
    root = Path(__file__).parent
    path =  root / 'SETTINGS.json'
    paths = load_json_file(path)
    train_data_clean_path = root / paths['TRAIN_DATA_CLEAN_PATH'] / 'stage_1'
    test_data_clean_path = root / paths['TEST_DATA_CLEAN_PATH'] / 'stage_1'
    train_data_clean_path.mkdir(parents=True, exist_ok=True)
    test_data_clean_path.mkdir(parents=True, exist_ok=True)
    np.save(train_data_clean_path / 'x.npy', x)
    np.save(train_data_clean_path / 'y.npy', y)
    np.save(test_data_clean_path / 'test_x.npy', test_x)


def prepare_data_stage_2():
    train_df = load_train_data()
    pseudolabel = load_pseudolabels()
    train_df = pd.concat([train_df, pseudolabel]).reset_index(drop=True)
    test_df = load_test_data()
    x, y, test_x = preprocess_data(train_df, test_df)
    root = Path(__file__).parent
    path =  root / 'SETTINGS.json'
    paths = load_json_file(path)
    train_data_clean_path = root / paths['TRAIN_DATA_CLEAN_PATH'] / 'stage_2'
    test_data_clean_path = root / paths['TEST_DATA_CLEAN_PATH'] / 'stage_2'
    train_data_clean_path.mkdir(parents=True, exist_ok=True)
    test_data_clean_path.mkdir(parents=True, exist_ok=True)
    np.save(train_data_clean_path / 'x.npy', x)
    np.save(train_data_clean_path / 'y.npy', y)
    np.save(test_data_clean_path / 'test_x.npy', test_x)


if __name__ == '__main__':
    prepare_data_stage_1()