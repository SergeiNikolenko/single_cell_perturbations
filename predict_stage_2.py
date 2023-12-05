from pathlib import Path

import pandas as pd

from models import MODELS_STAGE_2
from params_stage_2 import WEIGHTS_STAGE_2, PARAMS_STAGE_2
from util import clip, load_pseudolabels, load_test_data, load_train_data, predict, preprocess_data, save_preds


REPS = 10

def predict_stage_2() -> None:
    """
    Saves a final submission. 
    """
    train_df = load_train_data()
    pseudolabel = load_pseudolabels()
    train_df = pd.concat([train_df, pseudolabel]).reset_index(drop=True)
    test_df = load_test_data()
    x, y, x_test = preprocess_data(train_df, test_df)
    model_dir = Path(__file__).parent / 'data'
    model_dir.mkdir(parents=True, exist_ok=True)
    preds = predict(x, y, x_test, MODELS_STAGE_2, PARAMS_STAGE_2, WEIGHTS_STAGE_2, reps=REPS)
    preds = clip(preds)
    save_preds(preds, 'submission.csv')


if __name__ == '__main__':
    predict_stage_2()