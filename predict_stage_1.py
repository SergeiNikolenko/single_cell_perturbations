from pathlib import Path
from models import MODELS_STAGE_1
from params_stage_1 import WEIGHTS_STAGE_1, PARAMS_STAGE_1
from util import clip, load_test_data, load_train_data, predict, preprocess_data, save_preds


def prepare_data(stage):
    train_df = load_train_data()
    test_df = load_test_data()
    x, y, x_test = preprocess_data(train_df, test_df)





REPS = 10

def predict_stage_1():
    """
    Save pseudolabels
    """
    train_df = load_train_data()
    test_df = load_test_data()
    x, y, x_test = preprocess_data(train_df, test_df)
    model_dir = Path(__file__).parent / 'data'
    model_dir.mkdir(parents=True, exist_ok=True)
    preds = predict(x, y, x_test, MODELS_STAGE_1, PARAMS_STAGE_1, WEIGHTS_STAGE_1, reps=REPS)
    preds = clip(preds)
    save_preds(preds, 'submission_stage_1.csv')

    #save data
    # load data
    #save models
    # predict


if __name__ == '__main__':
    predict_stage_1()


