from util import load_json_file, load_training_data, train_nn
from pathlib import Path
from models import MODELS_STAGE_1
from params_stage_1 import WEIGHTS_STAGE_1, PARAMS_STAGE_1
from util import clip, load_test_data, load_train_data, predict, preprocess_data, save_preds

REPS = 10

def train_stage_1():
    x, y =  load_training_data('stage_1')
    model_idxs = [1, 2, 3, 5, 6, 7, 8]
    path = Path(__file__).parent / 'SETTINGS.json'
    models_dir = Path(__file__).parent / load_json_file(path)['MODEL_CHECKPOINT_DIR']
    for model_id, model_costructor, model_params in zip(model_idxs, MODELS_STAGE_1, PARAMS_STAGE_1):
        for rep in range(REPS):
            model_path = models_dir / 'stage_1' / f"model_{model_id}" /  f"model_{model_id}_{rep}.keras"
            train_nn(x, y, model_costructor, model_params, model_path)


if __name__ == '__main__':   
    train_stage_1()