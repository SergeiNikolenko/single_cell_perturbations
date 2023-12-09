import argparse
from util import load_json_file, load_training_data, parse_stage, train_nn
from pathlib import Path
from models import MODELS_STAGE_1, MODELS_STAGE_1_IDXS, MODELS_STAGE_2, MODELS_STAGE_2_IDXS, SIMPLE_MODEL, SIMPLE_MODEL_IDXS
from config.params_stage_1 import PARAMS_STAGE_1, SIMPLE_MODEL_PARAMS
from config.params_stage_2 import PARAMS_STAGE_2


REPS = 10

def load_config(stage: str):
    """
    Args:
        stage: one of simple, stage_1, stage_2
    Returns:
        models - a list of model constructors
        params - a list of hiperparameters
        model_idxs - a list of model idxs
        reps - number of repeated training.
    """
    if stage == 'stage_1':
        models = MODELS_STAGE_1
        params = PARAMS_STAGE_1
        model_idxs = MODELS_STAGE_1_IDXS
        reps = REPS
    elif stage == 'stage_2':
        models = MODELS_STAGE_2
        params = PARAMS_STAGE_2
        model_idxs = MODELS_STAGE_2_IDXS
        reps = REPS
    elif stage =='simple':
        models = SIMPLE_MODEL
        params = SIMPLE_MODEL_PARAMS
        model_idxs = SIMPLE_MODEL_IDXS
        reps = 1
    return models, params, model_idxs, reps


def train(stage: str) -> None:
    """
    Train and save models for the stage.
    Args:
        stage: one of simple, stage_1, stage_2
    """
    x, y =  load_training_data(stage)
    models, params, model_idxs, reps = load_config(stage)
    path = Path(__file__).parent / 'SETTINGS.json'
    models_dir = Path(__file__).parent / load_json_file(path)['MODEL_CHECKPOINT_DIR']
    for model_id, model_costructor, model_params in zip(model_idxs, models, params):
        for rep in range(reps):
            model_path = models_dir / stage / f"model_{model_id}" /  f"model_{model_id}_{rep}.keras"
            print("Model path", model_path)
            train_nn(x, y, model_costructor, model_params, model_path)


if __name__ == '__main__':
    stage = parse_stage()
    train(stage)