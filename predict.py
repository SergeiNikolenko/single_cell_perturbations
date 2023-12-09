from collections import defaultdict
from pathlib import Path
import glob
from typing import List
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tensorflow import keras
from config.params_stage_1 import WEIGHTS_SIMPLE_MODEL, WEIGHTS_STAGE_1
from config.params_stage_2 import WEIGHTS_STAGE_2

from util import clip, custom_mean_rowwise_rmse, load_json_file, load_test_x, load_training_data, parse_stage, save_preds


def predict_single_model(x_test: np.array, y: np.array, model_path: Path) -> np.array:
    """
    Args:
        x_test - a test input
        y - labels of training dataset - needed for a truncated SVD
    Returns:
        preds - predictions in a numpy array
    """
    model = keras.models.load_model(model_path, 
                                    custom_objects={'custom_mean_rowwise_rmse': custom_mean_rowwise_rmse})
    preds = model.predict(x_test, batch_size=1)
    decomposition = TruncatedSVD(preds.shape[1])
    decomposition.fit(y)
    preds = decomposition.inverse_transform(preds)
    return preds


def load_weights(stage: str) -> List[float]:
    if stage == 'simple':
        weights = WEIGHTS_SIMPLE_MODEL
    elif stage == 'stage_1':
        weights = WEIGHTS_STAGE_1
    elif stage == 'stage_2':
        weights = WEIGHTS_STAGE_2
    return weights


def predict(stage: str) -> None:
    """
    Make and save predictions.
    Args:
        stage: one of 'simple', 'stage_1', 'stage_2'.
    """
    _, y = load_training_data(stage)
    weights = load_weights(stage)
    x_test = load_test_x(stage)
    root = Path(__file__).parent
    path = root / load_json_file(root / 'SETTINGS.json')['MODEL_CHECKPOINT_DIR']
    model_dir = str(root / path / stage)
    models = glob.glob(f"{model_dir}/**/*")
    groupped_models = defaultdict(list)
    for model_path in models:
        stem = Path(model_path).stem
        model_id = stem.split('_')[-2]
        groupped_models[model_id].append(model_path)
    predictions = []
    sorted_keys = sorted(list(groupped_models.keys()))
    for k in sorted_keys:
        temp_preds = []
        for p in groupped_models[k]:
            preds = predict_single_model(x_test, y, p)
            temp_preds.append(preds)
        preds = np.median(temp_preds, axis=0)
        predictions.append(preds)
    preds = np.sum([w * p for w, p in zip(weights, predictions)], axis=0) / sum(weights)
    preds = clip(preds)
    submission_name = stage + '_' + 'submission.csv'
    save_preds(preds, submission_name)

if __name__ == '__main__':
    stage = parse_stage()
    predict(stage)