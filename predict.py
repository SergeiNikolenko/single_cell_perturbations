from collections import defaultdict
from pathlib import Path
import glob
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tensorflow import keras
from params_stage_1 import WEIGHTS_STAGE_1

from util import clip, custom_mean_rowwise_rmse, load_json_file, load_test_x, load_training_data, save_preds

def predict_stage_1():
    _, y = load_training_data('stage_1')
    x_test = load_test_x('stage_1')
    root = Path(__file__).parent
    path = root / load_json_file(root / 'SETTINGS.json')['MODEL_CHECKPOINT_DIR']
    model_dir = str(root / path / 'stage_1')
    models = glob.glob(f"{model_dir}/**/*")
    groupped_models = defaultdict(list)
    for model_path in models:
        stem = Path(model_path).stem
        model_id = stem.split('_')[-2]
        groupped_models[model_id].append(model_path)
    
    predictions = []
    for k in ['1', '2', '3', '5', '6', '7', '8']:
        temp_preds = []
        for p in groupped_models[k]:
        #p = groupped_models['7'][0]
            model = keras.models.load_model(p, custom_objects={'custom_mean_rowwise_rmse': custom_mean_rowwise_rmse})
            preds = model.predict(x_test, batch_size=1)
            decomposition = TruncatedSVD(preds.shape[1])
            decomposition.fit(y)
            preds = decomposition.inverse_transform(preds)
            temp_preds.append(preds)
            #print(temp_preds)
        preds = np.median(temp_preds, axis=0)
        #print(preds.shape)
        predictions.append(preds)
    weights = WEIGHTS_STAGE_1
    preds = np.sum([w * p for w, p in zip(weights, predictions)], axis=0) / sum(weights)
    preds = clip(preds)
    save_preds(preds, 'simple.csv')
    #print(preds.shape)
    #print(preds)
    #

if __name__ == '__main__':
    predict_stage_1()
