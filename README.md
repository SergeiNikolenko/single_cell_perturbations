## General info
This is 3rd place solution in challenge Open Problems – Single-Cell Perturbations. 
The stage 1 doesn't use the pseudolabels, but the stage 2 does. The stage simple is related to fast, single model. The outputs are placed in submissions directory. 

### Useful links
 * A detailed solution https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/458750.
 * A required data https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data.

### Needed data
* de_train.parquet - a training data
* id_map.csv - a test data 
* sample_submission.csv - a sample of the submission in kaggle challenge
	
## Technologies
Project is created with:
* python 3.9
* tensorflow 2.12.0
* numpy 1.23.5
* pandas 2.0.3
* scikit-learn 1.3.2

## Hardware:
Ubuntu 18.04.6 LTS (256 GB boot disk)
Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz (8 cores)
8GB RAM

## Model training
To train simple model 
```python -m prepare_data --stage simple && python -m train --stage simple```  
This model (model_7_0.keras) is serialized and placed in models/simple/model_7, so training can be omitted.   
To train stage 1 models
```python -m prepare_data --stage stage_1 && python -m train --stage stage_1```  
To train stage 2 models 
```python -m repro --stage stage_1 && python -m prepare_data --stage stage_2 && python -m train --stage stage_2```  
In the stage 2 the id_map.csv has to be replaced by a new test set, due to creating the pseudolabels from the stage 1. 

### Prediction
Replace id_map.csv with new test set. To predict execute  
```python -m predict --stage {one of simple, stage_1, stage_2}```


### Assumptions
Clear all directories with models, except of models/simple directory, which contains a serialized model. 


## Setup and reproduction
* Download from challenge website and copy de_train.parquet, id_map.csv, sample_submission.csv files to data directory.
* Build docker image using ```sudo docker build -t env .```
* Run docker image ```sudo docker run --name container env```
* Copy predicted file to host ```sudo docker cp container:/app/submissions/stage_2_submission.csv ./submission.csv```.
