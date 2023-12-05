## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This is 3rd place solution in challenge Open Problems – Single-Cell Perturbations. 

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

## Setup
To run this project:
* Download from challenge website and copy de_train.parquet, id_map.csv, sample_submission.csv files to data dictionary
* Build docker image using ```sudo docker build -t env .```
* Run docker image ```sudo docker run --name container env```
* Copy predicted file to host ```sudo docker cp container:/app/data/submission.csv ./submission.csv```.