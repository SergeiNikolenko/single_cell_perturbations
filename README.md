## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This is 3rd place solution in challenge Open Problems – Single-Cell Perturbations. 
Link to a detailed solution https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/458750.
Link to a required data https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data.
### Needed data
* de_train.parquet - training data
* id_map.csv - test data - prediction is made for it.
* sample_submission.csv - sample of submission in kaggle challange
	
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
* Run docker image ```sudo docker run env```
* Copy predicted file to host ```sudo docker cp env:/app/data/submmision.csv ./submission.csv```.