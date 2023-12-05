.ONESHELL:

SHELL=/bin/bash

CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

CONDA_ENV=kaggle

PWD := $(shell pwd)

# build env
build: conda-env-update pip-compile pip-sync

conda-env-update:
	conda env update --prune

# Compile exact pip packages
pip-compile:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	pip-compile -v requirements/dev.in

# Install pip packages
pip-sync:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	pip-sync requirements/dev.txt
