#!/bin/bash
python -m pip install --upgrade pip
pip install virtualenv
python3 -m venv venv_ml
source venv_ml/bin/activate
pip install -r requirements.txt
python setup.py build_ext --inplace
ipython kernel install --user --name=titanic_ml_kernel
jupyter notebook titanic_survival_logistic_regr.ipynb
