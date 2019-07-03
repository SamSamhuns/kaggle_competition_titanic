#!/bin/bash
pip install virtualenv
python3 -m venv venv_ml
source venv_ml/bin/activate
pip install -r requirements.txt
ipython kernel install --user --name=titanic_ml_kernel
jupyter notebook titanic_survival_logistic_regr.ipynb
