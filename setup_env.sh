#!/usr/bin/bash
set -e

python3.8 -m venv --copies venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=erpinator
deactivate

pip install --user pre-commit
pre-commit install

pip install --user jupyterlab
pip install --user plotly ipywidgets
# jupyter labextension install jupyterlab-plotly @jupyter-widgets/jupyterlab-manager plotlywidget
