#!/usr/bin/bash
set -e

python3.8 -m venv --copies venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=erpinator
deactivate

pip install --user pre-commit
pre-commit install

# for plotly widget support
pip install --user jupyterlab==1.0.0 plotly ipywidgets
jupyter labextension install jupyterlab-plotly plotlywidget
