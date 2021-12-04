#!/usr/bin/bash
set -e

python3.8 -m venv --copies venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
python -m ipykernel install --user --name=erpinator

pip install pre-commit
pre-commit install
deactivate

# for plotly widget support
pip install --user jupyterlab plotly==4.14.3 "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly@4.14.3
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
