# risk-assessment

In this repo, I'm working on creating, deploying, and monitoring a risk assessment model that estimates the attrition risk for a company's client list. Furthermore, setting up processes to re-train, re-deploy, monitor and report on the machine learning model.

## Setup local environment

- `conda create --name risk-assessment python=3.8`
- `pip install -r requirements.txt`


## Run the project

### data ingestion

- `python src/ingestion.py`

### Model training

- `python src/training.py`

### scoring

`python src/scoring.py`

### deployment

`python src/deployment.py`

### diagnostics

`python src/diagnostics.py`

## API

- `python src/app.py`
- `python src/wsgi.py`

## reporting

`python src/reporting.py`

