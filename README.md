# Disaster Response Pipeline Project

### Summary:

This project consists of a python-based web-app that automatically pre-processes and classifies text messages into different categories of disaster management. It also provides you with some insights into the model's performance.

In this project, I used supervised learning methods, and specifically used a Random Forest Classifier with Grid Search Cross Validation

### Files

There are three main files that you should be aware of:

1. ETL pipeline (process_data.py) - cleans, processes, and stores data for classification
2. ML pipeline (train_classifier.py) - trains, evaluates and save a model
3. Flask web app (run.py) - interactive web app 

### Requirements:
You will need to install the following python packages: 

* flask
* plotly
* sqlalchemy
* pandas
* numpy
* sklearn
* nltk

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
