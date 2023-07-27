## Abstract

The general aim for this project is to read compare different classification metrics on a dataset of our choosing.

We combined the One million reddit jokes dataset and highlights from the CNN dailymail dataset and marked them as 1 for jokes and -1 for not jokes. (Everything from reddit was considered a joke and highlights from cnn/dailymail was considered to not be one).

## Description

This problem is a binary classification problem. After preprocessing the data we will test a variety of different classification models to determine which model works best. 

### Dimensionality Reduction
- TfIdf Vectorization
- Singluar value decomposition(SVD)
- Word2Vec Processing

### Models
The models we will test are:

- Decision Tree
- Logistic Regression
- Neural Network
- Random Forest Classifier
- SVC (Support Vector Classifier)
- XGBoost

We will use the same metrics for each model to compare them.

### Classification Metrics
The classification metrics used to compare models will be:

- F1-Score
- Accuracy
- Precision
- Recall

### Running main.py
This is a command line program, so you can run it using the command `python3 main.py`
This program reads from the file 
This file must be in the same directory as main.py. 
Alternatively, you can run the jupyter notebook for the same results. This file takes text as input
and will classify the input as a joke or a headline.
