# Fall-2021-CMPE-255-Project-Group-1

This is our group project for CMPE 255.

## Abstract

The general aim for this project is to read compare different classification metrics on a dataset of our choosing.

We combined the One million reddit jokes dataset and highlights from the CNN dailymail dataset and marked them as 1 for jokes and -1 for not jokes. (Everything from reddit was considered a joke and highlights from cnn/dailymail was considered to not be one).

The combined file is available at: [Google Drive Link](https://drive.google.com/drive/folders/1YNhdT8fcHVJrEFEoP6c913kB3gUGkDPs?usp=sharing)

## Description

This problem is a binary classification problem. After preprocessing the data we will test a variety of different classification models to determine which model works best. 

### Dimensionality Reduction
- TfIdf Vectorization
- Singluar value decomposition(SVD): [Link for data](https://drive.google.com/drive/folders/1_Ym2UIX5lv12EzSRTNZbjWxqNfrlxk4K?usp=sharing)
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
This program reads from the file [cleaned_data.csv](https://drive.google.com/drive/folders/1_Ym2UIX5lv12EzSRTNZbjWxqNfrlxk4K?usp=sharing). 
This file must be in the same directory as main.py. 
Alternatively, you can run the jupyter notebook [main.ipynb](main.ipynb) for the same results. This file takes text as input
and will classify the input as a joke or a headline.
