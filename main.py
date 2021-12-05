#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def initialize():
    print("Loading Data...", end='\r')
    df = pd.read_csv("cleaned_data.csv")
    df.drop(columns=df.columns.values[0:2], inplace=True)
    df.dropna(inplace=True)
    df_jokes = df[df['joke'] > 0]
    df_non_jokes = df[df['joke'] < 0]
    df = pd.concat([df_jokes.sample(25000, random_state=42),
                    df_non_jokes.sample(25000, random_state=42)])

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned text'], df['joke'], test_size=0.2, random_state=42)
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("Building Vocab from training data...", end='\r')
    X_train = [doc.split() for doc in X_train]
    X_test = [doc.split() for doc in X_test]
    w2v = Word2Vec(size=200, window=5, min_count=4,
                   workers=8)  # vector_size = size
    # w2v = Word2Vec(vector_size=200, window=5, min_count=4, workers=8) # vector_size = size
    k = w2v.build_vocab(X_train)
    w2v.train(X_train, total_examples=len(X_train), epochs=32)
    vectors = w2v.wv

    print("Processing training data...               ", end='\r')
    
    new_X_train = []
    train_size = len(X_train)
    num = 0
    for words in X_train:
        if(num % 5000 == 0):
            print("Training: %d/%d          " %(num,train_size), end='\r')
            
        temp = []
        for word in words:
            if word in vectors:
                temp.append(vectors[word])
        if temp:
            doc_vector = np.mean(temp, axis=0).tolist()
        else:
            doc_vector = [np.nan] * 200

        new_X_train.append(doc_vector)
        num += 1
        
    X_train = pd.DataFrame(new_X_train)
    
    print("Processing testing data...   ",end='\r')
    new_X_test = []
    num = 0
    test_size = len(X_test)
    for words in X_test:
        if(num % 5000 == 0):
            print("Testing: %d/%d          " %(num,test_size), end='\r')
        
        temp = []
        for word in words:
            if word in vectors:
                temp.append(vectors[word])
        if temp:
            doc_vector = np.mean(temp, axis=0).tolist()
        else:
            doc_vector = [np.nan] * 200

        new_X_test.append(doc_vector)
        num += 1
        
    X_test = pd.DataFrame(new_X_test)

    X_train['label'] = y_train
    X_test['label'] = y_test

    X_train = X_train.dropna()
    y_train = X_train['label']
    drop_labels = ['label']
    X_train = X_train.drop(columns=drop_labels)
    y_train = np.array([max(i, 0) for i in y_train])

    X_test = X_test.dropna()
    y_test = X_test['label']
    X_test = X_test.drop(columns=drop_labels)
    y_test = np.array([max(i, 0) for i in y_test])
    

    print("Building neural network....       ", end='\r')
    model = Sequential([
        Dense(2048, activation='relu', input_shape=(200,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print("Training neural network....       ", end='\r')
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=100, callbacks=[es], verbose=0)
    
    return vectors, model


# In[3]:


def process_text(vectors, input_text):
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_text)
    
    tokens = [w.lower() for w in tokens]
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    input_text = ' '.join(words)
    
    new_X_train = []
    num = 0
    for words in [input_text.split()]:
        temp = []
        for word in words:
            if word in vectors:
                temp.append(vectors[word])
        if temp:
            doc_vector = np.mean(temp, axis=0).tolist()

        new_X_train.append(doc_vector)
        num += 1
        
    X_train = pd.DataFrame(new_X_train).dropna()
    
    return X_train


# In[4]:


def main():
    vectors, model = initialize()
    
    count = 0
    while count < 500:
        input_text = input("Tell a joke (type e to exit): ")
        if not input_text:
            input_text = ""
            print("Empty Input")
        elif input_text.lower() in ['e','exit']:
            print("Exiting...")
            break
        else:
            embedding = process_text(vectors, input_text)
            pred = model.predict(embedding)
            pred = np.round(pred)[0,0] 

            if pred > 0:
                print("Joke\n")
            else:
                print("Headline\n")
        
        count += 1


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




