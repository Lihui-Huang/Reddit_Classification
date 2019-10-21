from nltk import WordNetLemmatizer
from nltk import word_tokenize
import feature_extraction
import models as model
from sklearn.metrics import accuracy_score
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from nltk.corpus import stopwords
import csv
import numpy as np

logreg = model.LogReg()

NB = model.Bernoulli()

SVC = model.LSVC()

Ensemble = model.ensemble()

MultiNB = model.MultinomialNB()

#Kaggle_predict = model.MultiNB_Kaggle()

X, y = feature_extraction.pre_process_comments()  # get the data

BNB = model.NB(X, y)

BNB_pred = BNB.predict(X)

MultiNB_acc = accuracy_score(y_pred=BNB_pred, y_true=y)





