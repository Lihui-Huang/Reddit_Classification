
# coding: utf-8

# In[1]:


import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize  
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
import re

#extract the original text
def pre_process_comments():
    with open("reddit_train.csv", 'r') as f:
            reddit_comments = list(csv.reader(f, delimiter=","))
    #headers = np.array(reddit_comments[0])
    #ID = np.array(reddit_comments).transpose()[0][1:]
    comments = np.array(reddit_comments).transpose()[1][1:]
    subreddits = np.array(reddit_comments).transpose()[2][1:]

    stop = stopwords.words('english')
    lemmer=WordNetLemmatizer()
    for i in range(len(comments)):
        no_stop_words = []
        for words in word_tokenize(comments[i]):
            if words not in stop:
                no_stop_words.append(words)
        new_comment= ' '.join(lemmer.lemmatize(word) for word in no_stop_words)
        comments[i] = new_comment

    """
    vectorizer = CountVectorizer(stop_words='english')
    X_sparse = vectorizer.fit_transform(comments)
    X = np.array(X_sparse.toarray())

    X_frequency = np.sum(X, axis=0)
    delete_features = []
    for i in range(len(X_frequency)):
        if X_frequency[i] <= 1:
            delete_features.append(i)
    delete_X = np.delete(X, delete_features, 1)
    """
    tfidf = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 1))
    X_sparse = tfidf.fit_transform(comments)
    X = np.array(X_sparse.toarray())

    with open("reddit_test.csv", 'r') as f:
            reddit_test_comments = list(csv.reader(f, delimiter=","))
    test_ID = np.array(reddit_test_comments).transpose()[0][1:]
    test_comments = np.array(reddit_test_comments).transpose()[1][1:]
    Test_sparse = tfidf.transform(test_comments)
    Test = np.array(Test_sparse.toarray())


    #np.savetxt('X.npy', X, fmt='%s')
    #np.savetxt('y.npy', subreddits, fmt='%s')

    return X, subreddits, Test, test_ID

#after stop words and lemmatization
#remove the words that appear very rarely
"""
X_frequency = np.sum(X, axis=0)
np.sort(X_frequency)[::-1][90:100]

np.sort(X_frequency)[::-1][190:200]

#there are about 35000 features that have frequency 1
#there are about 10000 featrues that have frequency 2
delete_features = []
for i in range(len(X_frequency)):
    if X_frequency[i] <= 1:
        delete_features.append(i)
delete_X = np.delete(bache_X, delete_features, 1)

delete_X.shape
"""

