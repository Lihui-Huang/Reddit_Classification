
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
from sklearn.decomposition import PCA

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


    tfidf = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 1))
    X = tfidf.fit_transform(comments)

    #used to process test data
    """
    with open("reddit_test.csv", 'r') as f:
            reddit_test_comments = list(csv.reader(f, delimiter=","))
    test_ID = np.array(reddit_test_comments).transpose()[0][1:]
    test_comments = np.array(reddit_test_comments).transpose()[1][1:]
    Test_sparse = tfidf.transform(test_comments)
    Test = np.array(Test_sparse.toarray())
    """

    return X, subreddits #Test, test_ID
    #if we want to submmit a csv prediction to kaggle, just simply delete the python comments
