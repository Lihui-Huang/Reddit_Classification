from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
import feature_extraction


def LSVC(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data
    X = SelectKBest(chi2, k=17000).fit_transform(X, y)

    LSVC_clf = LinearSVC(penalty='l1', dual=False)

    print("before fitting")

    LSVC_cv = cross_val_score(LSVC_clf, X, y, cv=n)

    LSVC_clf.fit(X, y)

    LSVC_pred = LSVC_clf.predict(X)

    LSVC_acc = accuracy_score(y_pred=LSVC_pred, y_true=y)

    print("cross validation accuracy", LSVC_cv.mean())

    print("training accuracy", LSVC_acc)

    return LSVC_clf


def LogReg(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data
    X = SelectKBest(chi2, k=17000).fit_transform(X, y)
    '''
    LogReg_clf = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, multi_class='crammer_singer', max_iter=5000), threshold="0.8*mean", max_features=50000)),
        ('classification', LogisticRegression(penalty='l1', solver='saga'))
    ])
    '''

    LogReg_clf = LogisticRegression(penalty='l1', dual=False)

    print("before fitting")

    LogReg_cv = cross_val_score(LogReg_clf, X, y, cv=n)

    LogReg_clf.fit(X, y)

    LogReg_pred = LogReg_clf.predict(X)

    LogReg_acc = accuracy_score(y_pred=LogReg_pred, y_true=y)

    print(LogReg_cv.mean())

    print(LogReg_acc)

    return LogReg_clf


def MultiNB(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data
    X = SelectKBest(chi2, k=17000).fit_transform(X, y)

    MultiNB_clf = MultinomialNB(alpha=0.15)

    MultiNB_cv = cross_val_score(MultiNB_clf, X, y, cv=n)

    MultiNB_clf.fit(X, y)

    MultiNB_pred = MultiNB_clf.predict(X)

    MultiNB_acc = accuracy_score(y_pred=MultiNB_pred, y_true=y)

    print(MultiNB_cv.mean())
    print(MultiNB_acc)

    #np.savetxt('predict.csv', np.array([test_ID, MultiNB_pred]).transpose(), delimiter=',', fmt='%s',header='Id, Category')

    return MultiNB_clf