from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import feature_extraction
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def LogReg(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data

    LogReg_clf = LogisticRegression()

    LogReg_cv = cross_val_score(LogReg_clf, X, y, cv=n)

    LogReg_clf.fit(X, y)

    LogReg_pred = LogReg_clf.predict(X)

    LogReg_acc = accuracy_score(y_pred=LogReg_pred, y_true=y)

    print(LogReg_cv.mean())
    print(LogReg_acc)

    return LogReg_clf

def DecTree(n=5):
    X, y, test_X, test_ID = feature_extraction.pre_process_comments()  # get the data

    DecTree_clf = DecisionTreeClassifier()

    DecTree_cv = cross_val_score(DecTree_clf, X, y, cv=3)

    print(DecTree_cv.mean())

    DecTree_clf.fit(X, y)

    DecTree_pred = DecTree_clf.predict(test_X)

    DecTree_acc = accuracy_score(y_pred=DecTree_pred, y_true=y)

    print(DecTree_acc)



    return DecTree_clf

def MultiNB(n=5):
    X, y, test_X, test_ID = feature_extraction.pre_process_comments()

    MultiNB_clf = MultinomialNB()

    #MultiNB_cv = cross_val_score(MultiNB_clf, X, y, cv=n)

    MultiNB_clf.fit(X, y)

    MultiNB_pred = MultiNB_clf.predict(test_X)

    #MultiNB_acc = accuracy_score(y_pred=MultiNB_pred, y_true=y)

    #print(MultiNB_cv.mean())
    #print(MultiNB_acc)

    np.savetxt('predict.csv', np.array([test_ID, MultiNB_pred]).transpose(), delimiter=',', fmt='%s',
               header='Id, Category')

    return MultiNB_clf


model = MultiNB()

