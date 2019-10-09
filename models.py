from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import feature_extraction

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
    X, y = feature_extraction.pre_process_comments()  # get the data

    DecTree_clf = DecisionTreeClassifier()

    DecTree_cv = cross_val_score(DecTree_clf, X, y, cv=3)

    print(DecTree_cv.mean())

    DecTree_clf.fit(X, y)

    DecTree_pred = DecTree_clf.predict(X)

    DecTree_acc = accuracy_score(y_pred=DecTree_pred, y_true=y)

    print(DecTree_acc)

    return DecTree_clf