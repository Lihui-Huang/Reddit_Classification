from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import models.models as models
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder

X, y = feature_extraction.pre_process_comments()  # get the data

X = SelectKBest(chi2, k=17000).fit_transform(X, y)

LogReg_clf = LogisticRegression(penalty='l1', dual=False)
MNB_clf = MultinomialNB(alpha=0.3)
LinearSVC_clf = LinearSVC(penalty='l1', dual=False)

Ensemble_clf = VotingClassifier(estimators=[('lg', LogReg_clf), ('mnb', MNB_clf)], voting='soft')

Ensemble_cv = cross_val_score(Ensemble_clf, X, y, cv=3)

Ensemble_clf.fit(X, y)

score = Ensemble_clf.score(X, y)

print(score)






