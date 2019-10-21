from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np
#import tensorflow as tf
#from tensorflow.keras import layers
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.naive_bayes import BernoulliNB
import feature_extraction


def LSVC(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data
    X = SelectKBest(k=17000).fit_transform(X, y)

    LSVC_clf = LinearSVC(penalty='l1', dual=False)
    #LSVC_clf = BaggingClassifier(base_estimator=LSVC_clf, max_samples=0.6, max_features=12000, n_estimators=24, random_state=1)

    print("before fitting")

    LSVC_cv = cross_val_score(LSVC_clf, X, y, cv=n)

    LSVC_clf.fit(X, y)

    LSVC_pred = LSVC_clf.predict(X)

    LSVC_acc = accuracy_score(y_pred=LSVC_pred, y_true=y)

    print("cross validation accuracy", LSVC_cv.mean())

    print("training accuracy", LSVC_acc)

    return LSVC_clf

def Bernoulli(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data
    X = SelectKBest(chi2, k=17000).fit_transform(X, y)

    Bernoulli_clf = BernoulliNB()
    Bernoulli_cv = cross_val_score(Bernoulli_clf , X, y, cv=n)

    Bernoulli_clf.fit(X, y)
    Bernoulli_pred = Bernoulli_clf.predict(X)
    Bernoulli_acc = accuracy_score(y_true=y, y_pred=Bernoulli_pred)

    print("acc of BNB(sklearn) is ", Bernoulli_acc)
    print("cv acc of BNB(sklearn) is ", Bernoulli_cv.mean())

    return Bernoulli_clf



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

    print("cv acc of LogReg is ", LogReg_cv.mean())

    print("acc of logreg is ", LogReg_acc)

    return LogReg_clf


def MultiNB(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data
    X = SelectKBest(chi2, k=17000).fit_transform(X, y)

    MultiNB_clf = MultinomialNB(alpha=0.22)

    MultiNB_cv = cross_val_score(MultiNB_clf, X, y, cv=n)

    MultiNB_clf.fit(X, y)

    MultiNB_pred = MultiNB_clf.predict(X)

    MultiNB_acc = accuracy_score(y_pred=MultiNB_pred, y_true=y)

    print("cv acc of multiNB is ", MultiNB_cv.mean())
    print("acc of multiNB is ", MultiNB_acc)

    return MultiNB_clf

def MultiNB_Kaggle():
    X, y, test, test_ID = feature_extraction.pre_process_comments()  # get the data
    k_best = SelectKBest(chi2, k=17000)
    X = k_best.fit_transform(X, y)
    MultiNB_clf = MultinomialNB(alpha=0.22)
    test_X = k_best.transform(test)

    MultiNB_clf.fit(X, y)
    MultiNB_pred = MultiNB_clf.predict(test_X)

    np.savetxt('predict.csv', np.array([test_ID, MultiNB_pred]).transpose(), delimiter=',', fmt='%s',header='Id, Category')
    return MultiNB_clf

def ensemble(n=5):
    X, y = feature_extraction.pre_process_comments()  # get the data

    X = SelectKBest(k=17000).fit_transform(X, y)

    LogReg_clf = LogisticRegression(penalty='l1', dual=False)
    MNB_clf = MultinomialNB(alpha=0.22)
    LinearSVC_clf = LinearSVC(penalty='l1', dual=False)

    Ensemble_clf = VotingClassifier(estimators=[('lg', LogReg_clf), ('mnb', MNB_clf)], voting='soft')

    Ensemble_cv = cross_val_score(Ensemble_clf, X, y, cv=n)

    Ensemble_clf.fit(X, y)

    score = Ensemble_clf.score(X, y)

    print("cv acc of ensemble is ", Ensemble_cv.mean())
    print("acc of ensemble is ", score)

    return Ensemble_clf


class NB:
    def __init__(self, x, y):
        x = np.array(x.toarray())
        self.x = x
        self.y = y.reshape(y.shape[0], 1)
        self.y_dict = dict()

        index = 0
        for y in self.y:
            if y[0] in self.y_dict:
                continue
            else:
                self.y_dict[y[0]] = index
                index += 1
            #        print(self.y_dict)
        # y_dict: this parameter is of type dict. The possible format would be like this
        # y_dict = {'funny' =1, 'nba' =2, 'movies' =3 .....} The 1 2 3 .... 20 are the indices
        # We use to store the frequency in data

        self.data_table = np.full((len(self.y_dict), self.x.shape[1] + 1), 0, dtype='f')
        self.bernoulli
        self.counting()
        #        print(self.data_table)
        # After counting(), data_table would be like in this form (Example for the First ROW)
        # [100, 21, 0 , 99 ............... 1000]
        # We've seen the index for 'funny' subreddit is 1. Then the first colomn stores the info
        # about the 'funny' subreddit. It means that in our input data. We've seen class 'funny'
        # 1000 times (last colomn). Featuer 1 appears 100 times, feature 2 times.
        self.frequency()

    #        print(self.data_table)
    # After requency(), the frequency for the appearence of each feature is calculated.
    # For the new data_table, the format could be like this (Example for the First ROW)
    # [0.1007983 = ||101/1002||, 0.021956, ..........., 0.05]
    # The last colomn means the class 'funny' takes 5% of our training set
    # Laplace smoothing is also done.

    def bernoulli(self):
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i, j] > 0:
                    self.x[i, j] = 1

    def counting(self):
        for i in range(self.x.shape[0]):
            index = self.y_dict[self.y[i, 0]]
            arr = np.random.rand(1, self.x.shape[1] + 1)
            arr[:, :-1] = self.x[i, :]
            arr[:, -1] = 1
            self.data_table[index, :] = self.data_table[index, :] + arr

    #    def frequency(self):
    #        total = sum(self.data_table[:,-1])
    #        for i in range(len(self.y_dict)):
    #            for j in range(self.x.shape[1]):
    #                self.data_table[i,j] = (self.data_table[i,j])/(self.data_table[i,-1])
    #            self.data_table[i,-1] = self.data_table[i,-1] / total

    def frequency(self):
        total = sum(self.data_table[:, -1])
        for i in range(len(self.y_dict)):
            for j in range(self.x.shape[1]):
                self.data_table[i, j] = (self.data_table[i, j] + 1) / (self.data_table[i, -1] + 2)
            self.data_table[i, -1] = self.data_table[i, -1] / total

    def predict(self, data_input):

        print("Input shape is: ", data_input.shape)
        result = np.full((data_input.shape[0], 1), 'Hello World')
        for i in range(data_input.shape[0]):
            max_prob = -1
            likely_class = ''
            for j in self.y_dict.keys():
                index = self.y_dict[j]
                prob = 0
                for k in range(self.x.shape[1]):
                    prob += data_input[i, k] * np.math.log(self.data_table[index, k])
                    + (1 - data_input[i, k]) * np.math.log(1 - self.data_table[index, k])
                prob = prob + np.math.log(self.data_table[index, -1])
                ## Estimate the probability based on the formula log(P(class = ?)) * sigma->log(P(Xi=??|class=?))
                if np.math.exp(prob) > max_prob:
                    max_prob = np.math.exp(prob)
                    likely_class = j
            print(likely_class)
            result[i, 0] = likely_class
        # Store the most likely class in result
        return result

