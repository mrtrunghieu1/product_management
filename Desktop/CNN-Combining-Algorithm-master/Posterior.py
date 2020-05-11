from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier


def Posterior(training, label, sample, flag, n_classes):
    mdl = PosteriorModel(training, label, flag, n_classes)
    P = PosteriorTest(mdl, sample)
    return P


def PosteriorTest(mdl, sample):
    P_predict = mdl.predict_proba(sample)
    return P_predict


def PosteriorModel(training, label, flag, n_classes):
    if flag == 0:
        if n_classes == 2:
            mdl = XGBClassifier(n_estimators=200, object='binary:logistic')
        else:
            mdl = XGBClassifier(n_estimators=200, object='multi:softmax')
        mdl.fit(training, label)
    elif flag == 1:
        mdl = GaussianNB()
        mdl.fit(training, label)
    elif flag == 2:
        neigh = KNeighborsClassifier(n_neighbors=5)
        mdl = neigh.fit(training, label)
    elif flag == 3:
        lr = LogisticRegression(solver='newton-cg')
        mdl = lr.fit(training, label)
    elif flag == 4:
        rf = RandomForestClassifier(n_estimators=200)
        mdl = rf.fit(training, label)
    elif flag == 5:
        neigh = KNeighborsClassifier(n_neighbors=20)
        mdl = neigh.fit(training, label)
    elif flag == 6:
        neigh = KNeighborsClassifier(n_neighbors=25)
        mdl = neigh.fit(training, label)
    elif flag == 7:
        neigh = KNeighborsClassifier(n_neighbors=50)
        mdl = neigh.fit(training, label)
    return mdl
