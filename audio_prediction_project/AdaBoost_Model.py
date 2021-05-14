from sklearn.ensemble import AdaBoostClassifier
import Preprocessing_Audio as pre
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import minmax_scale

Pkl_Filename = "ADA_MODEL.pkl"

x,y = pre.read_dataset()
estimators = [5,10,20,40,60,80]
learning_rates = [0.1, 0.3, 0.6, 0.8, 1]
fold = [5,10,15,20]

x = minmax_scale(x)


def AdaBoost_AccuracyScore():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,stratify=y)

    for n in estimators:
        for l in learning_rates:
            ada_multiclass = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=n,learning_rate=l))
            clf = ada_multiclass.fit(x_train, y_train)
            prediction = clf.predict(x_train)
            print("Accuratezza Training set di Adaboost con stimatori : " + str(n) + " con learning rate " + str(l) + " : " + str(
                accuracy_score(y_train, prediction)))
            prediction2 = clf.predict(x_test)
            print("Accuratezza Test-set di Adaboost con stimatori : " + str(n) + " con learning rate " + str(l) + " : " + str(
                accuracy_score(y_test, prediction2)))

def AdaBoost_CV():
    for n in estimators:
        ada_multiclass = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=n))
        clf = ada_multiclass.fit(x, y)
        for f in fold:
            cross_value = model_selection.cross_val_score(clf, x, y, cv=f)
            print("Accuratezza con stimatori = " + str(n) + " con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

def save_model():
    ada_multiclass = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=80))
    clf = ada_multiclass.fit(x, y)
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)