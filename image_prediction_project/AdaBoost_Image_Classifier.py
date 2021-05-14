from sklearn.ensemble import AdaBoostClassifier
import Preprocessing as pre
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pickle
import cv2

Pkl_Filename = "ADA_MODEL.pkl"
x,y = pre.read_file()
estimators = [5,10,20,40,60,80]
learning_rates = [0.1, 0.3, 0.6, 0.8, 1]
fold = [5,10,15,20]

def AdaBoost_AccuracyScore():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    x_train_flat_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train_flat, y, train_size=0.8,
                                                                                 stratify=y)
    for n in estimators:
        for l in learning_rates:
            ada_multiclass = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=n,learning_rate=l))
            clf = ada_multiclass.fit(x_train_flat_finale, y_train)
            prediction = clf.predict(x_train_flat_finale)
            print("Accuratezza Training set di Adaboost con stimatori : " + str(n) + " con learning rate " + str(l) + " : " + str(
                accuracy_score(y_train, prediction)))
            prediction2 = clf.predict(x_test)
            print("Accuratezza Test set di Adaboost con stimatori : " + str(n) + " con learning rate " + str(l) + " : " + str(
                accuracy_score(y_test, prediction2)))

def AdaBoost_CV():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    for n in estimators:
        ada_multiclass = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=n))
        clf = ada_multiclass.fit(x_train_flat, y)
        for f in fold:
            cross_value = model_selection.cross_val_score(clf, x_train_flat, y, cv=f)
            print("Accuratezza con stimatori = " + str(n) + " con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

def save_model():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    ada_multiclass = OneVsRestClassifier(estimator=AdaBoostClassifier(n_estimators=40))
    clf = ada_multiclass.fit(x_train_flat, y)
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)




