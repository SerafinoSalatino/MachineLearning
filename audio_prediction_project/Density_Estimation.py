from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
import Preprocessing_Audio as pre
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

Pkl_Filename = "Naive_Bayes.pkl"
KNN_Filename = "KNN.pkl"
fold = [5,10,15,20]

x,y = pre.read_dataset()

x =minmax_scale(x)


def Parametric():
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,
    #                                                                           stratify=y)

    print("GaussianNB")
    clf = GaussianNB()
    clf.fit(x, y)
    for f in fold:
        cross_value = model_selection.cross_val_score(clf, x, y, cv=f)
        print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

    print()
    print("MultinomialNB")
    clf = MultinomialNB()
    clf.fit(x, y)
    for f in fold:
        cross_value = model_selection.cross_val_score(clf, x, y, cv=f)
        print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

    print()
    print("ComplementNB")
    clf = ComplementNB()
    clf.fit(x, y)
    for f in fold:
        cross_value = model_selection.cross_val_score(clf, x, y, cv=f)
        print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

    print()
    print("BernoulliNB")
    clf = BernoulliNB(binarize=0.8)
    clf.fit(x, y)
    for f in fold:
        cross_value = model_selection.cross_val_score(clf, x, y, cv=f)
        print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))


def KNN():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,
                                                                        stratify=y)

    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(x_train, y_train)

    prediction_train = neigh.predict(x_train)
    print(accuracy_score(y_train, prediction_train))

    prediction_test = neigh.predict(x_test)
    print(accuracy_score(y_test, prediction_test))

    neigh.fit(x, y)

    for f in fold:
        cross_value = model_selection.cross_val_score(neigh, x, y, cv=f)
        print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

def save_model_Naive_Bayes():
    clf = BernoulliNB(binarize=0.8)
    clf.fit(x, y)
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)


def save_model_KNN():
    neigh = KNeighborsClassifier(n_neighbors=5,weights='distance')
    neigh.fit(x, y)
    with open(KNN_Filename, 'wb') as file:
        pickle.dump(neigh, file)