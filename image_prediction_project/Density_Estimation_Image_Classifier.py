from sklearn.neighbors import KNeighborsClassifier
import Preprocessing as pre
import numpy as np
from sklearn.decomposition import PCA
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import pickle

Pkl_Filename = "Naive_Bayes.pkl"
KNN_Filename = "KNN.pkl"
x,y = pre.read_file()

fold = [5,10,15,20]

y_ok = []
for i in y:
    if i == "shirts":
        y_ok.append(0)
    if i == "shorts":
        y_ok.append(1)
    if i == "sunglasses":
        y_ok.append(2)
    if i == "wallets":
        y_ok.append(3)
y_ok = np.array(y_ok)

def NaiveBayes_PCA():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    pca = PCA(n_components=2)
    pca.fit(x_train_flat)
    x_train = pca.transform(x_train_flat)
    x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_ok, train_size=0.8,
                                                                               stratify=y_ok)

    clf = GaussianNB()
    clf.fit(x_train_finale, y_train)
    predictions = clf.predict(x_train_finale)
    print(accuracy_score(y_train, predictions))

    predictions2 = clf.predict(x_test)
    print(accuracy_score(y_test, predictions2))

    colors = ['red', 'green', 'blue', 'purple']

    fig = plt.figure(1, figsize=(8, 8))
    plt.scatter(x_train_finale[:, 0], x_train_finale[:, 1], c=predictions,
                cmap=matplotlib.colors.ListedColormap(colors))
    fig = plt.figure(2, figsize=(8, 8))
    plt.scatter(x_train_finale[:, 0], x_train_finale[:, 1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

    fig = plt.figure(3, figsize=(8, 8))
    plt.scatter(x_test[:, 0], x_test[:, 1], c=predictions2, cmap=matplotlib.colors.ListedColormap(colors))
    fig = plt.figure(4, figsize=(8, 8))
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


def NaiveBayes():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train_flat, y_ok, train_size=0.8,
                                                                               stratify=y_ok)

    clf = GaussianNB()
    clf.fit(x_train_finale, y_train)
    predictions = clf.predict(x_train_finale)
    print(accuracy_score(y_train, predictions))

    predictions2 = clf.predict(x_test)
    print(accuracy_score(y_test, predictions2))

    for f in fold:
        cross_value = model_selection.cross_val_score(clf, x_train_flat, y, cv=f)
        print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))


def KNN():
  x_train = x / 255
  x_train_flat = np.array([i.flatten() for i in x_train])
  x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train_flat, y_ok, train_size=0.8,
                                                                               stratify=y_ok)

  neigh = KNeighborsClassifier(n_neighbors=4,weights='distance')
  neigh.fit(x_train_finale, y_train)

  prediction_train = neigh.predict(x_train_finale)
  print(accuracy_score(y_train,prediction_train))

  prediction_test = neigh.predict(x_test)
  print(accuracy_score(y_test,prediction_test))

  for f in fold:
      cross_value = model_selection.cross_val_score(neigh, x_train_flat, y, cv=f)
      print("Accuratezza  con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

def save_model_Naive_Bayes():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    clf = GaussianNB()
    clf.fit(x_train_flat, y_ok)
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)


def save_model_KNN():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    neigh = KNeighborsClassifier(n_neighbors=4,weights='distance')
    neigh.fit(x_train_flat, y_ok)
    with open(KNN_Filename, 'wb') as file:
        pickle.dump(neigh, file)

KNN()