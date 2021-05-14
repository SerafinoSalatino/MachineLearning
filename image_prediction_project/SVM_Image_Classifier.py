import Preprocessing as pre
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pickle
import cv2
import matplotlib.pyplot as plt

Pkl_Filename = "SVM_MODEL.pkl"


x,y = pre.read_file()

regularization = [0.1,0.3, 0.6, 0.8, 1]
kernel = ["linear","rbf","poly"]
array_pca = [10, 30, 80, 200, 400]
fold = [5,10,15,20]

def SVM_multiclass_No_PCA():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    x_train_flat_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train_flat, y, train_size=0.8,
                                                                                    stratify=y)
    for c in regularization:
        for k in kernel:
            svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel=k,C=c))
            clf = svm_multiclass.fit(x_train_flat_finale,y_train)
            prediction = clf.predict(x_train_flat_finale)
            print("Accuratezza Training set di SVM con kernel " + k + " con coefficiente "+ str(c) + " : " +str(accuracy_score(y_train,prediction)))
            prediction2 = clf.predict(x_test)
            print("Accuratezza Test set di SVM con kernel " + k + " con coefficiente "+ str(c) + " : " +str(accuracy_score(y_test,prediction2)))


def SVM_multiclass_PCA():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    x_train_flat_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train_flat, y, train_size=0.8,
                                                                              stratify=y)
    for val in array_pca:
        print("Avvio pca con valore= " + str(val))
        pca = PCA(n_components=val)
        pca.fit(x_train_flat_finale)
        x_train_c = pca.transform(x_train_flat_finale)
        x_test_c = pca.transform(x_test)
        print("Conclusa pca con valore= " + str(val))
        for k in kernel:
            svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel=k))
            clf = svm_multiclass.fit(x_train_c, y_train)
            prediction = clf.predict(x_train_c)
            print("Accuratezza Training set di SVM con kernel " + k + " con valore PCA = " + str(val) + " : " + str(
                accuracy_score(y_train,prediction)))
            print()
            prediction2 = clf.predict(x_test_c)
            print("Accuratezza Test set di SVM con kernel " + k + " con coefficiente " + str(val) + " : " + str(
                accuracy_score(y_test,prediction2)))

def SVM_multiclass_Cross_validation():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    for k in kernel:
        svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel=k))
        clf = svm_multiclass.fit(x_train_flat, y)
        for f in fold:
            cross_value = model_selection.cross_val_score(clf, x_train_flat, y , cv=f)
            print("Accuratezza con kernel = " + k + " con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

def save_model():
    x_train = x / 255
    x_train_flat = np.array([i.flatten() for i in x_train])
    svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel='poly'))
    clf = svm_multiclass.fit(x_train_flat, y)
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)

def load_model():
    with open(Pkl_Filename, 'rb') as file:
        model = pickle.load(file)
    im = cv2.imread("/home/serafino/Scaricati/maglietta-da-uomo-fronte.jpg")
    img = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,dsize=(80,60))
    x_train = img / 255
    x_train = x_train.flatten()
    x_ok = np.reshape(x_train,(-1,1))
    x_ok = x_ok.transpose()
    print(model.predict(x_ok))
