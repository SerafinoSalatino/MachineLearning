import Preprocessing_Audio as pre
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
import pickle

Pkl_Filename = "SVM_MODEL.pkl"

x,y = pre.read_dataset()

regularization = [0.1,0.3, 0.6, 0.8, 1]
kernel = ["linear","rbf","poly"]
fold = [5,10,15,20]

x = minmax_scale(x)
print(x.shape)

def SVM_multiclass_Accuracy():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,stratify=y)
    for c in regularization:
        for k in kernel:
            svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel=k,C=c))
            clf = svm_multiclass.fit(x_train,y_train)
            prediction = clf.predict(x_train)
            print("Accuratezza Training set di SVM con kernel " + k + " con coefficiente "+ str(c) + " : " +str(accuracy_score(y_train,prediction)))
            prediction2 = clf.predict(x_test)
            print("Accuratezza Test-set di SVM con kernel " + k + " con coefficiente "+ str(c) + " : " +str(accuracy_score(y_test,prediction2)))

def SVM_multiclass_Cross_validation():
    for k in kernel:
        svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel=k))
        clf = svm_multiclass.fit(x, y)
        for f in fold:
            cross_value = model_selection.cross_val_score(clf, x, y , cv=f)
            print("Accuratezza con kernel = " + k + " con fold = " + str(f) + "...valore = " + str(np.mean(cross_value)))

def save_model():
    svm_multiclass = OneVsRestClassifier(estimator=SVC(kernel='poly',probability=True))
    clf = svm_multiclass.fit(x, y)
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)

save_model()