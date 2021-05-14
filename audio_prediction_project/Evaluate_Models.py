import getopt
import pickle
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import Preprocessing_Audio as pre

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


SVM_Filename = "SVM_MODEL.pkl"
ADA_Filename = "ADA_MODEL.pkl"
Naive_Bayes_Filename = "Naive_Bayes.pkl"
KNN_Filename = "KNN.pkl"
DIR = ""  #DIRECTORY DI TEST

def load_model_SVM(DIR):
    with open(SVM_Filename, 'rb') as file:
        model = pickle.load(file)
    x,y = pre.create_test_set(DIR)
    prediction = model.predict(x)
    print("Accuratezza : " + str(accuracy_score(y,prediction)))

def load_model_ADA(DIR):
    with open(ADA_Filename, 'rb') as file:
        model = pickle.load(file)
    x,y = pre.create_test_set(DIR)
    prediction = model.predict(x)
    print("Accuratezza : " + str(accuracy_score(y,prediction)))

def load_model_Sequential_NN(DIR):
    x, y = pre.create_test_set(DIR)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print("ciao")
    model.load_weights('Dense_NN_Model/w16f')
    print("ciao")
    results = model.evaluate(x, y)
    print("test loss, test acc:", results)

def load_model_Convolutional_NN(DIR):
    x, y = pre.create_test_set(DIR)

    x = x.reshape((x.shape[0], 50, 1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(16, 2, padding='valid', activation='relu', input_shape=(50, 1)))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.Conv1D(32, 2, activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.Conv1D(64, 2, activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.Conv1D(128, 2, activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.load_weights('Convolutional_NN_Model/w32f')
    results = model.evaluate(x, y)
    print("test loss, test acc:", results)

def load_Naive_Bayes(DIR):
    with open(Naive_Bayes_Filename, 'rb') as file:
        model = pickle.load(file)
    x,y = pre.create_test_set(DIR)
    prediction = model.predict(x)
    print("Accuratezza : " + str(accuracy_score(y, prediction)))

def load_KNN(DIR):
    with open(KNN_Filename, 'rb') as file:
        model = pickle.load(file)
    x,y = pre.create_test_set(DIR)
    prediction = model.predict(x)
    print("Accuratezza : " + str(accuracy_score(y, prediction)))

def load_AutoEncoder(DIR):

    x,y = pre.create_test_set(DIR)
    d = 50
    size = len(x)

    input_img = tf.keras.Input(shape=(d,))
    encoder1 = tf.keras.layers.Dense(units=512, activation='relu', input_dim=d)(input_img)
    encoder2 = tf.keras.layers.Dense(units=256, activation='relu')(encoder1)
    encoder3 = tf.keras.layers.Dense(units=128, activation='relu')(encoder2)
    encoder4 = tf.keras.layers.Dense(units=64, activation='relu')(encoder3)
    encoder5 = tf.keras.layers.Dense(units=32, activation='relu')(encoder4)
    decoder1 = tf.keras.layers.Dense(units=64, activation='relu')(encoder5)
    decoder2 = tf.keras.layers.Dense(units=128, activation='relu')(decoder1)
    decoder3 = tf.keras.layers.Dense(units=256, activation='relu')(decoder2)
    decoder4 = tf.keras.layers.Dense(units=512, activation='relu')(decoder3)
    decoder5 = tf.keras.layers.Dense(units=d, activation='sigmoid')(decoder4)

    autoencoder = tf.keras.Model(inputs=input_img, outputs=decoder5)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.load_weights('AutoEncoder/ae')

    outlierness = np.zeros(size)

    for i in range(size):
        outlierness[i] = autoencoder.evaluate(x[i].reshape((1,d)),x[i].reshape((1,d)))

    y = np.where(y == 2, 1, y)
    y = np.where(y == 3, 1, y)
    y_test = y

    fpr,tpr,threshold = roc_curve(y_test,outlierness)
    auc = roc_auc_score(y_test,outlierness)
    print("Valore AUC =" + str(auc))

    plt.figure(2)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],linestyle='--')
    plt.show()


def main(argv):
   model = argv[1]
   if model == 'Adaboost':
       load_model_ADA(argv[0])
   if model == 'SVM':
       load_model_SVM(argv[0])
   if model == 'Dense_NN':
       load_model_Sequential_NN(argv[0])
   if model == 'Convolutional_NN':
       load_model_Convolutional_NN(argv[0])
   if model == 'Naive_Bayes':
       load_Naive_Bayes(argv[0])
   if model == 'KNN':
       load_KNN(argv[0])
   if model == 'Anomaly_Detection':
       load_AutoEncoder(argv[0])

main(argv=sys.argv[1:])
