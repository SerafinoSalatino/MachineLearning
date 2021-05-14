import pickle
import cv2
import numpy as np
import os,sys
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

SVM_Filename = "SVM_MODEL.pkl"
ADA_Filename = "ADA_MODEL.pkl"
Naive_Bayes_Filename = "Naive_Bayes.pkl"
KNN_Filename = "KNN.pkl"
DIR = "Test set" #DIRECTORY DI TEST

def load_model_SVM(DIR):
    with open(SVM_Filename, 'rb') as file:
        model = pickle.load(file)
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            y.append(name.split('_')[0])
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    x_test = np.array(x)
    x_test = x_test/255
    x_test_flat = np.array([i.flatten() for i in x_test])
    prediction = model.predict(x_test_flat)
    print("Accuratezza : " + str(accuracy_score(y,prediction)))

def load_model_ADA(DIR):
    with open(ADA_Filename, 'rb') as file:
        model = pickle.load(file)
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            y.append(name.split('_')[0])
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    x_test = np.array(x)
    x_test = x_test/255
    x_test_flat = np.array([i.flatten() for i in x_test])
    prediction = model.predict(x_test_flat)
    print("Accuratezza : " + str(accuracy_score(y,prediction)))

def load_model_Sequential_NN(DIR):
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            namet = name.split('_')[0]
            if namet == "shirts":
                 y.append(0)
            if namet == "shorts":
                y.append(1)
            if namet == "sunglasses":
                y.append(2)
            if namet == "wallets":
                y.append(3)
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    y = np.array(y)
    x_test = np.array(x)
    x_test = x_test/255

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(80, 60)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.load_weights('Sequential_NN/w16f')
    results = model.evaluate(x_test, y)
    print("test loss, test acc:", results)

def load_model_Convolutional_NN(DIR):
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            namet = name.split('_')[0]
            if namet == "shirts":
                 y.append(0)
            if namet == "shorts":
                y.append(1)
            if namet == "sunglasses":
                y.append(2)
            if namet == "wallets":
                y.append(3)
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    y = np.array(y)
    x_test = np.array(x)
    x_test = x_test/255

    shape = x_test.shape
    new_shape = (shape[0],shape[1],shape[2],1)

    x_test = x_test.reshape(new_shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(80, 60, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(4))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.load_weights('Convolutional_NN/w32f')
    results = model.evaluate(x_test, y)
    print("test loss, test acc:", results)

def load_Naive_Bayes(DIR):
    with open(Naive_Bayes_Filename, 'rb') as file:
        model = pickle.load(file)
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            namet = name.split('_')[0]
            if namet == "shirts":
                y.append(0)
            if namet == "shorts":
                y.append(1)
            if namet == "sunglasses":
                y.append(2)
            if namet == "wallets":
                y.append(3)
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    x_test = np.array(x)
    x_test = x_test / 255
    x_test_flat = np.array([i.flatten() for i in x_test])
    prediction = model.predict(x_test_flat)
    print("Accuratezza : " + str(accuracy_score(y, prediction)))

def load_KNN(DIR):
    with open(KNN_Filename, 'rb') as file:
        model = pickle.load(file)
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            namet = name.split('_')[0]
            if namet == "shirts":
                y.append(0)
            if namet == "shorts":
                y.append(1)
            if namet == "sunglasses":
                y.append(2)
            if namet == "wallets":
                y.append(3)
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    x_test = np.array(x)
    x_test = x_test / 255
    x_test_flat = np.array([i.flatten() for i in x_test])
    prediction = model.predict(x_test_flat)
    print("Accuratezza : " + str(accuracy_score(y, prediction)))


def load_AutoEncoder(DIR):
    x = []
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            namet = name.split('_')[0]
            if namet == "shirts":
                y.append(0)
            if namet == "shorts":
                y.append(1)
            if namet == "sunglasses":
                y.append(2)
            if namet == "wallets":
                y.append(3)
            im = cv2.imread(os.path.join(DIR, name))
            img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            img2 = cv2.resize(img, dsize=(60, 80))
            x.append(img2)
    y = np.array(y)
    x_test = np.array(x)
    x_test = x_test / 255
    x_test_flat = np.array([i.flatten() for i in x_test])

    d = 4800
    size = len(x_test_flat)

    input_img = tf.keras.Input(shape=(4800,))
    encoder1 = tf.keras.layers.Dense(units=512, activation='relu', input_dim=d)(input_img)
    encoder2 = tf.keras.layers.Dense(units=256, activation='relu')(encoder1)
    encoder3 = tf.keras.layers.Dense(units=128, activation='relu')(encoder2)
    encoder4 = tf.keras.layers.Dense(units=64, activation='relu')(encoder3)
    decoder1 = tf.keras.layers.Dense(units=128, activation='relu')(encoder4)
    decoder2 = tf.keras.layers.Dense(units=256, activation='relu')(decoder1)
    decoder3 = tf.keras.layers.Dense(units=512, activation='relu')(decoder2)
    decoder4 = tf.keras.layers.Dense(units=d, activation='sigmoid')(decoder3)

    autoencoder = tf.keras.Model(inputs=input_img, outputs=decoder4)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.load_weights('AutoEncoder/ae')

    outlierness = np.zeros(size)

    for i in range(size):
        outlierness[i] = autoencoder.evaluate(x_test_flat[i].reshape((1,d)),x_test_flat[i].reshape((1,d)))

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