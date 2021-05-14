import tensorflow as tf
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold
import Preprocessing_Audio as pre

x,y = pre.read_dataset()

x =minmax_scale(x)
x = x.reshape((x.shape[0],50,1))

def Convolutional_NN():

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,
                                                                               stratify=y)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(16, 2, padding='valid', activation='relu',input_shape=(50,1)))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.Conv1D(32, 2, activation='relu',padding='valid'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32,activation='relu',activity_regularizer=regularizers.l2(1e-5)))
    model.add(tf.keras.layers.Dense(4,activity_regularizer=regularizers.l2(1e-5)))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print(x_train.shape)
    print(x_test.shape)
    history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Conv1D(16, 2, padding='valid', activation='relu', input_shape=(50, 1)))
    model2.add(tf.keras.layers.MaxPool1D(2))
    model2.add(tf.keras.layers.Conv1D(32 ,2, activation='relu',padding='valid'))
    model2.add(tf.keras.layers.MaxPool1D(2))
    model2.add(tf.keras.layers.Conv1D(64 ,2, activation='relu',padding='valid'))
    model2.add(tf.keras.layers.Dropout(0.5))
    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(32, activation='relu',activity_regularizer=regularizers.l2(1e-5)))
    model2.add(tf.keras.layers.Dense(16, activation='relu',activity_regularizer=regularizers.l2(1e-5)))
    model2.add(tf.keras.layers.Dense(4, activation='softmax',activity_regularizer=regularizers.l2(1e-5)))


    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history2 = model2.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

    model3 = tf.keras.Sequential()
    model3.add(tf.keras.layers.Conv1D(16, 2, padding='valid', activation='relu', input_shape=(50, 1)))
    model3.add(tf.keras.layers.MaxPool1D(2))
    model3.add(tf.keras.layers.Conv1D(32, 2, activation='relu',padding='valid'))
    model3.add(tf.keras.layers.MaxPool1D(2))
    model3.add(tf.keras.layers.Conv1D(64, 2, activation='relu',padding='valid'))
    model3.add(tf.keras.layers.MaxPool1D(2))
    model3.add(tf.keras.layers.Conv1D(128, 2, activation='relu',padding='valid'))
    model3.add(tf.keras.layers.Dropout(0.5))
    model3.add(tf.keras.layers.Flatten())
    model3.add(tf.keras.layers.Dense(32, activation='relu',activity_regularizer=regularizers.l2(1e-5)))
    model3.add(tf.keras.layers.Dense(16, activation='relu',activity_regularizer=regularizers.l2(1e-5)))
    model3.add(tf.keras.layers.Dense(8, activation='relu',activity_regularizer=regularizers.l2(1e-5)))
    model3.add(tf.keras.layers.Dense(4, activation='softmax',activity_regularizer=regularizers.l2(1e-5)))

    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history3 = model3.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

    plt.plot(history.history["val_accuracy"], label="Rete con 2 livelli convoluzionali")
    plt.plot(history2.history["val_accuracy"], label="Rete con 3 livelli convoluzionali")
    plt.plot(history3.history["val_accuracy"], label="Rete con 4 livelli convoluzionali")
    plt.title("Reti Neurali Convoluzionali")
    plt.legend()
    plt.show()

def Convolutional_NN_CV():
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)
    f = 1
    for train, test in kfold.split(x, y):
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
        history2 = model.fit(x[train], y[train], epochs=100, verbose=0)

        scores = model.evaluate(x[test], y[test], verbose=0)
        print(f'Accuratezza per fold ' + str(f) + ':' + str(scores[1]))

        f = f + 1

def save_Convolutional_NN():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,
                                                                               stratify=y)
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
    model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))
    model.save_weights("w32f")
