import tensorflow as tf
from sklearn.model_selection import KFold

import Preprocessing_Audio as pre
import sklearn.model_selection as model_selection
from sklearn.preprocessing import minmax_scale
import pickle
import matplotlib.pyplot as plt



x,y = pre.read_dataset()
print(x.shape)
print(y.shape)

x =minmax_scale(x)

def Sequential_NN():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,
                                                                               stratify=y)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4)
    ])
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history2 = model2.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history3 = model3.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

    plt.plot(history.history["val_accuracy"], label="Rete a 3 livelli")
    plt.plot(history2.history["val_accuracy"], label="Rete a 4 livelli")
    plt.plot(history3.history["val_accuracy"], label="Rete a 5 livelli")
    plt.title("Reti Neurali Dense")
    plt.legend()
    plt.show()

def Sequential_NN_CV():
  num_folds = 10
  kfold = KFold(n_splits=num_folds, shuffle=True)
  f = 1
  for train, test in kfold.split(x, y):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history = model.fit(x[train], y[train],epochs=100,verbose=0)

    scores = model.evaluate(x[test], y[test], verbose=0)
    print(f'Accuratezza per fold ' + str(f) + ':' + str(scores[1]))

    f = f + 1

def save_Sequential_NN():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8,
                                                                               stratify=y)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))
    model.save_weights("w16f")
