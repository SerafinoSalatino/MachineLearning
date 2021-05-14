import tensorflow as tf
from sklearn.metrics import accuracy_score
import Preprocessing as pre
import numpy as np
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt

fold = [5,10,15,20]
x,y = pre.read_file()

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

def Sequential_NN():
    x_train = x / 255
    x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_ok, train_size=0.8,
                                                                               stratify=y_ok)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(80, 60)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4)
    ])
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    model2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(80, 60)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history2 = model2.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    model3 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(80, 60)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history3 = model3.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    plt.plot(history.history["val_accuracy"], label="Rete a 4 livelli")
    plt.plot(history2.history["val_accuracy"], label="Rete a 5 livelli")
    plt.plot(history3.history["val_accuracy"], label="Rete a 6 livelli")
    plt.title("Reti Neurali Dense")
    plt.legend()
    plt.show()


def Convolutional_NN():
    x_train = x / 255
    x_train = x_train.reshape((5192, 80, 60, 1))
    x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_ok, train_size=0.8,
                                                                               stratify=y_ok)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(40, (3, 3), padding='valid', activation='relu', input_shape=(80, 60, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(80, 60, 1)))
    model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(32, activation='relu'))
    model2.add(tf.keras.layers.Dense(4))

    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history2 = model2.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    model3 = tf.keras.Sequential()
    model3.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(80, 60, 1)))
    model3.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model3.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model3.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model3.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model3.add(tf.keras.layers.Flatten())
    model3.add(tf.keras.layers.Dense(64, activation='relu'))
    model3.add(tf.keras.layers.Dense(32, activation='relu'))
    model3.add(tf.keras.layers.Dense(4))

    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history3 = model3.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    plt.plot(history.history["val_accuracy"], label="Rete con 1 livello denso")
    plt.plot(history2.history["val_accuracy"], label="Rete con 2 livelli densi")
    plt.plot(history3.history["val_accuracy"], label="Rete con 3 livelli densi")
    plt.title("Reti Neurali Convoluzionali")
    plt.legend()
    plt.show()


def save_Sequential_NN():
    x_train = x / 255
    x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_ok, train_size=0.8,
                                                                               stratify=y_ok)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(80, 60)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))
    model.save_weights("w16f")

def save_Convolutional_NN():
    x_train = x / 255
    x_train = x_train.reshape((5192, 80, 60, 1))
    x_train_finale, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_ok, train_size=0.8,
                                                                               stratify=y_ok)
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
    model.fit(x_train_finale, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))
    model.save_weights("w32f")



