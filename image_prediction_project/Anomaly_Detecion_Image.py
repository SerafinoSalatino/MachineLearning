import Preprocessing as pre
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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

d = 4800
size = 5192


def preprocessing(x_f, y_f):
    x = x_f
    y = y_f
    y = np.where(y == 2, 1, y)
    y = np.where(y == 3, 1, y)

    x_train = x[y == 0]
    y_train = np.zeros(len(x_train))

    tmp_x = x[y != 0]
    tmp_y = y[y != 0]

    x_test = np.concatenate((x_train, tmp_x))
    y_test = np.concatenate((y_train, tmp_y))

    x_test = x_test / 255
    x_train = x_train / 255

    x_test_flat = np.array([i.flatten() for i in x_test])
    x_train_flat = np.array([i.flatten() for i in x_train])

    return x_train_flat, x_test_flat, y_train, y_test


def AutoEncoder():
    x_train_flat, x_test_flat, y_train, y_test = preprocessing(x, y_ok)

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

    autoencoder.fit(x_train_flat, x_train_flat, epochs=50, validation_data=(x_test_flat, x_test_flat))

    encoder = tf.keras.Model(inputs=input_img, outputs=encoder4)

    latent_vector = encoder.predict(x_test_flat)

    encoded_input = tf.keras.Input(shape=(64,))
    decoder_layer1 = autoencoder.layers[-4]
    decoder_layer2 = autoencoder.layers[-3]
    decoder_layer3 = autoencoder.layers[-2]
    decoder_layer4 = autoencoder.layers[-1]
    decoder = tf.keras.Model(inputs=encoded_input,
                             outputs=decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))

    reconstructed_imgs = decoder.predict(latent_vector)

    outlierness = np.zeros(5192)

    for i in range(len(x_test_flat)):
        outlierness[i] = autoencoder.evaluate(x_test_flat[i].reshape((1, d)), x_test_flat[i].reshape((1, d)))

    plt.figure(1)
    plt.plot(outlierness, '.')
    plt.xlabel('test id')
    plt.ylabel('outlierness')

    fpr, tpr, threshold = roc_curve(y_test, outlierness)
    auc = roc_auc_score(y_test, outlierness)

    plt.figure(2)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.show()

    print(auc)

    # n = 10
    # plt.figure(figsize=(20,4))
    # for i in range(n):
    # ax = plt.subplot(2,n,i+1)
    # plt.imshow(x_test_flat[i].reshape(80,60))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # ax = plt.subplot(2,n,i+1+n)
    # plt.imshow(reconstructed_imgs[i].reshape(80,60))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

def save_AutoEncoder():
  x_train_flat, x_test_flat, y_train, y_test = preprocessing(x, y_ok)

  input_img = tf.keras.Input(shape=(4800,))
  encoder1 = tf.keras.layers.Dense(units =512, activation='relu',input_dim = d)(input_img)
  encoder2 = tf.keras.layers.Dense(units =256, activation='relu')(encoder1)
  encoder3 = tf.keras.layers.Dense(units = 128, activation='relu')(encoder2)
  encoder4 = tf.keras.layers.Dense(units=64,activation='relu')(encoder3)
  decoder1 = tf.keras.layers.Dense(units = 128, activation='relu')(encoder4)
  decoder2 = tf.keras.layers.Dense(units = 256,activation='relu')(decoder1)
  decoder3 = tf.keras.layers.Dense(units = 512,activation='relu')(decoder2)
  decoder4 = tf.keras.layers.Dense(units = d, activation='sigmoid')(decoder3)

  autoencoder = tf.keras.Model(inputs=input_img,outputs=decoder4)

  autoencoder.compile(optimizer='adam',loss='mse')

  autoencoder.fit(x_train_flat,x_train_flat,epochs=50,validation_data=(x_test_flat,x_test_flat))

  autoencoder.save_weights("ae")

AutoEncoder()