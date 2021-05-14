import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import Preprocessing_Audio as pre
from sklearn.preprocessing import minmax_scale

x,y = pre.read_dataset()

x =minmax_scale(x)



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

    return x_train, x_test, y_train, y_test


def AutoEncoder():
    d = 50
    x_train, x_test, y_train, y_test = preprocessing(x, y)

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

    autoencoder.fit(x_train, x_train, epochs=100, validation_data=(x_test, x_test))

    autoencoder.save_weights("ae")

    encoder = tf.keras.Model(inputs=input_img, outputs=encoder5)

    latent_vector = encoder.predict(x_test)

    encoded_input = tf.keras.Input(shape=(32,))
    decoder_layer1 = autoencoder.layers[-5]
    decoder_layer2 = autoencoder.layers[-4]
    decoder_layer3 = autoencoder.layers[-3]
    decoder_layer4 = autoencoder.layers[-2]
    decoder_layer5 = autoencoder.layers[-1]
    decoder = tf.keras.Model(inputs=encoded_input,
                             outputs=decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))))

    reconstructed_imgs = decoder.predict(latent_vector)

    outlierness = np.zeros(len(x_test))

    for i in range(len(x_test)):
        outlierness[i] = autoencoder.evaluate(x_test[i].reshape((1, d)), x_test[i].reshape((1, d)))

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