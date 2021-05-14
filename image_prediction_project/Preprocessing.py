import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

DIR = 'immagini-4'
size = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR,name))])


def create_dataset():
    with open('array_image.npy', 'wb') as f:
        for name in os.listdir(DIR):
            if os.path.isfile(os.path.join(DIR, name)):
                im = cv2.imread(os.path.join(DIR, name))
                img = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
                img2 = cv2.resize(img, dsize= (60,80))
                np.save(f, img2)


def read_file():
    x = np.zeros(shape=(size,80,60))
    string = []
    with open('array_image.npy', 'rb') as f:
        i = 0;
        for name in os.listdir(DIR):
            if os.path.isfile(os.path.join(DIR, name)):
                x[i] = np.load(f)
                i = i+1
                string.append(name.split('_')[0])
    return x,string

def save_y():
    y = []
    for name in os.listdir(DIR):
        if os.path.isfile(os.path.join(DIR, name)):
            y.append(name.split('_')[0])
    with open('array_labels.npy', 'wb') as f:
        np.save(f,y)

