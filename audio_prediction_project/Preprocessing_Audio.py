import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
import librosa.feature
import os
from sklearn.preprocessing import minmax_scale

DIR = "audio-1"

dim = 6721

def create_dataset():
    audio_files_happy = glob(DIR+'/Happy/*.wav')
    audio_files_sad = glob(DIR+'/Sad/*.wav')
    audio_files_neutral = glob(DIR+'/Neutral/*.wav')
    audio_files_surprised = glob(DIR+'/Suprised/*.wav')

    y_happy = np.zeros(len(audio_files_happy))

    y_sad = np.ones(len(audio_files_sad))

    y_neutral = np.zeros(len(audio_files_neutral))
    y_neutral = np.where(y_neutral == 0,2,y_neutral)

    y_suprised = np.zeros(len(audio_files_surprised))
    y_suprised = np.where(y_suprised == 0,3,y_suprised)

    args = (y_sad,y_happy,y_suprised,y_neutral)
    y_dataset = np.concatenate(args)
    np.save("array_labels.npy",y_dataset)
    x = []
    for dir in os.listdir('./audio-1'):
        audio_files= glob(DIR + '/' + dir + '/*.wav')
        for file in range(len(audio_files)):
            audio,sr = lr.load(audio_files[file])
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            #feature audio

            rms = np.mean(librosa.feature.rms(audio)) #prima feature
            zc = np.sum(librosa.zero_crossings(audio)) #seconda feature
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)) #terza feature
            sc = np.mean(librosa.feature.spectral_centroid(audio,sr=sr)) #quarta feature
            sro = np.mean(librosa.feature.spectral_rolloff(audio,sr=sr)) #quinta feature
            sbw = np.mean(librosa.feature.spectral_bandwidth(audio,sr=sr)) #sesta feature
            chroma = np.mean(librosa.feature.chroma_stft(audio, sr=sr)) #settima feature
            mfcc = librosa.feature.mfcc(audio,sr=sr,n_mfcc=40)  #40 feature di mfcc

            delta = np.mean(librosa.feature.delta(mfcc)) #ottava feature
            pitch = np.mean(pitches) #nona feature
            max_amplitude = np.max(audio)


            a = [rms,zc,zcr,sc,sro,sbw,chroma,delta,pitch,max_amplitude]
            for s in mfcc:
                a.append(np.mean(s))

            x.append(a)
    np.save("array_audio.npy",x)

def read_dataset():
    y = np.load("array_labels.npy")

    x = np.load("array_audio.npy")

    return x,y

def create_test_set(dir):
    y = []
    x = []
    for dir1 in os.listdir('./'+dir):
        audio_files= glob(dir + '/' + dir1 + '/*.wav')
        label = 0
        if dir1 == 'Sad':
            label = 1
        if dir1 == 'Neutral':
            label = 2
        if dir1 == 'Suprised':
            label = 3
        for file in range(len(audio_files)):
            audio,sr = lr.load(audio_files[file])
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            #feature audio

            rms = np.mean(librosa.feature.rms(audio)) #prima feature
            zc = np.sum(librosa.zero_crossings(audio)) #seconda feature
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)) #terza feature
            sc = np.mean(librosa.feature.spectral_centroid(audio,sr=sr)) #quarta feature
            sro = np.mean(librosa.feature.spectral_rolloff(audio,sr=sr)) #quinta feature
            sbw = np.mean(librosa.feature.spectral_bandwidth(audio,sr=sr)) #sesta feature
            chroma = np.mean(librosa.feature.chroma_stft(audio, sr=sr)) #settima feature
            mfcc = librosa.feature.mfcc(audio,sr=sr,n_mfcc=40)  #40 feature di mfcc

            delta = np.mean(librosa.feature.delta(mfcc)) #ottava feature
            pitch = np.mean(pitches) #nona feature
            max_amplitude = np.max(audio)


            a = [rms,zc,zcr,sc,sro,sbw,chroma,delta,pitch,max_amplitude]
            for s in mfcc:
                a.append(np.mean(s))
            y.append(label)
            x.append(a)
    x = minmax_scale(x)
    return np.array(x),np.array(y)
