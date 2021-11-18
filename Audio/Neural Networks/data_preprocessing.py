# For training audio model
import librosa
import librosa.display
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numpy import savetxt

def generate_spectrograms():
    files = np.asarray(glob.glob("../data/**/*wav"))  # get all file names

    print(len(files))
    # create emotion dictionary
    emotions = {'01': "neutral", '02': "calm", '03': "happy", '04': "sad",
                '05': "angry", '06': "fearful", '07': "disgust", '08': "surprised"}

    # create emotion dictionary
    emotions = {'01': "neutral", '03': "happy", '04': "sad"}

    # if not os.path.exists('output_train'):
    #     os.mkdir('output_train')
    #
    # if not os.path.exists('output_test'):
    #     os.mkdir('output_test')
    counts = {}

    for audioFile in files:
        print(audioFile)
        emotion = emotions[audioFile[-18:-16]]  # get emotion label

        save_path_image = 'output_spectrogram/'

        # if not os.path.exists(save_path_train):
        #     os.mkdir(save_path_train)
        #
        # if not os.path.exists(save_path_test):
        #     os.mkdir(save_path_test)

        if not os.path.exists(save_path_image):
            os.mkdir(save_path_image)

        # Convert to melspectrogram
        y, sr = librosa.load(audioFile)  # Load the file with librosa
        yt, _ = librosa.effects.trim(y)  # Trim leading and trailing silence from an audio signal.
        y=yt
        # Make spectrogram from audio file
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect, fmax=20000)

        p = save_path_image + "/" + audioFile[-24:-4]

        plt.axis('off')
        plt.savefig(p, bbox_inches='tight', pad_inches=0)
        plt.close('all')


def generate_spectrogram_info():
    files = np.asarray(glob.glob("../data/**/*wav"))  # get all file names

    # store emotion and log-mel spectrogram info in dataframe
    emotion_labels = []

    # Not using neutral
    emotions = {'02': "calm", '03': "happy", '04': "sad",
                '05': "angry", '06': "fearful", '07': "disgust", '08': "surprised"}

    # create emotion dictionary
    emotions = {'01': "neutral", '03': "happy", '04': "sad"}

    df = pd.DataFrame(columns=['feature'])
    counter = 0
    for audioFile in files:
        print(audioFile)
        emotion = ""
        try:
            # try catch to skip neutral emotions
            emotion = emotions[audioFile[-18:-16]]  # get emotion label
            emotion_labels.append(emotion)
        except:
            continue
        try:
            # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
            X, sample_rate = librosa.load(audioFile,res_type='kaiser_fast')  # Load the file with librosa
            # X, sample_rate = librosa.load(audioFile, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)

            y, _ = librosa.effects.trim(X)  # Trim leading and trailing silence from an audio signal.

            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)
            # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            df.loc[counter] = [feature]
            counter = counter + 1
        except ValueError:
            continue

        # y, sr = librosa.load(audioFile,res_type='kaiser_fast')  # Load the file with librosa
        # # yt, _ = librosa.effects.trim(y)  # Trim leading and trailing silence from an audio signal.
        # # y=yt
        # # get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
        # mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    labels = pd.DataFrame(emotion_labels)
    df2 = pd.DataFrame(df['feature'].values.tolist())
    newdf = pd.concat([df2, labels], axis=1)  # concat labels to features
    newdf = newdf.fillna(0)
    print(newdf[:10])
    print(newdf.shape)
    newdf.to_csv("data.csv", index=False)




generate_spectrogram_info()