#For training audio model
import librosa
import librosa.display
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


files = np.asarray(glob.glob("../data/**/*wav"))  # get all file names
print(len(files))
# create emotion dictionary
emotions = {'01': "neutral", '02': "calm", '03': "happy", '04': "sad",
            '05': "angry", '06': "fearful", '07': "disgust", '08': "surprised"}


if not os.path.exists('output_train'):
    os.mkdir('output_train')

if not os.path.exists('output_test'):
    os.mkdir('output_test')



counts = {}

for audioFile in files:
    print(audioFile)
    emotion = emotions[audioFile[-18:-16]]  # get emotion label

    save_path_train = 'output_train/' + emotion  # Create new file name
    save_path_test = 'output_test/' + emotion  # Create new file name

    if not os.path.exists(save_path_train):
        os.mkdir(save_path_train)

    if not os.path.exists(save_path_test):
        os.mkdir(save_path_test)

    # Convert to melspectrogram
    y, sr = librosa.load(audioFile)  # Load the file with librosa
    yt, _ = librosa.effects.trim(y)  # Trim leading and trailing silence from an audio signal.
    y=yt
    # Make spectrogram from audio file
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=20000, x_axis='time')

    # Logic to split up test and train data
    count = counts.get(emotion, 1)
    if (count % 8 == 0):
        p = save_path_test + "/" + str(count).zfill(2)
    else:
        p = save_path_train + "/" + str(count).zfill(2)
        counts[emotion] = count + 1

    plt.savefig(p)
    plt.close()