# For training audio model
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

counts = {}

for audioFile in files:
    print(audioFile)
    emotion = emotions[audioFile[-18:-16]]  # get emotion label

    save_path_image = 'unlabeled_spectrograms/'

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
