import glob
import re

import librosa as lb
import numpy as np
import pandas as pd
from librosa import feature as ft


def extract_audio_features(file):
    y, sr = lb.load(file)
    return {'chroma_stft': np.mean(ft.chroma_stft(y=y, sr=sr)),
            'chroma_cqt': np.mean(ft.chroma_cqt(y=y, sr=sr)),
            'chroma_cens': np.mean(ft.chroma_cens(y=y, sr=sr)),
            'melspectrogram': np.mean(ft.melspectrogram(y=y, sr=sr)),
            'mfcc': np.mean(ft.mfcc(y=y, sr=sr)),
            'rms': np.mean(ft.rms(y=y)),
            'spectral_centroid': np.mean(ft.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(ft.spectral_bandwidth(y=y, sr=sr)),
            'spectral_contrast': np.mean(ft.spectral_contrast(y=y, sr=sr)),
            'spectral_flatness': np.mean(ft.spectral_flatness(y=y)),
            'spectral_rolloff': np.mean(ft.spectral_rolloff(y=y, sr=sr)),
            'poly_features': np.mean(ft.poly_features(y=y, sr=sr)),
            'tonnetz': np.mean(ft.tonnetz(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(ft.zero_crossing_rate(y=y))}


def load_audio_features():
    try:
        df = pd.read_csv("features.csv")
    except FileNotFoundError:
        files = np.asarray(glob.glob("../data/**/*.wav", recursive=True))
        emotions = {'01': "neutral", '02': "calm", '03': "happy", '04': "sad",
                    '05': "angry", '06': "fearful", '07': "disgust", '08': "surprised"}

        df = pd.DataFrame(extract_audio_features(file) for file in files)
        df["labels"] = [emotions[re.search(r"(?<=\d\d-\d\d-)\d\d", file).group(0)]
                        for file in files]
        df.to_csv("features.csv", index=False)
    return df


if __name__ == "__main__":
    df = load_audio_features()
    print(df.labels.value_counts())
