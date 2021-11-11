import glob
import numpy as np
import librosa as lb
from librosa import feature as ft

files = np.asarray(glob.glob("Traditional ML/data/audio_data/**/*wav"))

spectral_features = {'chroma_stft': [], 'chroma_cqt': [], 'chroma_cens': [],
                     'melspectrogram': [], 'mfcc': [], 'rms': [], 'spectral_centroid': [],
                     'spectral_bandwidth': [], 'spectral_contrast': [],
                     'spectral_flatness': [], 'spectral_rolloff': [], 'poly_features': [],
                     'tonnetz': [], 'zero_crossing_rate': []}

y, sr = lb.load(files[0])

spectral_features['chroma_stft'] = ft.chroma_stft(y, sr)
spectral_features['chroma_cqt'] = ft.chroma_cqt(y, sr)
spectral_features['chroma_cens'] = ft.chroma_cens(y, sr)
spectral_features['melspectrogram'] = ft.melspectrogram(y, sr)
spectral_features['mfcc'] = ft.mfcc(y, sr)
spectral_features['rms'] = ft.rms(y, sr)
spectral_features['spectral_centroid'] = ft.spectral_centroid(y, sr)
spectral_features['spectral_bandwidth'] = ft.spectral_bandwidth(y, sr)
spectral_features['spectral_contrast'] = ft.spectral_contrast(y, sr)

for item in spectral_features:
    print(item.title(), ': ', spectral_features[item])

# commented code for looping through all data
# count = 0
# for elem in spectral_features['chroma_stft']:
#     for elem2 in elem:
#         for elem3 in elem2:
#             print('elem: ', elem)
#             count += 1
#
# print(count)

# # extract features!
# for filename in files:
#     y, sr = lb.load(filename)
#
#     # each feature
#     spectral_features['chroma_stft'].append(ft.chroma_stft(y, sr))
#     print(spectral_features['chroma_stft'])
#     spectral_features['chroma_cqt'].append(ft.chroma_cqt(y, sr))
#     spectral_features['chroma_cens'].append(ft.chroma_cens(y, sr))
#     spectral_features['melspectrogram'].append(ft.melspectrogram(y, sr))
#     spectral_features['mfcc'].append(ft.mfcc(y, sr))
#     spectral_features['rms'].append(ft.rms(y, sr))
#     spectral_features['spectral_centroid'].append(ft.spectral_centroid(y, sr))
#     spectral_features['spectral_bandwidth'].append(ft.spectral_bandwidth(y, sr))
#     spectral_features['spectral_contrast'].append(ft.spectral_contrast(y, sr))

# # loop through to extract actual files
# count = 0
# for filename in files:
#     with open(filename) as file:
#         files[count] = file
#         print(files[count])
#     count += 1
#
# # files[i] = open(files[i]) for i in range(len(files))
#
# for file in files:
#     print(file)
