import glob
import re

import librosa as lb
import numpy as np
import pandas as pd
from librosa import feature as ft
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt


# def extract_audio_features(file):
#     y, sr = lb.load(file)
#     return {'chroma_stft': np.mean(ft.chroma_stft(y=y, sr=sr)),
#             'chroma_cqt': np.mean(ft.chroma_cqt(y=y, sr=sr)),
#             'chroma_cens': np.mean(ft.chroma_cens(y=y, sr=sr)),
#             'melspectrogram': np.mean(ft.melspectrogram(y=y, sr=sr)),
#             'mfcc': np.mean(ft.mfcc(y=y, sr=sr)),
#             'rms': np.mean(ft.rms(y=y)),
#             'spectral_centroid': np.mean(ft.spectral_centroid(y=y, sr=sr)),
#             'spectral_bandwidth': np.mean(ft.spectral_bandwidth(y=y, sr=sr)),
#             'spectral_contrast': np.mean(ft.spectral_contrast(y=y, sr=sr)),
#             'spectral_flatness': np.mean(ft.spectral_flatness(y=y)),
#             'spectral_rolloff': np.mean(ft.spectral_rolloff(y=y, sr=sr)),
#             'poly_features': np.mean(ft.poly_features(y=y, sr=sr)),
#             'tonnetz': np.mean(ft.tonnetz(y=y, sr=sr)),
#             'zero_crossing_rate': np.mean(ft.zero_crossing_rate(y=y))}


# def extract_audio_features(file):
#     y, sr = lb.load(file)
#
#     return {'mean': np.mean(ft.mfcc(y=y, sr=sr, n_mfcc=40)),
#             'median': np.median(ft.mfcc(y=y, sr=sr, n_mfcc=40)),
#             'max': np.nanmax(ft.mfcc(y=y, sr=sr, n_mfcc=40)),
#             'min': np.nanmin(ft.mfcc(y=y, sr=sr, n_mfcc=40)),
#             'std': np.std(ft.mfcc(y=y, sr=sr, n_mfcc=40)),
#             'mean2': np.mean(ft.zero_crossing_rate(y=y)),
#             'median2': np.median(ft.zero_crossing_rate(y=y)),
#             'max2': np.nanmax(ft.zero_crossing_rate(y=y)),
#             'min2': np.nanmin(ft.zero_crossing_rate(y=y)),
#             'std2': np.std(ft.zero_crossing_rate(y=y))}
count = 0
def extract_audio_features(file):
    y, sr = lb.load(file)
    global count
    dict = {}
    for i in range(0, 40):
        dict[i] = np.mean(ft.mfcc(y=y, sr=sr, n_mfcc=40)[i])
    for i in range(0, 40):
        dict[i + 40] = np.median(ft.mfcc(y=y, sr=sr, n_mfcc=40)[i])
    for i in range(0, 40):
        dict[i + 80] = np.nanmax(ft.mfcc(y=y, sr=sr, n_mfcc=40)[i])
    for i in range(0, 40):
        dict[i + 120] = np.nanmin(ft.mfcc(y=y, sr=sr, n_mfcc=40)[i])
    for i in range(0, 40):
        dict[i + 160] = np.std(ft.mfcc(y=y, sr=sr, n_mfcc=40)[i])
    print(count)
    count += 1
    return dict


# emotions = {'01': "neutral", '02': "calm", '03': "happy", '04': "sad",
#                     '05': "angry", '06': "fearful", '07': "disgust", '08': "surprised"}

neutral = {'01': "neutral", '02': "neutral", '03': "not neutral", '04': "not neutral",
           '05': "not neutral", '06': "not neutral", '07': "not neutral", '08': "not neutral"}

happy = {'01': "not happy", '02': "not happy", '03': "happy", '04': "not happy",
         '05': "not happy", '06': "not happy", '07': "not happy", '08': "happy"}

sad = {'01': "not sad", '02': "not sad", '03': "not sad", '04': "sad",
         '05': "not sad", '06': "sad", '07': "not sad", '08': "not sad"}

angry = {'01': "not angry", '02': "not angry", '03': "not angry", '04': "not angry",
       '05': "angry", '06': "not angry", '07': "angry", '08': "not angry"}


def load_audio_features():
    try:
        df = pd.read_csv("features5.csv")
    except FileNotFoundError:
        files = np.asarray(glob.glob("../data/**/*.wav"))

        df = pd.DataFrame(extract_audio_features(file) for file in files)

        df["labels_N"] = [neutral[re.search(r"(?<=\d\d-\d\d-)\d\d", file).group(0)]
                          for file in files]

        df["labels_H"] = [happy[re.search(r"(?<=\d\d-\d\d-)\d\d", file).group(0)]
                          for file in files]

        df["labels_A"] = [angry[re.search(r"(?<=\d\d-\d\d-)\d\d", file).group(0)]
                          for file in files]

        df["labels_S"] = [sad[re.search(r"(?<=\d\d-\d\d-)\d\d", file).group(0)]
                          for file in files]

        df.to_csv("features5.csv", index=False)

    return df


if __name__ == "__main__":
    df = load_audio_features()
    data = df.to_numpy()
    X = data[:,0:200]
    y = data[:,200:204]
    # y_neutral = data[:,200]
    # y_happy = data[:,201]
    # y_sad = data[:,202]
    # y_angry = data[:,203]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    n_train = y_train[:,0]
    h_train = y_train[:,1]
    s_train = y_train[:,2]
    a_train = y_train[:,3]

    n_test = y_test[:, 0]
    h_test = y_test[:, 1]
    s_test = y_test[:, 2]
    a_test = y_test[:, 3]

    train_list = [n_train, h_train, s_train, a_train]
    test_list = [n_test, h_test, s_test, a_test]

    # index = 0
    # for item in train_list:
    #     knn = KNeighborsClassifier(n_neighbors=7)
    #     knn.fit(X_train, item)
    #     prediction = knn.predict(X_test)
    #     accuracy = accuracy_score(test_list[index], prediction)
    #     print(accuracy)
    #     index +=1

    # knn = KNeighborsClassifier(n_neighbors=7)
    # knn.fit(X_train, s_train)
    # ConfusionMatrixDisplay.from_estimator(knn, X_test, s_test)
    # plt.show()
    #
    # tree_clf = DecisionTreeClassifier()
    # tree_clf.fit(X_train, y_train)
    # prediction = tree_clf.predict(X_test)
    # accuracy = accuracy_score(y_test, prediction)
    # print(accuracy)
    # ConfusionMatrixDisplay.from_estimator(tree_clf, X_test, y_test)
    # plt.show()
    #
    #
    # bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=150, bootstrap=True, n_jobs=-1)
    # bag_clf.fit(X_train, a_train)
    # prediction = bag_clf.predict(X_test)
    # accuracy = accuracy_score(a_test, prediction)
    # print(accuracy)
    # ConfusionMatrixDisplay.from_estimator(bag_clf, X_test, a_test)
    # plt.show()
    #
    # clf = svm.SVC()
    # clf.fit(X_train, y_train)
    # prediction = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, prediction)
    # print(accuracy)
    # ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    # plt.show()
    #
    # bag_clf = BaggingClassifier(svm.SVC(), n_estimators=1000, max_samples=750, bootstrap=True, n_jobs=-1)
    # bag_clf.fit(X_train, y_train)
    # prediction = bag_clf.predict(X_test)
    # accuracy = accuracy_score(y_test, prediction)
    # print(accuracy)
    # ConfusionMatrixDisplay.from_estimator(bag_clf, X_test, y_test)
    # plt.show()

    mlp = MLPClassifier(hidden_layer_sizes=(400),solver='adam',activation='logistic',max_iter=400)
    mlp.fit(X_train, n_train)
    prediction = mlp.predict(X_test)
    accuracy = accuracy_score(n_test, prediction)
    print(accuracy)
    ConfusionMatrixDisplay.from_estimator(mlp, X_test, n_test)
    plt.show()

    import pickle

    # save
    with open('neutral_model.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    # load
    with open('neutral_model.pkl', 'rb') as f:
        clf2 = pickle.load(f)

    clf2.predict(X[0:1])


    #
    # x_axis = []
    # y_axis = []
    # for k in range(1,30):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     prediction = knn.predict(X_test)
    #     accuracy = accuracy_score(y_test, prediction)
    #     x_axis.append(k)
    #     y_axis.append(accuracy)
    #
    # plt.plot(x_axis, y_axis)
    # plt.scatter(x_axis, y_axis)
    # plt.title('K-NN: How Accuracy Changes with K')
    # plt.xlabel('K Number')
    # plt.ylabel('Accuracy')
    # plt.show()

# spectral_features = {'chroma_stft': [], 'chroma_cqt': [], 'chroma_cens': [],
#                      'melspectrogram': [], 'mfcc': [], 'rms': [], 'spectral_centroid': [],
#                      'spectral_bandwidth': [], 'spectral_contrast': [],
#                      'spectral_flatness': [], 'spectral_rolloff': [], 'poly_features': [],
#                      'tonnetz': [], 'zero_crossing_rate': []}

# y, sr = lb.load(files[0])
#
# spectral_features['chroma_stft'] = ft.chroma_stft(y, sr)
# spectral_features['chroma_cqt'] = ft.chroma_cqt(y, sr)
# spectral_features['chroma_cens'] = ft.chroma_cens(y, sr)
# spectral_features['melspectrogram'] = ft.melspectrogram(y, sr)
# spectral_features['mfcc'] = ft.mfcc(y, sr)
# spectral_features['rms'] = ft.rms(y, sr)
# spectral_features['spectral_centroid'] = ft.spectral_centroid(y, sr)
# spectral_features['spectral_bandwidth'] = ft.spectral_bandwidth(y, sr)
# spectral_features['spectral_contrast'] = ft.spectral_contrast(y, sr)

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