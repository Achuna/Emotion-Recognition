import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
import pickle
import seaborn as sns
from numpy import loadtxt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def train_on_mfccs(e1, label):

    df = pd.read_csv("data.csv")

    positives = df.loc[df['0.1'].isin(e1)]
    negatives = df.loc[~df['0.1'].isin(e1)]
    # negatives = negatives.sample(n=positives.shape[0])

    positives.iloc[:]['0.1'] = 'is_'+label
    negatives.iloc[:]['0.1'] = 'not_'+label

    frames = [positives, negatives]

    newdf = pd.concat(frames)
    newdf.reset_index(inplace=True)
    newdf.drop(columns=['index'], inplace=True)

    print(newdf.shape)
    print(newdf.head())

    print(positives.shape)
    print(positives.head())

    print(negatives.shape)
    print(negatives.head())


    X = newdf.iloc[:, :-1]
    y = newdf.iloc[:, -1:]

    print("Original Data")
    print("X: ", X.shape)
    print("y: ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("==============\nTraining Set")
    print("X: ", X_train.shape)
    print("y: ", y_train.shape)
    print("==============\nTest Set")
    print("X: ", X_test.shape)
    print("y: ", y_test.shape)


    mlp = MLPClassifier(hidden_layer_sizes=(500),solver='adam',activation='logistic',max_iter=300)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(accuracy)
    ConfusionMatrixDisplay.from_estimator(mlp, X_test, y_test)
    plt.show()

    # knn = KNeighborsClassifier(n_neighbors=4)
    # knn.fit(X_train, y_train)
    # ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
    # plt.show()

    # print(mlp.predict_proba(X[0:1]))
    # print(mlp.predict(X[0:1]))



  # learning curve model overfitting under fitting

    # save
    with open(label+'_model.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    # load
    # with open(e1+'_model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
    #
    # print(clf2.predict_proba(X[0:1]))


def train_on_images():
    img_width, img_height = 496, 369

    files = np.asarray(glob.glob("unlabeled_spectrograms/*"))
    df = pd.DataFrame(columns=['image'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    print("Original Data")
    print("X: ", X.shape)
    print("y: ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("==============\nTraining Set")
    print("X: ", X_train.shape)
    print("y: ", y_train.shape)
    print("==============\nTest Set")
    print("X: ", X_test.shape)
    print("y: ", y_test.shape)

    lb = LabelEncoder()
    # change labels to categorical
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))

emotions = {'01': "neutral", '02': "calm", '03': "happy", '04': "sad",
                '05': "angry", '06': "fearful", '07': "disgust", '08': "surprised"}

train_on_mfccs(['fearful'], "fearful")