import numpy as np
import pandas as pd
import re, sys, os, csv, keras, pickle
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
# from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint

MAX_NB_WORDS = 56000 # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 200 # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "glove.twitter.27B.200d.txt"
print("Loaded Parameters:\n", MAX_NB_WORDS,MAX_SEQUENCE_LENGTH+5, VALIDATION_SPLIT,EMBEDDING_DIM,"\n", GLOVE_DIR)

texts, labels = [], []
print("Reading from csv file...", end="")
with open('emotion_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        texts.append(row[0])
        labels.append(row[1])
print("\nDone!")

