import json
from keras.models import model_from_json
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd
import librosa
import glob
import matplotlib.pyplot as plt
import librosa.display

data, sampling_rate = librosa.load('achuna_scream.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
# plt.show()


json_file = open('./Neural Networks/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Neural Networks/model.h5")



X, sample_rate = librosa.load('achuna_scream.wav' ,res_type='kaiser_fast')
sample_rate = np.array(sample_rate)
y, _ = librosa.effects.trim(X)  # Trim leading and trailing silence from an audio signal.
mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T


print(livedf2)


twodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim,
                         batch_size=32,
                         verbose=1)

print(loaded_model.predict(twodim))
