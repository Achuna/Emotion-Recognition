import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import pandas as pd
from keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical
import seaborn as sns
from numpy import loadtxt

df = pd.read_csv("data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

print("Original Data")
print("X: ", X.shape)
print("y: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

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


# Changing dimension for CNN model
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

print("================\nInput for Model")
print(x_traincnn.shape)
print(y_train.shape)

model = Sequential([
    Conv1D(256, 5,padding='same', input_shape=(40,1), activation='relu'),
    Conv1D(128, 5,padding='same', activation='relu'),
    Dropout(0.1),
    MaxPooling1D(pool_size=(8)),
    Conv1D(128, 5,padding='same', activation='relu'),
    Conv1D(128, 5,padding='same', activation='relu'),
    Flatten(),
    Dense(3, activation='softmax'),
])

# opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


cnnhistory = model.fit(x_traincnn, y_train, batch_size=64, epochs=500, validation_data=(x_testcnn, y_test))
#
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
#
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
#
# cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=2000, validation_data=(x_testcnn, y_test))
#
# model_name = 'Emotion_Voice_Detection_Model_60acc.h5'
# save_dir = ''
# # Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)
#
# plt.plot(cnnhistory.history['loss'])
# plt.plot(cnnhistory.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
#
# plt.plot(cnnhistory.history['acc'])
# plt.plot(cnnhistory.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()



