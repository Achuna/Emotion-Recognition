import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pickle
import sounddevice as sd
import soundfile as sf

# data, sampling_rate = librosa.load('achuna_scream.wav')
# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(data, sr=sampling_rate)
# plt.show()


# load

print("Loading Models...")

with open('Neural Networks/angry_model.pkl', 'rb') as f:
    angryModel = pickle.load(f)

with open('Neural Networks/happy_model.pkl', 'rb') as f:
    happyModel = pickle.load(f)

with open('Neural Networks/calm_model.pkl', 'rb') as f:
    calmModel = pickle.load(f)

with open('Neural Networks/sad_model.pkl', 'rb') as f:
    sadModel = pickle.load(f)

with open('Neural Networks/disgust_model.pkl', 'rb') as f:
    disgustModel = pickle.load(f)

with open('Neural Networks/surprised_model.pkl', 'rb') as f:
    surprisedModel = pickle.load(f)

with open('Neural Networks/fearful_model.pkl', 'rb') as f:
    fearModel = pickle.load(f)



sample_rate = 44100  # Sample rate
seconds = 5  # Duration of recording

print("TALK NOW...")
sd.default.latency = ('low', 'low')
X = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()  # Wait until recording is finished
write('output.wav', sample_rate, X)  # Save as WAV file

# import speech_recognition as sr
# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Say something!")
#     audio = r.listen(source)

# X, sample_rate = librosa.load('output.wav', res_type='kaiser_fast')
# sample_rate = np.array(sample_rate)

y, _ = librosa.effects.trim(X.flatten())  # Trim leading and trailing silence from an audio signal.
mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)

print(mfccs)

def plotEmotion(mfccs):
    categories = ['Angry', 'Happy', 'Calm', 'Sad', 'Disgust', 'Surprised', 'Fear']
    emotion_values = []

    # print(angryModel.predict_proba([mfccs])[0][0])
    # print(happyModel.predict_proba([mfccs])[0][0])
    # print(calmModel.predict_proba([mfccs])[0][0])
    # print(sadModel.predict_proba([mfccs])[0][0])
    # print(disgustModel.predict_proba([mfccs])[0][0])
    # print(surprisedModel.predict_proba([mfccs])[0][0])
    # print(fearModel.predict_proba([mfccs])[0][0])

    emotion_values.append(angryModel.predict_proba([mfccs])[0][0])
    emotion_values.append(happyModel.predict_proba([mfccs])[0][0])
    emotion_values.append(calmModel.predict_proba([mfccs])[0][0])
    emotion_values.append(sadModel.predict_proba([mfccs])[0][0])
    emotion_values.append(disgustModel.predict_proba([mfccs])[0][0])
    emotion_values.append(surprisedModel.predict_proba([mfccs])[0][0])
    emotion_values.append(fearModel.predict_proba([mfccs])[0][0])

    plt.bar(categories, emotion_values, color='blue', width=0.4)
    plt.show()


plotEmotion(mfccs)

# print(angryModel.predict([mfccs]))
# print(angryModel.predict_proba([mfccs])[0][0])
