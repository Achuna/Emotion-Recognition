import text2emotion as te

import speech_recognition as sr

text = ""
r = sr.Recognizer()

audio = 'testing.wav'

with sr.AudioFile(audio) as source:
    audio = r.record(source)
try:
    text = r.recognize_google(audio)
    print(text)
except Exception as e:
    print(e)

print(te.get_emotion(text))