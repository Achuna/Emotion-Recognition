import pickle
import threading
import warnings
from time import sleep
from tkinter import *

import librosa.display
import numpy as np
import sounddevice as sd
from cv2 import cv2
from fer import FER
from matplotlib import pyplot as plt, animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from Video.video import VideoCamera

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class Dashboard:
    def __init__(self, stream_source=0):
        # Styles
        txt_color = "#f0f0f0"
        bg_color = "#1f1f1f"
        btn_color = "#4f4f4f"

        # Internal variables
        self.detector = FER()
        self.vid_source = stream_source
        self.vid_stream = VideoCamera(source=self.vid_source)
        self.frame = self.vid_stream.next_frame()
        self.empty_frame = np.full(shape=(480, 640, 3), fill_value=0x1f, dtype=np.uint8)
        self.bb = None
        self.mfccs = None

        print("Loading & Preparing Video Model...")
        # Preload emotion recognition model for video input
        self.detector.detect_emotions(np.zeros((48, 48, 3), dtype=np.uint8),
                                      face_rectangles=[(0, 0, 48, 48)])

        # Create window
        self.root = Tk()
        self.root.title("Emotion Recognition")
        canvas = Canvas(self.root, width=1280, height=720, bg=bg_color)
        canvas.grid(columnspan=3)

        # Create video component
        frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        self.vid_panel = Label(canvas, width=640, height=480, image=frame)
        self.vid_panel.image = frame
        self.vid_panel.grid(columnspan=2, row=0, column=0, padx=10, pady=10)

        # Initialize plots
        self.fig, axs = plt.subplots(2)
        self.fig.subplots_adjust(hspace=.8)
        self.fig.patch.set_facecolor(bg_color)
        self.vid_bars = axs[0].bar(emotions, [0] * len(emotions))
        axs[0].set_facecolor(bg_color)
        axs[0].set_title("Video Predictions", color=txt_color)
        axs[0].set_xlabel("", color=txt_color)
        axs[0].set_ylabel("Probability", color=txt_color)
        axs[0].set_ylim((0, 1))
        for spine in axs[0].spines:
            axs[0].spines[spine].set_color(txt_color)
        axs[0].tick_params(axis='x', colors=txt_color)
        axs[0].tick_params(axis='y', colors=txt_color)

        self.audio_bars = axs[1].bar(emotions, [0] * len(emotions), color="red")
        axs[1].set_facecolor(bg_color)
        axs[1].set_title("Audio Predictions", color=txt_color)
        axs[1].set_xlabel("Emotions", color=txt_color)
        axs[1].set_ylabel("Probability", color=txt_color)
        axs[1].set_ylim((0, 1))
        for spine in axs[1].spines:
            axs[1].spines[spine].set_color(txt_color)
        axs[1].tick_params(axis='x', colors=txt_color)
        axs[1].tick_params(axis='y', colors=txt_color)

        # Create plot component
        fig_canvas = FigureCanvasTkAgg(self.fig, master=canvas)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().grid(row=0, column=3, padx=10, pady=10)

        # Create buttons
        self.vid_toggle = Button(canvas, text="Toggle Webcam", command=self.toggle_video,
                                 fg=txt_color, bg=btn_color)
        self.vid_toggle.grid(row=1, column=0, padx=10, pady=(0, 10))
        self.audio_toggle = Button(canvas, text="Toggle Microphone", command=self.toggle_audio,
                                   fg=txt_color, bg=btn_color)
        self.audio_toggle.grid(row=1, column=1, padx=10, pady=(0, 10))

        # Create events and threads
        self.app_stop_event = threading.Event()
        self.vid_stop_event = threading.Event()
        self.audio_stop_event = threading.Event()

        self.vid_thread = threading.Thread(target=self.stream_frames, args=())
        self.vid_thread.start()

        self.audio_thread = threading.Thread(target=self.stream_audio, args=())
        self.audio_thread.start()

        # Exit logic
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def display_frame(self, frame):
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        self.vid_panel.configure(image=frame)
        self.vid_panel.image = frame
        self.vid_panel.update()

    def stream_frames(self):
        # Loop until the window is closed
        while not self.app_stop_event.is_set():
            # Only use webcam when allowed
            if not self.vid_stop_event.is_set():
                # Initiate webcam if it had been disabled prior
                if self.vid_stream is None:
                    self.vid_stream = VideoCamera(self.vid_source)

                self.frame = self.vid_stream.next_frame()
                frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
                output = self.detector.find_faces(frame)
                bb = output[0].tolist() if len(output) > 0 else None
                if bb is not None:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
                                          color=(121, 49, 132), thickness=3)
                self.bb = bb
                self.display_frame(frame)
            # Stop using webcam when no longer allowed
            else:
                # Release webcam, display an empty frame in its place
                if self.vid_stream is not None:
                    self.vid_stream = None
                    self.display_frame(self.empty_frame)
                # Purposely-induced delay that prevents resource-hogging when disabled
                sleep(0.1)

    def stream_audio(self):
        # Loop until the window is closed
        while not self.app_stop_event.is_set():
            # Only use audio when allowed
            if not self.audio_stop_event.is_set():
                sample_rate = 44100  # Sample rate
                seconds = 3  # Duration of recording

                sd.default.latency = ('low', 'low')
                rec = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)
                sd.wait()  # Wait until recording is finished

                y, _ = librosa.effects.trim(rec.flatten())  # Trim leading and trailing silence from an audio signal.
                self.mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T, axis=0)

            else:
                # Release audio recording
                # Purposely-induced delay that prevents resource-hogging when disabled
                sleep(0.1)

    def update_predictions(self, i):
        self.update_audio_predictions(i)
        self.update_video_predictions()

    def update_audio_predictions(self, i):
        if not self.audio_stop_event.is_set() and self.mfccs is not None:
            predictions = []
            if i % 3 == 0:
                mfccs = self.mfccs
                predictions = [angryModel.predict_proba([mfccs])[0][0],
                               disgustModel.predict_proba([mfccs])[0][0],
                               fearModel.predict_proba([mfccs])[0][0],
                               happyModel.predict_proba([mfccs])[0][0],
                               sadModel.predict_proba([mfccs])[0][0],
                               surprisedModel.predict_proba([mfccs])[0][0],
                               calmModel.predict_proba([mfccs])[0][0]]
                predictions = [x / sum(predictions) for x in predictions]
        else:
            predictions = [0] * len(emotions)
        for bar, p in zip(self.audio_bars, predictions):
            bar.set_height(p)

    def update_video_predictions(self):
        if not self.vid_stop_event.is_set() and self.bb is not None:
            predictions = self.detector.detect_emotions(self.frame, [self.bb])[0]["emotions"]
            predictions = list(predictions.values())
        else:
            predictions = [0] * len(emotions)
        for bar, p in zip(self.vid_bars, predictions):
            bar.set_height(p)

    def toggle_audio(self):
        if self.audio_stop_event.is_set():
            self.audio_stop_event.clear()
            print("Enabling Microphone")
        else:
            self.audio_stop_event.set()
            print("Disabling Microphone")

    def toggle_video(self):
        if self.vid_stop_event.is_set():
            self.vid_stop_event.clear()
            print("Enabling Webcam")
        else:
            self.vid_stop_event.set()
            print("Disabling Webcam")

    def on_close(self):
        self.app_stop_event.set()
        self.vid_stop_event.set()
        self.audio_stop_event.set()
        self.vid_thread.join()
        self.audio_thread.join()
        self.vid_stream = None
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    # Filter annoying warnings from Scikit-learn
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading Audio Models...")

    with open('Audio/Neural Networks/angry_model.pkl', 'rb') as f:
        angryModel = pickle.load(f)

    with open('Audio/Neural Networks/happy_model.pkl', 'rb') as f:
        happyModel = pickle.load(f)

    with open('Audio/Neural Networks/calm_model.pkl', 'rb') as f:
        calmModel = pickle.load(f)

    with open('Audio/Neural Networks/sad_model.pkl', 'rb') as f:
        sadModel = pickle.load(f)

    with open('Audio/Neural Networks/disgust_model.pkl', 'rb') as f:
        disgustModel = pickle.load(f)

    with open('Audio/Neural Networks/surprised_model.pkl', 'rb') as f:
        surprisedModel = pickle.load(f)

    with open('Audio/Neural Networks/fearful_model.pkl', 'rb') as f:
        fearModel = pickle.load(f)

    print("Loading Dashboard...")
    app = Dashboard()
    ani = animation.FuncAnimation(app.fig, app.update_predictions, interval=1000)
    print(">> Launching Dashboard! <<")
    app.root.mainloop()
