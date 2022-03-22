import threading
from time import sleep
from tkinter import *

import numpy as np
from fer import FER
from matplotlib import pyplot as plt, animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from cv2 import cv2

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
        self.fig, vid_ax = plt.subplots()
        self.fig.patch.set_facecolor(bg_color)
        self.bars = vid_ax.bar(emotions, [0] * len(emotions))
        vid_ax.set_facecolor(bg_color)
        vid_ax.set_title("Video Predictions", color=txt_color)
        vid_ax.set_xlabel("Emotions", color=txt_color)
        vid_ax.set_ylabel("Probability of Expression", color=txt_color)
        vid_ax.set_ylim((0, 1))
        for spine in vid_ax.spines:
            vid_ax.spines[spine].set_color(txt_color)
        vid_ax.tick_params(axis='x', colors=txt_color)
        vid_ax.tick_params(axis='y', colors=txt_color)

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
        self.vid_thread = threading.Thread(target=self.stream_frames, args=())
        self.vid_thread.start()

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

    def update_predictions(self, _):
        if not self.vid_stop_event.is_set() and self.bb is not None:
            predictions = self.detector.detect_emotions(self.frame, [self.bb])[0]["emotions"]
            predictions = list(predictions.values())
        else:
            predictions = [0] * len(emotions)
        for bar, p in zip(self.bars, predictions):
            bar.set_height(p)

    def toggle_audio(self):
        # Insert code
        return

    def toggle_video(self):
        if self.vid_stop_event.is_set():
            self.vid_stop_event.clear()
        else:
            self.vid_stop_event.set()

    def on_close(self):
        self.app_stop_event.set()
        self.vid_stop_event.set()
        self.vid_thread.join()
        self.vid_stream = None
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    app = Dashboard()
    ani = animation.FuncAnimation(app.fig, app.update_predictions, interval=1000)
    app.root.mainloop()
