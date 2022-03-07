from cv2 import cv2


class VideoCamera:
    def __init__(self, source=0):
        self.camera = cv2.VideoCapture(source)  # Captures webcam feed by default
        self.frame = None

    def __del__(self):
        self.camera.release()

    def next_frame(self):
        _, self.frame = self.camera.read()
        return self.frame

    def current_frame(self):
        return self.frame
