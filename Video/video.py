from cv2 import cv2
from fer import FER


def capture_live_data(detector, video_capture):
    _, frame = video_capture.read()
    output = detector.detect_emotions(frame)
    return frame, output[0]["box"], output[0]["emotions"]


def display_frame(frame, bounding_box, predictions):
    if len(bounding_box and predictions) > 0:
        bb = bounding_box

        # Color is in BGR format, not RGB! (Blame OpenCV)
        cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
                      color=(121, 49, 132), thickness=3)

        for i, (emotion, prob) in enumerate(predictions.items()):
            y_offset = 20
            result = "%s: %.3f%%" % (emotion, 100 * prob)
            cv2.putText(frame, result, (bb[0], bb[1] + bb[3] + i * y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0),
                        thickness=5, lineType=cv2.LINE_AA)
            cv2.putText(frame, result, (bb[0], bb[1] + bb[3] + i*y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(230, 230, 230),
                        thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("Live Output", frame)


def main_loop():
    detector = FER()
    vid_feed = cv2.VideoCapture(0)
    while True:
        display_frame(*capture_live_data(detector, vid_feed))
        # Press 'Q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid_feed.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
