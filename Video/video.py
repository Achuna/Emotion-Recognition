from cv2 import cv2
from fer import FER


def capture_live_data(detector, video_capture):
    _, frame = video_capture.read()
    output = detector.detect_emotions(frame)
    if len(output) > 0:
        return frame, output[0]["box"], output[0]["emotions"]
    return frame, None, None


def display_frame(frame, bounding_box, predictions, mirror=False):
    if mirror:
        frame = cv2.flip(frame, 1)

    if bounding_box is not None and predictions is not None:
        if mirror:
            bb = (frame.shape[1] - bounding_box[0] - bounding_box[2], bounding_box[1],
                  bounding_box[2], bounding_box[3])
        else:
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
    frame_count = 0
    frame_cap = 5

    while True:
        frame_count += 1
        frame, bounding_box, predictions = capture_live_data(detector, vid_feed)
        display_frame(frame, bounding_box, predictions, mirror=True)
        # Press 'Q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_count >= frame_cap:
            print(predictions)
    vid_feed.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
