from fer import FER
import cv2

img = cv2.imread("Test_Image.jpg")
detector = FER()
print(detector.detect_emotions(img))