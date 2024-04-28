import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

def initialize_capture():
    cap = cv2.VideoCapture(0)
    return cap

def load_classifier():
    model_path = r"C:\Users\altha\OneDrive\Desktop\clg project\Sign-Language-Interpreter-using-Deep-Learning\Model\keras_model.h5"
    labels_path = r"C:\Users\altha\OneDrive\Desktop\clg project\Sign-Language-Interpreter-using-Deep-Learning\Model\labels.txt"
    classifier = Classifier(model_path, labels_path)
    return classifier

def get_labels():
    digits = list(map(str, range(10)))  # Convert digits to strings
    letters = [chr(ord('a') + i) for i in range(26)]
    phrases = ["hello", "thank you"]
    return digits + letters + phrases

def resize_crop(imgCrop, aspectRatio, imgSize):
    if aspectRatio > 1:
        k = imgSize / imgCrop.shape[0]
        wCal = math.ceil(k * imgCrop.shape[1])
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
        k = imgSize / imgCrop.shape[1]
        hCal = math.ceil(k * imgCrop.shape[0])
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgWhite[hGap:hCal + hGap, :] = imgResize
    return imgWhite

def main():
    cap = initialize_capture()
    classifier = load_classifier()
    labels = get_labels()
    offset = 20
    imgSize = 300

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit if the video capture fails

        imgOutput = img.copy()
        hands, img = HandDetector(maxHands=1).findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            aspectRatio = h / w
            imgWhite = resize_crop(imgCrop, aspectRatio, imgSize)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        cv2.imshow('Image', imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit loop if 'q' is pressed

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
