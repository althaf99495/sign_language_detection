import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
counter = 0

folder = r"C:\Users\altha\OneDrive\Desktop\clg project\Sign-Language-Interpreter-using-Deep-Learning\Data\Thank you"
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgcropshape = imgcrop.shape
        
        aspectratio = h/w
        
        if aspectratio > 1:
            k = imgsize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            imgresizeshape = imgresize.shape
            wGap = math.ceil((imgsize-wcal)/2)
            imgWhite[:, wGap: wcal + wGap] = imgresize
            
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            imgresizeshape = imgresize.shape
            hGap = math.ceil((imgsize-hcal)/2)
            imgWhite[hGap: hcal + hGap, :] = imgresize
            
        cv2.imshow("ImageCrop", imgcrop)
        cv2.imshow("ImageWhite", imgWhite)
        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("q"):
        break