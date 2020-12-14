import sys
import numpy as np
import imutils
import cv2
from keras.models import load_model
from base64 import b64decode

class AslDetector:
    def __init__(self):
        self.model = load_model("ASL_MNIST_CNN_20.h5")
        self.lookUpTable1 = np.empty((1,256), np.uint8)
        for i in range(256):
            self.lookUpTable1[0,i] = np.clip(pow(i / 255.0, 0.5) * 255.0, 0, 255)
        self.lookUpTable2 = np.empty((1,256), np.uint8)
        for i in range(256):
            self.lookUpTable2[0,i] = np.clip(pow(i / 255.0, 32) * 255.0, 0, 255) 
            # greate the 16, higher contrast
    
    def detect(self, frame):
        ####################################################
        # Frame processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.LUT(frame, self.lookUpTable1)
        frame = cv2.GaussianBlur(frame, (31,31),0)
        ret2, frame_mask1 = cv2.threshold(frame, 0 ,255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame = cv2.LUT(frame, self.lookUpTable2)
        frame_mask = frame_mask1 == 255
        frame = 255 - (255-frame) * frame_mask
        frame = cv2.resize(frame,(28,28))
        ####################################################
        
        ####################################################
        img = frame.reshape(-1,28,28,1)
        y_pred = self.model.predict(img)
        num = np.argmax(y_pred)
        y_prob = y_pred[0,num]
        if num <= 8:
            letter = chr(num + 65)
        else:
            letter = chr(num + 66)
        return y_prob, letter