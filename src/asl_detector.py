import sys
import numpy as np
import imutils
import cv2
from keras.models import load_model
# from google.colab.patches import cv2_imshow

# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# from base64 import b64decode

class AslDetector:
    def __init__(self):
        self.model = load_model("/gdrive/MyDrive/504/CNN.h5")
    
    def detect(self, img):
        img_width = img.shape[1]
        img_height = img.shape[0]
        cut_point1 = int((img_width - img_height ) / 2) - 1
        cut_point2 = int((img_width + img_height ) / 2)
        img = img[:, cut_point1:cut_point2 ,:]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(28,28))
        img = img.reshape(-1,28,28,1)
        y_pred = self.model.predict(img)
        return y_pred