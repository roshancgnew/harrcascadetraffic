# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:44:25 2021

@author: roshacho
"""

import cv2
import os 
wdir = "C:/Users/roshacho/Documents/opencv_tasks/facedetect/finaltraining/dasar_haartrain"
os.chdir(wdir)
cascadefile= "myhaar.xml"
signal_cascade=cv2.CascadeClassifier(cascadefile)
###path of cascade file

# imgpath = os.path.join(os.getcwd() + r'\positive\rawdata\\' )
listfiles = []
imgpa = "C:/Users/roshacho/Documents/opencv_tasks/facedetect/finaltraining/dasar_haartrain/positive/rawdata/"
for path in os.listdir(imgpa):
    full_path = os.path.join(imgpa, path)
    if os.path.isfile(full_path):
        listfiles.append(full_path)
        # print(full_path)
    
img= cv2.imread(listfiles[6]) #working
# img= cv2.imread("C:/Users/roshacho/Documents/opencv_tasks/facedetect/pole.jpg")
# img= cv2.imread("C:/Users/roshacho/Documents/opencv_tasks/facedetect/finaltraining/dasar_haartrain/positive/rawdata/imag2light.bmp")
cv2.imshow("ori",img)
cv2.waitKey(0)

# resized = cv2.resize(img,(400,200))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

signal=signal_cascade.detectMultiScale(gray,1.5,3)

if len(signal) == 0:
    print("No Signal found")
else:
    print("Signal found successfully")
    for(x,y,w,h) in signal:
        gray=cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,255),2)
        cv2.imshow('Signal Detection', gray)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()