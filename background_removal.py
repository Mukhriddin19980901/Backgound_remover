#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvzone
import cv2
import matplotlib.pyplot as plt
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import os


# In[10]:


video = cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)
segment = SelfiSegmentation() # default holati 1 , 0 ham ishlatsa boladi 
fonfps = cvzone.FPS()
list_rasm = os.listdir("images")
im_lists = []
for im in list_rasm:
    rasm = cv2.imread(f"images/{im}")
    im_lists.append(rasm)
index = 0   
while True : 
    _,rasm = video.read()
    natija = segment.removeBG(rasm,im_lists[index],threshold = 0.6)
    
    imgQosh = cvzone.stackImages([rasm,natija],2,1)
    _,imgQosh = fonfps.update(imgQosh,color = (0,133,255))
    cv2.imshow("olingan",imgQosh)
    key = cv2.waitKey(1) 
    if key == ord('f'):
        if index < len(im_lists)-1:
            index += 1
    elif key == ord('b'):
        if index > 0:
            index -= 1
    elif key == ord("q"):
        break
video.release()
cv2.destroyAllWindows()


# In[ ]:




