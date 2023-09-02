#!/usr/bin/env python
# coding: utf-8

# In[29]:


import cv2
import numpy as np
import math


# In[33]:


# changing the image color 
img = cv2.imread("traffic_light_red.jpg", 1)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appling Bilateral filter  
sigma_color = 10.0
sigma_space = 10.0
filtered_image = cv2.bilateralFilter(gray_img, -1, sigma_color, sigma_space)

# Detect circle 
circles = cv2.HoughCircles(filtered_image, cv2.HOUGH_GRADIENT, 2, 30, param1=200, param2=50, minRadius=10, maxRadius=50)
if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        k = i[0]
        j = i[1]
        B = img[j, k, 0]
        G = img[j, k, 1]
        R = img[j, k, 2]

        if R - B > 20 and R - G > 20 and R > 100:
            print("Stop due to the RedLight")
        if G - R > 20 and G - B > 20 and G > 100:
            print("Detected GreenLight")


# In[ ]:




