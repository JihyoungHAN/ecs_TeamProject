import cv2
import numpy as np
import math
import os
from PIL import Image
import io 

def classify(img):
    
    #loading YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    #bringing the image
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    #Detecting Objects 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 9: 
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # co-ordinate
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                #extract traffic light image
                img_trim = img[y:y+h, x:x+w]
                if img_trim is not None and not img_trim.size == 0:
                    cv2.imwrite('tl_trim_image.jpg', img_trim)
                    org_img = cv2.imread('tl_trim_image.jpg')
                    org_img = cv2.resize(org_img, dsize = (200, 400))
                    
    img_path = "tl_trim_image.jpg"
    if os.path.exists(img_path): 
        # changing the image color 
        img = cv2.imread("tl_trim_image.jpg", 1)
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
                B = float(img[j, k, 0])
                G = float(img[j, k, 1])
                R = float(img[j, k, 2])
                if R - B > 10 and R - G > 10 and R > 100:
                    return 100
                if G - R > 10 and G - B > 10 and G > 100:
                    return 110
    else: 
        return 111
    




