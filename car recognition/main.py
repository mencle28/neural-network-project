# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:17:22 2021

@author: TRABET Clément
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:09:20 2021

@author: TRABET Clément
"""

#https://www.youtube.com/watch?v=O3b8lVF93jU
import cv2      #open CV
from tracker import *
import math as math

haar_cascade=cv2.CascadeClassifier('haar_car.xml')
compteur=0
simplifieur=0

def rescaleFrame(frame, scale=1):
    #works for everything
        width=int(frame.shape[1]*scale)
        height=int(frame.shape[0]*scale)

        dimensions=(width,height)
        return cv2.resize(frame,dimensions, interpolation=cv2.INTER_AREA)


# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.avi") 
# Object detection from Stable camera

object_detector = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=50)
iid=0
while True:
    iid=iid+1                         #each loop takes one pict
    ret, frame = cap.read()
    
    if ret:
        
        frame=rescaleFrame(frame)
        
        #1280 x 720
        #height, width, _ = frame.shape()
        # Extract Region of interest
        # highway roi = frame[340: 720,500: 800]
        roi=frame [250:620,500:800]
        roi_car=frame [250:620,500:800]
        roi_car=cv2.cvtColor(roi_car, cv2.COLOR_BGR2GRAY)
        #               heights      width

        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for cnt in contours:
        # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if 900 >area > 600:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                
                

                detections.append([x, y, w, h])
        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
    #création de boite
        test=""
        for box_id in boxes_ids:
            
            x, y, w, h, id = box_id
            if (id>= compteur) and (simplifieur==2):
                #simplifieur=1
                compteur=id
                face_rect = haar_cascade.detectMultiScale(roi_car, scaleFactor=1.04, minNeighbors=2)
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                for (x2,y2,w2,h2) in face_rect:
                    cx2 = (x2 + x2 + w2) // 2
                    cy2 = (y2 + y2 + h2) // 2
                    dist = math.hypot(cx - cx2, cy - cy2)
                    if dist<150:
                        test="car"
                        print("test réussi")
                    #cv2.rectangle(roi, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 3)
                cv2.putText(roi, str(id) + test, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 250, 0), 3)
                
                print("test réalisé")
            else:
                simplifieur=2
                cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        #cv2.imshow("roi", roi)
        cv2.imshow("Frame", frame)
        #cv2.imshow("Frame car", roi_car)
        #cv2.imshow("Mask", mask)
        
        if (iid % 20==0):
            cv2.imwrite("image"+str(iid)+".png", frame)
        key = cv2.waitKey(30)
        if cv2.waitKey(20) & 0xFF==ord('d'):
            break
    else:
        print("problème de fichier")
        break
            
cap.release()
cv2.destroyAllWindows()


