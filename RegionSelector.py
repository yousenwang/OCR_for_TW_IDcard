# https://www.youtube.com/watch?v=cUOcY9ZpKxw&ab_channel=Murtaza%27sWorkshop-RoboticsandAI

import cv2
import numpy as np
import pytesseract
import os
import random

# scale = 0.5
scale = 1
circles = []
counter = 0
counter2 = 0
point1 = []
point2 = []
myPoints = []
myColor = []

def mousePoints(event, x, y, flags, params):
    global counter, point1, point2, counter2, circles, myColor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            point1 = int(x // scale), int(y // scale)
            counter += 1
            myColor = (
                random.randint(0, 2) * 200, 
                random.randint(0, 2) * 200, 
                random.randint(0, 2) * 200
                )
        elif counter == 1:
            point2 = int(x//scale), int(y//scale)
            type = input("Enter Type")
            name = input("Enter Name")
            myPoints.append([point1, point2, type, name])
            counter = 0
            print(myPoints)
        circles.append([x, y, myColor])
        counter2 += 1



img = cv2.imread("./TWnationalIDcard.jpg")

img = cv2.resize(img, (0,0), None, scale, scale)


while True:
    for x, y, color in circles:
        cv2.circle(img, (x,y), 3, color, cv2.FILLED)
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", mousePoints)
    if cv2.waitKey(1) == ord('s'):
        print(myPoints)
        break

print(myPoints)