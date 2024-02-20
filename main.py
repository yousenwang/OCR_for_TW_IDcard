import cv2
import numpy as np
import pytesseract
import os

per = 25
pixelThreshold = 500

roi = [[(49, 101), (123, 126), 'text', 'Name']]


imgQ = cv2.imread('./TWnationalIDcard.jpg')

h, w, c = imgQ.shape

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)


# path = "UserForms"
path = "input_images"
myPicList = os.listdir(path)
print(myPicList)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/"+y)

    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(bf.match(des2, des1))
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(
        img,
        kp2,
        img,
        kp1,
        good[:100],
        None,
        flags = 2
    )

    srcPoints = np.float32(
        [kp2[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dstPoints = np.float32(
        [kp1[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    for x, r in enumerate(roi):
        cv2.rectangle(
            imgMask,
            (r[0][0], r[0][1]),
            (r[1][0], r[1][1]),
            (0, 255, 0),
            cv2.FILLED
        )
        
        imgShow = cv2.addWeighted(imgShow, 0.9, imgMask, 0.1, 0)

        # imgCrop = imgScan[r[0][1]: r[1][1], r[0][0]:r[1][0]]

        # if r[2] == 'text':
    
    cv2.imshow(y+"2", imgShow)
    cv2.waitKey(0)

