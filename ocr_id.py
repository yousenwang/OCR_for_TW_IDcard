import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import pytesseract
import cv2
import numpy as np


class IDCardOCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Taiwan National ID Card OCR')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.result_label = QLabel(self)

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.openImage)

        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def openImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if filename:
            self.processImage(filename)

    def processImage(self, filename):
        img = cv2.imread(filename)

        ##

        per = 25
        pixelThreshold = 500

        roi = [
            [(49, 101), (123, 126), 'text', 'Name (Chinese)'],
            [(47, 125), (195, 146), 'text', 'Name (English)'],
            [(48, 199), (162, 220), 'text', 'Date of Birth'],
            [(178, 200), (190, 218), 'text', 'Sex'],
            [(261, 200), (319, 220), 'text', 'Authority (English)'],
            [(50, 241), (153, 263), 'text', 'ID No.'],
            [(168, 247), (236, 261), 'text', 'Date of Issue'],
            [(251, 244), (320, 261), 'text', 'Date of Expiry'],
            [(320, 66), (435, 217), 'box', 'ID Photo'],
            [(323, 215), (436, 248), 'box', 'ID Signature']
            ]

        imgQ = cv2.imread('./TWnationalIDcard.jpg')

        h, w, c = imgQ.shape

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)

        ##
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

        myData = {}

        for x, r in enumerate(roi):
            cv2.rectangle(
                imgMask,
                (r[0][0], r[0][1]),
                (r[1][0], r[1][1]),
                (0, 255, 0),
                cv2.FILLED
            )
            
            imgShow = cv2.addWeighted(imgShow, 0.9, imgMask, 0.1, 0)

            imgCrop = imgScan[r[0][1]: r[1][1], r[0][0]:r[1][0]]

            config = r'--oem 3'
            if r[2] == 'text':
                extracted_entity = pytesseract.image_to_string(
                    imgCrop,
                    # config = config,
                    lang='chi_tra+eng'
                    )
                # r[3] is the name
                print(f'{r[3]}: {extracted_entity}')
                mapping = {r[3]: extracted_entity.strip()}
                myData.update(mapping)

        self.result_label.setText(str(myData))


        # Convert image to QPixmap for display
        image = cv2.cvtColor(imgShow, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytesPerLine = ch * w
        qImg = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Display the image with bounding boxes
        self.image_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IDCardOCRApp()
    window.show()
    sys.exit(app.exec_())
