import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
import pytesseract
import cv2

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
        image = cv2.imread(filename)
        text = pytesseract.image_to_string(image, lang='chi_tra+eng')
        self.result_label.setText(text)
        pixmap = QPixmap(filename)
        self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IDCardOCRApp()
    window.show()
    sys.exit(app.exec_())
