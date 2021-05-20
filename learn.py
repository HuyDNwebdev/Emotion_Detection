import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap


emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('E:\Office Portable\Compressed\emoji-creator-project-code\model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "Neutral  ", 5: "    Sad    ", 6: "Surprised"}


emoji_dist={0:"emojis/angry.png",1:"emojis/disgusted.png",2:"emojis/fearful.png",3:"emojis/happy.png",4:"emojis/neutral.png",5:"emojis/sad.png",6:"emojis/surpriced.png"}



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1103, 817)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/NGHIA-CSK13/Desktop/Emoji/tải xuống.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 60, 1021, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(220, 310, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(720, 310, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.fromIMG = QtWidgets.QRadioButton(self.centralwidget)
        self.fromIMG.setGeometry(QtCore.QRect(80, 240, 95, 20))
        self.fromIMG.setObjectName("fromIMG")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(100, 260, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(589, 289, 451, 471))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.outLabel = QtWidgets.QLabel(self.groupBox)
        self.outLabel.setGeometry(QtCore.QRect(30, 60, 400, 400))
        self.outLabel.setText("")
        #self.outLabel.setPixmap(QtGui.QPixmap("emojis/angry.png"))
        self.outLabel.setObjectName("outLabel")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 170, 471, 591))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.fromVID = QtWidgets.QRadioButton(self.groupBox_2)
        self.fromVID.setGeometry(QtCore.QRect(40, 40, 121, 20))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.fromVID.setFont(font)
        self.fromVID.setObjectName("fromVID")
        self.inLabel = QtWidgets.QLabel(self.groupBox_2)
        self.inLabel.setGeometry(QtCore.QRect(40, 170, 400, 400))
        self.inLabel.setText("")
        self.inLabel.setObjectName("inLabel")
        self.groupBox.raise_()
        self.groupBox_2.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.fromIMG.raise_()
        self.pushButton.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1103, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.fromVID.toggled.connect(self.fVid)
        self.pushButton.clicked.connect(self.fIMG)
#from video
    def fVid(self):
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            bounding_box = cv2.CascadeClassifier('D:\Python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            num_faces = bounding_box.detectMultiScale(gray_frame)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # self.outLabel.setPixmap(QPixmap("emoji_dist[maxindex]"))
            cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
            self.outLabel.setPixmap(QtGui.QPixmap(emoji_dist[maxindex]))
            #emoji_dist[5]
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#form image
    def fIMG(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        pixmap = QtGui.QPixmap(imagePath)
        self.inLabel.setPixmap(pixmap)
        self.inLabel.resize(pixmap.size())
        self.inLabel.adjustSize()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "D-emoji 1.0"))
        self.label.setText(_translate("MainWindow", "ỨNG DỤNG NHẬN DIỆN CẢM XÚC VÀ TỰ ĐỘNG CHỌN EMOJI TƯƠNG ỨNG"))
        self.label_2.setText(_translate("MainWindow", "ẢNH ĐẦU VÀO"))
        self.label_3.setText(_translate("MainWindow", "BIỂU CẢM TƯƠNG ỨNG"))
        self.fromIMG.setText(_translate("MainWindow", "Từ ảnh"))
        self.pushButton.setText(_translate("MainWindow", "Chọn ảnh"))
        self.groupBox.setTitle(_translate("MainWindow", "Kết quả"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Đầu vào"))
        self.fromVID.setText(_translate("MainWindow", "Từ live video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
