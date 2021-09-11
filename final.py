#Import cac package can thiet
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

#Model
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
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

#Emotion dic
emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "Neutral  ", 5: "    Sad    ", 6: "Surprised"}
emoji_dist={0:"emojis/angry.png",1:"emojis/disgusted.png",2:"emojis/fearful.png",3:"emojis/happy.png",4:"emojis/neutral.png",5:"emojis/sad.png",6:"emojis/surpriced.png"}

#GUI
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1103, 817)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(589, 169, 451, 591))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.groupBox)
        self.graphicsView_2.setGeometry(QtCore.QRect(30, 180, 400, 400))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.outLabel = QtWidgets.QLabel(self.groupBox)
        self.outLabel.setGeometry(QtCore.QRect(30, 180, 400, 400))
        self.outLabel.setText("")
        self.outLabel.setObjectName("outLabel")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(190, 70, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.eLabel = QtWidgets.QLabel(self.groupBox)
        self.eLabel.setGeometry(QtCore.QRect(170, 100, 131, 30))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.eLabel.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.eLabel.setFont(font)
        self.eLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.eLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.eLabel.setText("")
        self.eLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.eLabel.setObjectName("eLabel")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.groupBox)
        self.graphicsView_3.setGeometry(QtCore.QRect(170, 100, 131, 30))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_3.raise_()
        self.graphicsView_2.raise_()
        self.outLabel.raise_()
        self.label_4.raise_()
        self.eLabel.raise_()
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 170, 471, 591))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_2)
        self.graphicsView.setGeometry(QtCore.QRect(30, 180, 400, 400))
        self.graphicsView.setObjectName("graphicsView")
        self.inLabel = QtWidgets.QLabel(self.groupBox_2)
        self.inLabel.setGeometry(QtCore.QRect(30, 180, 400, 400))
        self.inLabel.setText("")
        self.inLabel.setObjectName("inLabel")
        self.fromVid = QtWidgets.QPushButton(self.groupBox_2)
        self.fromVid.setGeometry(QtCore.QRect(190, 50, 93, 28))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.fromVid.setFont(font)
        self.fromVid.setObjectName("fromVid")
        self.fromImg = QtWidgets.QPushButton(self.groupBox_2)
        self.fromImg.setGeometry(QtCore.QRect(190, 90, 93, 28))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.fromImg.setFont(font)
        self.fromImg.setObjectName("fromImg")
        self.groupBox.raise_()
        self.groupBox_2.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
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
        self.fromVid.clicked.connect(self.fVid)
        self.fromImg.clicked.connect(self.fIMG)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "D-emoji 1.0"))
        self.label.setText(_translate("MainWindow", "ỨNG DỤNG NHẬN DIỆN CẢM XÚC VÀ TỰ ĐỘNG CHỌN EMOJI TƯƠNG ỨNG"))
        self.label_2.setText(_translate("MainWindow", "ẢNH ĐẦU VÀO"))
        self.label_3.setText(_translate("MainWindow", "BIỂU CẢM TƯƠNG ỨNG"))
        self.groupBox.setTitle(_translate("MainWindow", "Kết quả"))
        self.label_4.setText(_translate("MainWindow", "CẢM XÚC"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Đầu vào"))
        self.fromVid.setText(_translate("MainWindow", "Từ live video"))
        self.fromImg.setText(_translate("MainWindow", "Từ ảnh"))

    #Nhan dien cam xuc tu live video
    def fVid(self):
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
            self.eLabel.setText(emotion_dict[maxindex])
            #emoji_dist[5]
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def fIMG(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        pixmap = QtGui.QPixmap(imagePath)
        pixmap = pixmap.scaledToWidth(400)
        self.inLabel.setPixmap(pixmap)
        self.inLabel.resize(pixmap.size())
        self.inLabel.adjustSize()
        frame = cv2.imread(imagePath)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        num_faces = bounding_box.detectMultiScale(gray_frame)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        self.outLabel.setPixmap(QtGui.QPixmap(emoji_dist[maxindex]))
        self.eLabel.setText(emotion_dict[maxindex])

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


#Nothing changes
