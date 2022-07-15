import random
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QFileDialog, QFrame, \
    QListView
from PyQt5.QtGui import QPen, QPainter, QPixmap,QIcon
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets
from tensorflow import keras
import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class MainWindows(QMainWindow):
    def __init__(self):
        super(MainWindows, self).__init__()
        self.InitUI()

    def InitUI(self):
        self.model = keras.models.load_model('./weights.32-0.9965.hdf5')
        self.num = 0
        self.resize(500, 400)
        self.setWindowTitle('手写数字识别')

        self.button_regonize = QPushButton(self)
        self.button_regonize.setObjectName("button_regonize")
        self.button_regonize.setGeometry(73, 270, 70, 50)
        self.button_regonize.setText('识别')
        self.button_regonize.setStyleSheet("#button_regonize{\n"
                                           "    background-color:white;\n"
                                           "    color:black;\n"
                                           "    border:1px solid black;\n"
                                           "    border-radius:5px;\n"
                                           "}\n"
                                           "#button_regonize:hover{\n"
                                           "    background-color:black;\n"
                                           "    color:white\n"
                                           "}")
        # vbox.addWidget(self.button_regonize)
        # vbox.addStretch()

        self.button_clear = QPushButton(self)
        self.button_clear.setObjectName('button_clear')
        self.button_clear.setGeometry(176, 270, 70, 50)
        self.button_clear.setText('清空')
        self.button_clear.setStyleSheet("#button_clear{\n"
                                           "    background-color:white;\n"
                                           "    color:black;\n"
                                           "    border:1px solid black;\n"
                                           "    border-radius:5px;\n"
                                           "}\n"
                                           "#button_clear:hover{\n"
                                           "    background-color:black;\n"
                                           "    color:white\n"
                                           "}")
        # vbox.addWidget(self.button_clear)
        # vbox.addStretch()    # 什么意思

        self.result_text = QLabel(self)
        self.result_text.setText('结果为:')
        self.result_text.setGeometry(300, 160, 50, 50)

        self.result_lb = QLabel(self)
        self.result_lb.setText('5')
        self.result_lb.setGeometry(300, 200, 91, 81)
        self.result_lb.setAlignment(Qt.AlignCenter)
        self.result_lb.setStyleSheet('font: 87 48pt "Arial Black";border:1px solid rgb(0,0,0)')
        # vbox.addWidget(self.result_lb)
        # vbox.addStretch()


        # self.label_border = QLabel(self)
        # self.label_border.setGeometry(50, 50, 200, 200)
        # self.label_border.setStyleSheet('border:2px solid black')

        self.lb = MyPaint_label(self)
        self.lb.setGeometry(50, 50, 200, 200)
        # self.lb.setStyleSheet('border:2px solid black')

        self.cb = QtWidgets.QComboBox(self)
        self.cb.addItem('手写数字识别')
        self.cb.addItem('样本识别')
        self.cb.setGeometry(300, 120, 150, 50)
        self.cb.setStyleSheet('QAbstractItemView #item{height:120px;}')
        self.cb.setView(QListView())
        # self.cb.setStyleSheet('border:1px black solid;'
        #                       'border-radius:5px')

        self.file_button = QPushButton(self)
        self.file_button.setObjectName('file_button')
        self.file_button.setText("选择图片")
        self.file_button.setStyleSheet("#file_button{\n"
                                           "    background-color:white;\n"
                                           "    color:black;\n"
                                           "    border:1px solid black;\n"
                                           "    border-radius:5px;\n"
                                           "}\n"
                                           "#file_button:hover{\n"
                                           "    background-color:black;\n"
                                           "    color:white\n"
                                           "}")
        self.file_button.setGeometry(300, 60, 80, 50)
        # self.file_button.setGeometry(QtCore.QRect(30, 20, 30, 30))

        self.button_clear.clicked.connect(lambda: self.clear_list())
        self.button_regonize.clicked.connect(lambda: self.regonize())
        self.file_button.clicked.connect(lambda:self.openFile())
        self.cb.activated.connect(lambda:self.current_comb())
        # self.cb.currentIndexChanged[str].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目内容
        # self.cb.highlighted[int].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目索引
    def random_pic(self):
        filepath = './pic\\mnist_pic'
        fname = os.listdir('./pic\\mnist_pic')
        num = len(fname)
        rand = random.randint(1, num)
        fname = os.path.join(filepath, fname[rand])
        pic = QPixmap(fname).scaled(self.lb.pixmap.width(), self.lb.pixmap.height())
        self.lb.pixmap = pic
        self.lb.update()
    def current_comb(self):
        if self.cb.currentText() == '手写数字识别':
            self.button_clear.setText('清空')
            self.lb.pixmap.fill(Qt.black)
            self.lb.update()
        elif self.cb.currentText() == '样本识别':
            self.button_clear.setText('刷新')
            self.random_pic()

    # def print_value(self, i):
    #     # print('暂时弃用')
    #     if i == '样本识别':
    #         self.clear_list()
    #         filepath = 'D:\python_game\game_hyol\MyMinst\pic\mnist_pic'
    #         fname = os.listdir('D:\python_game\game_hyol\MyMinst\pic\mnist_pic')
    #         num = len(fname)
    #         rand = random.randint(1,num)
    #         fname = os.path.join(filepath,fname[rand])
    #         pic = QPixmap(fname).scaled(self.lb.pixmap.width(), self.lb.pixmap.height())
    #         self.lb.pixmap = pic
    #         self.lb.update()
    #         print('刷新了')
    #     elif i == '手写数字识别':
    #         self.lb.pixmap.fill(Qt.black)
    #         self.lb.update()
    #         self.button_clear.setText('清空')

    def clear_list(self):
        # self.lb.pos_xy.clear()
        if self.cb.currentText() == '手写数字识别':
            self.lb.pixmap.fill(Qt.black)
            self.lb.update()
        elif self.cb.currentText() == '样本识别':
            self.random_pic()
            print('样本识别 刷新')
            # def


    def regonize(self):
        print('识别了')
        # img = self.lb.pixmap().toImage()
        self.num += 1
        savePath = f'./pic\\save_recgonize\\1.png'
        self.lb.pixmap.save(savePath)   # 保存pixmap图片
        # print('保存了')
        # savePath = "D:\\python_game\\game_hyol\\MyMinst\\web_mnist\\pic\\test.png"
        # image = self.__paintBoard.GetContentAsQImage()
        # image.save(savePath)
        # print(savePath)
        # 加载图像
        # img_PIL = Image.open(f'D:\\python_game\\game_hyol\\MyMinst\\pic\\{self.num}.png')
        # print(img_PIL)
        img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        img = img.convert('L')
        x = keras.preprocessing.image.img_to_array(img)
        # x = abs(255 - x)
        # x = x.reshape(28,28)
        # x = np.expand_dims(x, axis=0)
        # print(x.shape)
        x = np.reshape(x,[1,28,28,1])
        x = x / 255.0
        # model = keras.models.load_model('D:\\python_game\\game_hyol\\MyMinst\\mnist_file\\weights.32-0.9965.hdf5')
        prediction = self.model.predict(x)
        output = np.argmax(prediction, axis=1)
        # np.set_printoptions(precision=4,suppress=None)
        np.set_printoptions(suppress=None)
        print(prediction[0][output])
        self.result_lb.setText(str(output[0]))
        print(output[0])

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "选择图片文件", ".")
        if fname[0]:
            pic = QPixmap(fname[0]).scaled(self.lb.pixmap.width(), self.lb.pixmap.height())
            self.lb.pixmap = pic

class MyPaint_label(QLabel):
    def __init__(self,parent):
        super(MyPaint_label, self).__init__(parent)
        # self.resize(200,200)
        self.pixmap = QPixmap(200, 200)
        self.pixmap.fill(Qt.black)
        # self.setStyleSheet("border:2px solid black")


        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.painter = QPainter()
        self.setMouseTracking(False)

    def paintEvent(self, event):
        self.painter.begin(self)
        self.painter.drawPixmap(0,0,self.pixmap)
        self.painter.end()

    def mousePressEvent(self, event):
        # if event.button() == Qt.LeftButton:
        self.lastPoint = event.pos()
        self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):  # 重写鼠标移动事件
        # if event.buttons() == Qt.LeftButton:
        self.endPoint = event.pos()

        self.painter.begin(self.pixmap)   # 16
        pen = QPen(Qt.white, 16, Qt.SolidLine)
        self.painter.setPen(pen)
        self.painter.drawLine(self.lastPoint,self.endPoint)
        self.painter.end()

        self.lastPoint = self.endPoint
        self.update()  # 更新绘图事件,每次执行update都会触发一次paintEvent(self, event)函数


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./pic\\icon\\five_icon.png'))
    mainWindow = MainWindows()
    mainWindow.show()
    sys.exit(app.exec_())
