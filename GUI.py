from PyQt5 import QtGui, QtCore
from bonesi_aspect_based import aspect2
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication





class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Amazon food rewievs'
        self.left = 700
        self.top = 100
        self.width = 940
        self.height = 450
        self.initUI()
        self.days = 0
        self.method = 0
        self.product = 0


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        myFont = QtGui.QFont()
        myFont.setBold(True)



        self.title_label = QLabel("Amazon Food Reviews", self)
        self.title_label.move(300, 10)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)


        self.choose_method_label = QLabel('Select type of analysis:', self)
        self.choose_method_label.move(20, 115)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.choose_method_label.setFont(font)



        self.radioButton = QRadioButton(self)
        self.radioButton.move(30,150)
        self.radioButton.setText("Sentiment Analysis")
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radioButton.setFont(font)
        self.radioButton.setChecked(True)
        self.radioButton.toggled.connect(lambda: self.on_radio(self.radioButton))

        self.radioButton_2 = QRadioButton(self)
        self.radioButton_2.move(30,180)
        self.radioButton_2.setText("Aspect Based Sentiment Analysis")
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radioButton_2.setFont(font)
        self.radioButton_2.toggled.connect(lambda: self.on_radio(self.radioButton_2))

        self.choose_product_label = QLabel('Select product:', self)
        self.choose_product_label.move(580, 115)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.choose_product_label.setFont(font)

        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(580, 160, 150, 25) # x, y, width, height
        font = QtGui.QFont()
        font.setPointSize(9)
        self.comboBox.setFont(font)
        self.comboBox.addItems(
            ["Test #1", "Test #2", "Test #3", "Test #4", "Test #5", "Test #6", "Test #7", "Test #8", "Test #9",
             "Test #10"])
        self.comboBox.currentIndexChanged.connect(self.set_product)


        self.startButton = QPushButton(self)
        self.startButton.setGeometry(QtCore.QRect(390, 320, 141, 61))
        self.startButton.setText("START")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.startButton.setFont(font)
        self.startButton.clicked.connect(lambda: self.on_start())



        self.show()



    # prendo il valore dei radio button
    def on_radio(self, b):
        # imposto il tipo di analisi
        if b.text() == "Sentiment Analysis":
            if b.isChecked() == True:
                self.method = 1

        elif b.text() == "Aspect Based Sentiment Analysis":
            if b.isChecked() == True:
                self.method = 2

    def set_product(self, i):
        # i è l'indice del comboBox (parte da 0)
        self.product = i



    def on_start(self):
        print("Bottone START")

        # se non ho selezionato il metodo metto di default 1
        if self.method == 0:
            self.method = 1

        print("METHOD: {}".format(self.method))
        print("PRODUCT: {}".format(self.product))

        if self.method == 1:
            print("metodo sentiment")
        else:
            aspect2(self.product)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
