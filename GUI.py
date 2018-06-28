from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QDialog, QApplication



class Ui_Form(object):

    def __init__(self):
        super().__init__()
        self.method = 0
        self.product = 0



    def setupUi(self, Form):

        Form.setObjectName("Form")
        Form.resize(924, 409)

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(310, 40, 321, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(20, 115, 301, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.radioButton = QtWidgets.QRadioButton(Form)
        self.radioButton.setGeometry(QtCore.QRect(30, 150, 191, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton.setChecked(True)
        self.radioButton.toggled.connect(lambda: self.on_radio(self.radioButton))

        self.radioButton_2 = QtWidgets.QRadioButton(Form)
        self.radioButton_2.setGeometry(QtCore.QRect(30, 180, 251, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_2.toggled.connect(lambda: self.on_radio(self.radioButton_2))

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(580, 115, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setGeometry(QtCore.QRect(580, 160, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(["Test #1", "Test #2", "Test #3", "Test #4", "Test #5", "Test #6", "Test #7", "Test #8", "Test #9", "Test #10"])

        self.comboBox.currentIndexChanged.connect(self.set_product)
        # self.comboBox.addItem("")
        # self.comboBox.addItem("")

        self.startButton = QtWidgets.QPushButton(Form)
        self.startButton.setGeometry(QtCore.QRect(390, 320, 141, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.startButton.setFont(font)
        self.startButton.setObjectName("pushButton")
        self.startButton.clicked.connect(lambda: self.on_start(1))

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Amazon Food Reviews"))
        self.label_2.setText(_translate("Form", "Select type of analysis:"))
        self.radioButton.setText(_translate("Form", "Sentiment Analysis"))
        self.radioButton_2.setText(_translate("Form", "Aspect Based Sentimen Analysis"))
        self.label_3.setText(_translate("Form", "Select Product:"))
        # self.comboBox.setItemText(0, _translate("Form", "Prodotto 1"))
        # self.comboBox.setItemText(1, _translate("Form", "Prodotto 2"))
        self.startButton.setText(_translate("Form", "START"))


    # prendo il valore dei radio button
    def on_radio(self, b):

        # imposto il tipo di analisi

        if b.text() == "Sentiment Analysis":
            if b.isChecked() == True:
                self.method = 1

        elif b.text() == "Aspect Based Sentimen Analysis":
            if b.isChecked() == True:
                self.method = 2



    def set_product(self, i):
        # i è l'indice del comboBox (parte da 0)
        self.product = i+1








    # Metodo associato al tasto START
    def on_start(self, product):
        print("Bottone START")

        # se non ho selezionato il metodo metto di default 1
        if self.method == 0:
            self.method = 1

        # se non ho selezionato un prodotto metto di default il primo
        if self.product == 0:
            self.product = 1




        print("METHOD: {}".format(self.method))
        print("PRODUCT: {}".format(self.product))



















class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.show()

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
