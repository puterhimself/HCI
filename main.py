import sys
import engine
from PyQt5 import QtWidgets,QtGui
from keras.models import load_model
from interface import Ui_MainWindow

class UILoader(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(UILoader, self).__init__()
        self.setupUi(self)
        self.button_cam.clicked.connect(self.cam)
        self.button_train_more.clicked.connect(self.TrainMore)
        self.button_train_new.clicked.connect(self.TrainNew)
        self.button_new.clicked.connect(self.new)

        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        self.label = QtWidgets.QLabel(self.frame_open_cv)
        self.label.setFont(font)
        # self.gridLayout_2.addWidget(self.label)
        self.label.setText("Info:")

    def TrainNew(self):
        model = engine.loadNN()
        self.label.setText(engine.summary)
        engine.TrainModel(model)


    def TrainMore(self):
        model = load_model(engine.modelFile)
        engine.TrainModel(model)

    def new(self):
        name = self.textbox.text().strip().lower()
        if name != '':
            engine.GestureName = name
            engine.NewGesture()
        else:
            self.label.setText('Invalid gesture!')

    def cam(self):
        engine.main()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = UILoader()
    form.show()
    sys.exit(app.exec_())
