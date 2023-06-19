import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMainWindow, QAction


class window(QWidget):

    def __init__(self, win):
        super().__init__()
        self.win=win


    
    def build(self):
        
        self.win.label=QLabel("Tuto",self)
        self.win.bouton=QPushButton('button',self)
        layout=QVBoxLayout()
        layout.addWidget(self.win.bouton)
        self.setLayout(layout)
        self.resize(600,300)
        
        self.win.setWindowTitle("tuto")


if __name__=="__main__":
    app= QApplication(sys.argv)
    root= QWidget()
    mywin= window(root)
    mywin.build()
    mywin.show()

    sys.exit(app.exec_())