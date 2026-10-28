import sys
from pyqtgraph.Qt import QtWidgets
from rcppg.gui import ReciprocalPlayground

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    main = ReciprocalPlayground()
    main.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
