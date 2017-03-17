import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt
import tomograph


def main():
    app = QtGui.QApplication([])

    # Create window with ImageView widget
    # win = QtGui.QMainWindow()
    win = QtGui.QWidget()
    layout = QtGui.QHBoxLayout()
    win.setLayout(layout)

    win.resize(800, 400)
    imv = pg.ImageView()
    layout.addWidget(imv)

    imv2 = pg.ImageView()
    layout.addWidget(imv2)

    win.show()
    win.setWindowTitle('Computer Tomography')

    fname = 'img/07.png'
    img = plt.imread(fname)
    if len(img.shape) == 3: # RGB
        img = img[:,:,0]
    assert len(img.shape) == 2
    imv.setImage(img, autoLevels=False, levels=(0.01, 1))

    width=0.9
    sinogram = tomograph.radon_transform(img, 200, 40, width)
    result_img = tomograph.reverse_radon(sinogram, width, 400)
    imv2.setImage(result_img, autoLevels=True)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    main()
