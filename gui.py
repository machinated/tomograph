import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt
import tomograph

class Window(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Computer Tomography')

        #widgets
        self.image_view = pg.ImageView(self)
        self.result_view = pg.ImageView(self)
        self.progress_bar = QtGui.QProgressBar(self)
        self.button_calculate = QtGui.QPushButton('Calculate', self)

        param_tree = (
            {'name': 'img_size', 'type': 'int', 'value': 400},
            {'name': 'n_angles', 'type': 'int', 'value': 50},
            {'name': 'n_detectors', 'type': 'int', 'value': 10},
            {'name': 'width', 'type': 'float', 'value': 0.9},
            {'name': 'play_rate', 'type': 'int', 'value': 2}
            )
        self.parameters = pg.parametertree.Parameter.create(
            name = 'Settings', type='group', children=param_tree)
        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setParameters(self.parameters, showTop=False)
        self.param_tree.setSizePolicy(QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Preferred))
            #       horizontal                  vertical

        # layout
        self.layout = QtGui.QVBoxLayout(self)
        self.layout_images = QtGui.QHBoxLayout()

        self.layout.addWidget(self.param_tree)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.button_calculate)

        self.layout_images.addWidget(self.image_view)
        self.layout_images.addWidget(self.result_view)
        self.layout.addLayout(self.layout_images)

        # signals
        self.button_calculate.clicked.connect(self.calculate)

        self.load_image()

    def load_image(self):
        fname = 'img/07.png'
        self.img = plt.imread(fname)
        if len(self.img.shape) == 3: # RGB
            self.img = self.img[:,:,0]
        assert len(self.img.shape) == 2
        self.image_view.setImage(self.img, autoLevels=False, levels=(0.01, 1))

    def calculate(self):
        self.button_calculate.setEnabled(False)
        width = self.parameters.child('width').value()
        n_angles = self.parameters.child('n_angles').value()
        n_detectors = self.parameters.child('n_detectors').value()
        sinogram = tomograph.radon_transform(self.img, n_angles, n_detectors, width)

        img_size = self.parameters.child('img_size').value()
        result_img = np.zeros(shape=(n_angles, img_size, img_size), dtype=np.int64)
        #            np.empty
        # result_img[0].fill(0)

        for step in tomograph.reverse_radon(result_img, sinogram, width, img_size):
            self.progress_bar.setValue(int(100*(step+1)/n_angles))
        view_array = np.array(result_img, dtype=np.float64)
        self.result_view.setImage(view_array, autoLevels=False, levels=(0, 10**3))
        play_rate = self.parameters.child('play_rate').value()
        self.result_view.play(play_rate)
        # self.result_view.autoLevels()
        self.button_calculate.setEnabled(True)


def main():
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.resize(800, 500)
    window.show()
    sys.exit(app.exec_())
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    main()
