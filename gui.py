import datetime
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from dicom.dataset import Dataset, FileDataset
from pyqtgraph.Qt import QtGui

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
        self.button_load = QtGui.QPushButton('Load image', self)
        self.button_save = QtGui.QPushButton('Save current frame', self)
        self.button_save_dicom = QtGui.QPushButton('Save DICOM', self)

        param_tree = (
            {'name': 'img_size', 'type': 'int', 'value': 400},
            {'name': 'n_angles', 'type': 'int', 'value': 50},
            {'name': 'n_detectors', 'type': 'int', 'value': 10},
            {'name': 'width', 'type': 'float', 'value': 0.9},
            {'name': 'mask_size', 'type': 'float', 'value': 5},
            {'name': 'play_rate', 'type': 'int', 'value': 2},
            {'name': 'Patient data', 'type': 'group', 'children': (
                {'name': 'PESEL', 'type': 'str', 'value': ''},
                {'name': 'Name', 'type': 'str', 'value':''},
                {'name': 'Sex', 'type': 'list', 'values': ('male', 'female')},
                {'name': 'Comments', 'type': 'text'} )}
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

        self.layout.addWidget(self.button_load)
        self.layout.addWidget(self.param_tree)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.button_calculate)
        self.layout.addWidget(self.button_save)
        self.layout.addWidget(self.button_save_dicom)

        self.layout_images.addWidget(self.image_view)
        self.layout_images.addWidget(self.result_view)
        self.layout.addLayout(self.layout_images)

        # signals
        self.button_calculate.clicked.connect(self.calculate)
        self.button_load.clicked.connect(self.load_image)
        self.button_save.clicked.connect(self.save_current_frame)
        self.button_save_dicom.clicked.connect(self.save_dicom)

        # self.load_image()

    def load_image(self):
        # fname = 'img/02.png'
        fname = QtGui.QFileDialog.getOpenFileName(filter="Images (*.png *.xpm *.jpg)")[0]
        if fname == '':
            return
        print(fname)
        self.img = plt.imread(fname)
        if len(self.img.shape) == 3: # RGB
            self.img = self.img[:,:,0]
        assert len(self.img.shape) == 2
        self.image_view.setImage(self.img, autoLevels=False, levels=(0.01, 1))

    def save_current_frame(self):
        frame = self.result_view.currentIndex
        fileName = QtGui.QFileDialog.getSaveFileName()
        if fileName == '':
            return
        img = self.result_view.getProcessedImage()
        self.result_view.imageItem.setImage(img[frame], autoLevels=False)
        self.result_view.imageItem.save(fileName)
        self.result_view.updateImage()

    def save_dicom(self):
        filename = QtGui.QFileDialog.getSaveFileName(filter="DICOM (*.dcm)")[0]
        if filename == '':
            return
        img = self.result_view.getProcessedImage()[-1]

        tree = self.parameters.child('Patient data').getValues()
        info = OrderedDict()
        for key in tree:
            info[key] = tree[key][0]
        print(info)
        write_dicom(img, filename, info)

    def calculate(self):
        self.button_calculate.setEnabled(False)
        width = self.parameters.child('width').value()
        n_angles = self.parameters.child('n_angles').value()
        n_detectors = self.parameters.child('n_detectors').value()

        sinogram = np.zeros(shape=(n_angles, n_detectors), dtype=np.int64)
        for step in tomograph.radon_transform(self.img, sinogram, n_angles, n_detectors, width):
            self.progress_bar.setValue(int(100*(step+1)/n_angles))

        mask_size = self.parameters.child('mask_size').value()
        if mask_size > 0:
            mask = tomograph.get_mask(mask_size)
            print("Mask: {}".format(mask))
            sinogram = tomograph.filter_sinogram(sinogram, mask)

        img_size = self.parameters.child('img_size').value()
        result_img = np.zeros(shape=(n_angles, img_size, img_size), dtype=np.float64)
        #            np.empty
        # result_img[0].fill(0)

        for step in tomograph.reverse_radon(result_img, sinogram, width, img_size):
            self.progress_bar.setValue(int(100*(step+1)/n_angles))
        # view_array = np.array(result_img, dtype=np.float64)
        test_slice = result_img[-1, img_size//2, :]
        min_value = np.percentile(test_slice, 5.0)
        max_value = np.percentile(test_slice, 95.0)
        print("Levels: ({}, {})".format(min_value, max_value))
        self.result_view.setImage(result_img, autoLevels=False,
                                  levels=(min_value, max_value))
        play_rate = self.parameters.child('play_rate').value()
        self.result_view.play(play_rate)
        # self.result_view.autoLevels()
        self.button_calculate.setEnabled(True)


def write_dicom(pixels, filename, info=None):
    if (info==None):
        info = {'pesel': '12345678900', 'name': 'Jan Kowalski', 'sex': 'Mezczyzna', 'comments': 'Tutaj komentarz'}
    pixels -= np.min(pixels)
    pixels *= 65536 / np.max(pixels)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=("\0" * 128).encode())
    ds.Modality = 'WSD'

    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = pixels.shape[0]
    ds.Rows = pixels.shape[1]
    if pixels.dtype != np.uint16:
        pixels = pixels.astype(np.uint16)
    ds.PixelData = pixels.tostring()

    ds.PatientID = info['PESEL']
    ds.PatientsName = info['Name']
    ds.PatientsSex = info['Sex']
    ds.ImageComments = info['Comments']
    ds.StudyDate = str(datetime.datetime.now()).replace('-', '')
    time = str(datetime.datetime.now().time()).replace(':', '').split('.')
    ds.StudyTime = time[0] + '.' + time[1][:3]
    ds.save_as(filename)


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
