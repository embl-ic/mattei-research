# =============================================================================
# COPYRIGHT, ATTRIBUTION, AND LICENSE NOTICE
# =============================================================================
#
# Copyright (c) 2026 Linhua Tai,
# Mahamid Lab, Molecular Systems Biology Unit,
# European Molecular Biology Laboratory (EMBL), Heidelberg, Germany.
#
# SOFTWARE DEVELOPMENT:
# This script was developed by Linhua Tai in the Mahamid Lab,
# Molecular Systems Biology Unit, at the European Molecular Biology
# Laboratory (EMBL), Heidelberg, Germany, with the assistance of
# Artificial Intelligence (AI) tools for code development,
# optimization, and documentation.
#
# OPEN-SOURCE AVAILABILITY:
# This software is released as open-source software under the MIT License
# to support scientific research, reproducibility, and collaboration.
#
# Users are permitted to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of this software, subject to the terms
# of the MIT License provided below.
#
# ATTRIBUTION AND CITATION:
# When using, modifying, adapting, or redistributing this software,
# users must retain the original copyright and license notices as
# required by the MIT License.
#
# When this software contributes to scientific publications,
# presentations, or other research outputs, users are kindly requested
# to acknowledge the original developer as:
#
# "Linhua Tai, Mahamid Lab, Molecular Systems Biology Unit,
#  European Molecular Biology Laboratory (EMBL), Heidelberg."
#
# Scientific citation and acknowledgment are strongly encouraged
# to recognize the contributions of the original developer.
#
# =============================================================================
# MIT LICENSE
# =============================================================================
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# =============================================================================

from PyQt5 import QtWidgets, QtGui, QtCore
import mrcfile
import pandas as pd
import os
from glob import glob
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO

class InitialGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.tiltstack_dir = None
        self.save_dir = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Initial Setup")
        self.setGeometry(100, 100, 400, 200)

        layout = QtWidgets.QVBoxLayout()

        tiltstack_btn = QtWidgets.QPushButton('Select tiltstack directory', self)
        tiltstack_btn.clicked.connect(self.browse_tiltstack)
        self.tiltstack_label = QtWidgets.QLabel("No directory selected")
        layout.addWidget(tiltstack_btn)
        layout.addWidget(self.tiltstack_label)

        save_dir_btn = QtWidgets.QPushButton('Select CSV save directory', self)
        save_dir_btn.clicked.connect(self.browse_save_dir)
        self.save_label = QtWidgets.QLabel("No directory selected")
        layout.addWidget(save_dir_btn)
        layout.addWidget(self.save_label)

        start_btn = QtWidgets.QPushButton("Let's Clicking!", self)
        start_btn.clicked.connect(self.next_step)
        layout.addWidget(start_btn)

        self.setLayout(layout)

    def browse_tiltstack(self):
        self.tiltstack_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory')
        if self.tiltstack_dir:
            self.tiltstack_label.setText(self.tiltstack_dir)

    def browse_save_dir(self):
        self.save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory')
        if self.save_dir:
            self.save_label.setText(self.save_dir)

    def next_step(self):
        if self.tiltstack_dir and self.save_dir:
            self.hide()
            self.selector_gui = MRCSelectorGUI(self.tiltstack_dir, self.save_dir)
            self.selector_gui.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Both directories must be selected")

class MRCSelectorGUI(QtWidgets.QWidget):
    def __init__(self, tiltstack_dir, save_dir):
        super().__init__()
        self.tiltstack_dir = tiltstack_dir
        self.save_dir = save_dir
        self.clicked_files = set()  # Track clicked files
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Select .st Files")
        self.setGeometry(100, 100, 600, 400)

        layout = QtWidgets.QVBoxLayout()

        self.st_list = QtWidgets.QListWidget(self)
        self.st_list.itemDoubleClicked.connect(self.on_st_select)
        layout.addWidget(self.st_list)

        self.setLayout(layout)
        self.list_st_files()

    def list_st_files(self):
        st_files = glob(os.path.join(self.tiltstack_dir, '**/*.st'), recursive=True)
        st_files.sort()  # Sort files by filename
        for st_file in st_files:
            item = QtWidgets.QListWidgetItem(os.path.basename(st_file))
            if st_file in self.clicked_files:
                item.setBackground(QtGui.QColor(200, 200, 255))  # Light blue background for clicked files
            self.st_list.addItem(item)

    def on_st_select(self, item):
        selected_file = item.text()
        full_path = glob(os.path.join(self.tiltstack_dir, '**', selected_file), recursive=True)[0]
        self.clicked_files.add(full_path)
        item.setBackground(QtGui.QColor(200, 200, 255))  # Highlight clicked item
        self.frame_selector_gui = FrameSelectorGUI(full_path, self.save_dir)
        self.frame_selector_gui.show()

class FrameSelectorGUI(QtWidgets.QWidget):
    def __init__(self, st_file_path, save_dir):
        super().__init__()
        self.st_file_path = st_file_path
        self.save_dir = save_dir
        self.selected_frames = []  # Store selected frame indices
        self.data = None
        self.frame_count = 0
        self.current_frame_index = 0
        self.global_min = None
        self.global_max = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Select Frames")
        layout = QtWidgets.QVBoxLayout()

        # QLabel for displaying the current frame
        self.frame_label = QtWidgets.QLabel(self)
        self.frame_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.frame_label)

        # QScrollBar for selecting frames
        self.scroll_bar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, self)
        self.scroll_bar.valueChanged.connect(self.on_scroll)
        layout.addWidget(self.scroll_bar)

        # Set the main layout
        self.setLayout(layout)

        # Load the .st file
        self.load_st_file()

    def load_st_file(self):
        with mrcfile.open(self.st_file_path) as mrc:
            self.data = mrc.data
            self.frame_count = len(self.data)
            # Calculate global min and max across all frames
            self.global_min = np.min(self.data)
            self.global_max = np.max(self.data)
            self.scroll_bar.setMaximum(self.frame_count - 1)
            self.display_frame(0)  # Display the first frame

    def convert_to_8bit_and_enhance_contrast(self, frame):
        # Normalize the frame using global min and max
        if self.global_max > self.global_min:
            frame_8bit = 255 * (frame - self.global_min) / (self.global_max - self.global_min)
        else:
            frame_8bit = np.zeros_like(frame)
        frame_8bit = frame_8bit.astype(np.uint8)

        img = Image.fromarray(frame_8bit, mode='L')

        # Enhance brightness slightly before contrast
        brightness_enhancer = ImageEnhance.Brightness(img)
        img_bright = brightness_enhancer.enhance(1.2)  # Slightly increase brightness

        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(img_bright)
        img_enhanced = contrast_enhancer.enhance(2.0)  # Increase contrast by a factor of 2
        return np.array(img_enhanced)

    def display_frame(self, index):
        frame = self.data[index]
        frame = self.convert_to_8bit_and_enhance_contrast(frame)

        img = Image.fromarray(frame, mode='L')
        bytes_io = BytesIO()
        img.save(bytes_io, format='PNG')
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(bytes_io.getvalue())

        self.frame_label.setPixmap(pixmap)
        self.frame_label.adjustSize()  # Adjust QLabel size to fit the image

        # Highlight selected frame
        if index in self.selected_frames:
            self.frame_label.setStyleSheet("background-color: yellow;")
        else:
            self.frame_label.setStyleSheet("background-color: none;")

        # Adjust window size to fit the image plus extra space
        self.setFixedSize(pixmap.width() + 200, pixmap.height() + 200)

    def on_scroll(self, value):
        self.current_frame_index = value
        self.display_frame(value)

    def wheelEvent(self, event):
        # Control scroll bar with mouse wheel
        delta = event.angleDelta().y()
        steps = delta // 120  # Each step corresponds to one notch of the wheel
        new_value = self.scroll_bar.value() - steps
        new_value = max(0, min(self.scroll_bar.maximum(), new_value))
        self.scroll_bar.setValue(new_value)

    def mousePressEvent(self, event):
        # Toggle frame selection on mouse click
        if self.current_frame_index in self.selected_frames:
            self.selected_frames.remove(self.current_frame_index)
        else:
            self.selected_frames.append(self.current_frame_index)

        print(f"Selected frames: {self.selected_frames}")
        self.display_frame(self.current_frame_index)  # Update frame display to reflect selection state

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up:
            if self.current_frame_index > 0:
                self.current_frame_index -= 1
                self.scroll_bar.setValue(self.current_frame_index)
        elif event.key() == QtCore.Qt.Key_Down:
            if self.current_frame_index < self.frame_count - 1:
                self.current_frame_index += 1
                self.scroll_bar.setValue(self.current_frame_index)
        elif event.key() == QtCore.Qt.Key_Space:
            # Toggle frame selection with Space key
            if self.current_frame_index in self.selected_frames:
                self.selected_frames.remove(self.current_frame_index)
            else:
                self.selected_frames.append(self.current_frame_index)

            print(f"Selected frames: {self.selected_frames}")
            self.display_frame(self.current_frame_index)  # Update frame display to reflect selection state

    def closeEvent(self, event):
        self.save_selected_frames()

    def save_selected_frames(self):
        if self.st_file_path and self.selected_frames:
            csv_file_name = os.path.splitext(os.path.basename(self.st_file_path))[0] + '.csv'
            csv_file_path = os.path.join(self.save_dir, csv_file_name)
            df = pd.DataFrame(self.selected_frames, columns=['Frame Index'])
            df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])  
    main_gui = InitialGUI()
    main_gui.show()  
    app.exec_()  