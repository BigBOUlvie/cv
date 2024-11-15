import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from collections import deque

class VideoProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_frame)
        self.model = YOLO("yolov8n.pt")
        self.frame_number = 0
        self.prev_positions = {}
        self.vehicle_window = deque(maxlen=25)
        self.df = pd.DataFrame(columns=['Frame Number', 'Flow (vehicles/s)', 'Density (vehicles/m)'])

    def initUI(self):
        self.setWindowTitle('Video Processor')

        # Layout
        layout = QVBoxLayout()

        # Video display label
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        # Buttons
        self.open_button = QPushButton('选择视频', self)
        self.open_button.clicked.connect(self.open_video)
        layout.addWidget(self.open_button)

        self.start_button = QPushButton('开始检测', self)
        self.start_button.clicked.connect(self.start_video)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def open_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if file_name:
            self.video_capture = cv2.VideoCapture(file_name)

    def start_video(self):
        if self.video_capture and self.video_capture.isOpened():
            self.timer.start(30)  # 30 ms for approximately 33.3 fps

    def show_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.save_data_and_close()
            return

        results = self.model.track(frame, persist=True)
        a = results[0].plot()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu()

        current_frame_vehicle_count = len(track_ids)
        current_ids = set(track_ids)
        self.vehicle_window.append(current_ids)
        unique_id = set().union(*self.vehicle_window)
        flow_rate = len(unique_id)
        vehicle_density = current_frame_vehicle_count

        # Data recording
        row_data = [self.frame_number, flow_rate, vehicle_density]
        self.df.loc[len(self.df)] = row_data

        # Convert OpenCV image format to Qt image format
        frame_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

        # Frame number increment
        self.frame_number += 1

    def save_data_and_close(self):
        # Save data to Excel file
        file_path = 'a111.xlsx'
        self.df.to_excel(file_path, index=False)
        # Close the window
        self.close()

    def closeEvent(self, event):
        if self.video_capture:
            self.video_capture.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoProcessor()
    ex.show()
    sys.exit(app.exec_())