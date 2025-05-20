import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGridLayout, QDialog, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import matplotlib.pyplot as plt

class ScaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ölçeklendirme Ayarları")
        self.layout = QVBoxLayout()
        self.width_input = QLineEdit(self)
        self.width_input.setPlaceholderText("Yeni Genişlik")
        self.height_input = QLineEdit(self)
        self.height_input.setPlaceholderText("Yeni Yükseklik")
        self.ok_button = QPushButton("Tamam", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.width_input)
        self.layout.addWidget(self.height_input)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def get_dimensions(self):
        width = int(self.width_input.text()) if self.width_input.text() else self.parent().image.shape[1]
        height = int(self.height_input.text()) if self.height_input.text() else self.parent().image.shape[0]
        return width, height

class GorselIslemeArayuzu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gelişmiş Görüntü İşleme Uygulaması")
        self.setGeometry(100, 100, 800, 600)
        self.image = None
        self.original_image = None
        self.history = []
        self.future = []

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("Görüntü İşleme Uygulaması")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #bdc3c7; background-color: #ecf0f1;")
        self.layout.addWidget(self.image_label)

        self.button_layout = QGridLayout()
        buttons = [
            ("Görüntü Yükle", self.load_image),
            ("Kaydet", self.save_image),
            ("Geriye Al", self.undo),
            ("İleri Al", self.redo),
            ("Parlaklık Arttır", self.increase_brightness),
            ("Parlaklık Azalt", self.decrease_brightness),
            ("Kontrast Arttır", self.increase_contrast),
            ("Negatif Al", self.negative_image),
            ("Dörtgen Çiz", self.draw_rectangle),
            ("Histogram Göster", self.show_histogram),
            ("Döndür (45°)", self.rotate_image),
            ("Bulanıklaştır", self.blur_image),
            ("Kenar Bul (Canny)", self.detect_edges),
            ("Kenar Bul (Sobel)", self.detect_edges_sobel),
            ("Kapalı Alan Renklendir", self.color_contours),
            ("İkinci Resmi Ekle", self.add_second_image),
            ("AND İşlemi", self.apply_and),
            ("NAND İşlemi", self.apply_nand),
            ("OR İşlemi", self.apply_or),
            ("NOR İşlemi", self.apply_nor),
            ("XOR İşlemi", self.apply_xor),
            ("Arka Planı Kaldır", self.remove_background),
            ("Gri Tonlama", self.convert_to_grayscale),
            ("Renk Kanallarını Göster", self.show_color_channels),
            ("Daire Çiz", self.draw_circle),
            ("Elips Çiz", self.draw_ellipse),
            ("Çokgen Çiz", self.draw_polygon),
            ("Çerçeve Dışını Kırp", self.crop_outside_frame),
            ("Ölçeklendir (2x)", self.resize_image),
            ("Aynala", self.flip_image),
            ("Eğ", self.warp_perspective),
            ("Ölçek Ayarla", self.scale_with_properties),
            ("Netleştir", self.sharpen_image),
            ("Asındır", self.erode_image),
            ("Genişlet", self.dilate_image),
            ("Resimleri Çıkar", self.subtract_images),
            ("Ağaç ve Gölge Birleştir", self.combine_tree_and_shadow)
        ]

        positions = [(i // 4, i % 4) for i in range(len(buttons))]
        for (text, func), pos in zip(buttons, positions):
            button = QPushButton(text)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            button.clicked.connect(func)
            self.button_layout.addWidget(button, *pos)

        self.layout.addLayout(self.button_layout)

        self.footer_label = QLabel("© 2025 Görüntü İşleme Projesi")
        self.footer_label.setStyleSheet("font-size: 12px; color: #7f8c8d; padding: 5px;")
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.footer_label)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.image = self.original_image.copy()
            self.history = [self.image.copy()]
            self.future = []
            self.display_image(self.image)

    def save_image(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, self.image)

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def save_to_history(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            self.future = []

    def undo(self):
        if len(self.history) > 1:
            self.future.append(self.history.pop())
            self.image = self.history[-1].copy()
            self.display_image(self.image)

    def redo(self):
        if self.future:
            self.history.append(self.image.copy())
            self.image = self.future.pop().copy()
            self.display_image(self.image)

    def increase_brightness(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.convertScaleAbs(self.image, beta=50)
            self.display_image(self.image)
            print("Parlaklık artırıldı!")

    def decrease_brightness(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.convertScaleAbs(self.image, beta=-50)
            self.display_image(self.image)
            print("Parlaklık azaltıldı!")

    def increase_contrast(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.convertScaleAbs(self.image, alpha=1.5)
            self.display_image(self.image)

    def negative_image(self):
        if self.image is not None:
            self.save_to_history()
            self.image = 255 - self.image
            self.display_image(self.image)

    def draw_rectangle(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.rectangle(self.image, (50, 50), (200, 200), (0, 255, 0), 3)
            self.display_image(self.image)

    def show_histogram(self):
        if self.image is not None:
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Histogram")
            plt.show()

    def rotate_image(self):
        if self.image is not None:
            self.save_to_history()
            rows, cols = self.image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
            self.image = cv2.warpAffine(self.image, M, (cols, rows))
            self.display_image(self.image)

    def blur_image(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.GaussianBlur(self.image, (15, 15), 0)
            self.display_image(self.image)

    def detect_edges(self):
        if self.image is not None:
            self.save_to_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(gray, 100, 200)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)

    def detect_edges_sobel(self):
        if self.image is not None:
            self.save_to_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            self.image = cv2.convertScaleAbs(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)

    def color_contours(self):
        if self.image is not None:
            self.save_to_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.image, contours, -1, (0, 255, 0), 3)
            self.display_image(self.image)

    def add_second_image(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.save_to_history()
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.image = cv2.addWeighted(self.image, 0.5, second_image, 0.5, 0)
                self.display_image(self.image)

    def apply_and(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.save_to_history()
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.image = cv2.bitwise_and(self.image, second_image)
                self.display_image(self.image)

    def apply_nand(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.save_to_history()
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.image = cv2.bitwise_not(cv2.bitwise_and(self.image, second_image))
                self.display_image(self.image)

    def apply_or(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.save_to_history()
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.image = cv2.bitwise_or(self.image, second_image)
                self.display_image(self.image)

    def apply_nor(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.save_to_history()
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.image = cv2.bitwise_not(cv2.bitwise_or(self.image, second_image))
                self.display_image(self.image)

    def apply_xor(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.save_to_history()
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.image = cv2.bitwise_xor(self.image, second_image)
                self.display_image(self.image)

    def remove_background(self):
        if self.image is not None:
            self.save_to_history()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            self.display_image(self.image)

    def convert_to_grayscale(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)

    def show_color_channels(self):
        if self.image is not None:
            b, g, r = cv2.split(self.image)
            channels = [b, g, r]
            titles = ['Blue Channel', 'Green Channel', 'Red Channel']
            for i in range(3):
                plt.figure()
                plt.imshow(channels[i], cmap='gray')
                plt.title(titles[i])
                plt.axis('off')
            plt.show()

    def draw_circle(self):
        if self.image is not None:
            self.save_to_history()
            center = (self.image.shape[1]//2, self.image.shape[0]//2)
            self.image = cv2.circle(self.image, center, 100, (0, 255, 0), 3)
            self.display_image(self.image)

    def draw_ellipse(self):
        if self.image is not None:
            self.save_to_history()
            center = (self.image.shape[1]//2, self.image.shape[0]//2)
            self.image = cv2.ellipse(self.image, center, (100, 50), 0, 0, 360, (0, 255, 0), 3)
            self.display_image(self.image)

    def draw_polygon(self):
        if self.image is not None:
            self.save_to_history()
            pts = np.array([[100, 100], [200, 50], [300, 100], [250, 200]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            self.image = cv2.polylines(self.image, [pts], True, (0, 255, 0), 3)
            self.display_image(self.image)

    def crop_outside_frame(self):
        if self.image is not None:
            self.save_to_history()
            center = (self.image.shape[1]//2, self.image.shape[0]//2)
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, 100, 255, -1)
            self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            self.display_image(self.image)

    def resize_image(self):
        if self.image is not None:
            self.save_to_history()
            plt.hist(self.image.ravel(), 256, [0, 256])
            plt.title("Ölçekleme Öncesi Histogram")
            plt.show()
            self.image = cv2.resize(self.image, (self.image.shape[1]*2, self.image.shape[0]*2))
            self.display_image(self.image)

    def flip_image(self):
        if self.image is not None:
            self.save_to_history()
            self.image = cv2.flip(self.image, 1)
            self.display_image(self.image)

    def warp_perspective(self):
        if self.image is not None:
            self.save_to_history()
            rows, cols = self.image.shape[:2]
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.image = cv2.warpPerspective(self.image, matrix, (cols, rows))
            self.display_image(self.image)

    def scale_with_properties(self):
        if self.image is not None:
            self.save_to_history()
            dialog = ScaleDialog(self)
            if dialog.exec_():
                width, height = dialog.get_dimensions()
                self.image = cv2.resize(self.image, (width, height))
                self.display_image(self.image)

    def sharpen_image(self):
        if self.image is not None:
            self.save_to_history()
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.image = cv2.filter2D(self.image, -1, kernel)
            self.display_image(self.image)

    def erode_image(self):
        if self.image is not None:
            self.save_to_history()
            kernel = np.ones((5,5), np.uint8)
            self.image = cv2.erode(self.image, kernel, iterations=1)
            self.display_image(self.image)

    def dilate_image(self):
        if self.image is not None:
            self.save_to_history()
            kernel = np.ones((5,5), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=1)
            self.display_image(self.image)

    def subtract_images(self):
        if self.image is not None:
            self.save_to_history()
            file_path, _ = QFileDialog.getOpenFileName(self, "İkinci Görüntüyü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                second_image = cv2.imread(file_path)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                result = cv2.subtract(self.image, second_image)
                result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                self.image = result
                self.display_image(self.image)

    def combine_tree_and_shadow(self):
        if self.image is not None:
            self.save_to_history()
            file_path, _ = QFileDialog.getOpenFileName(self, "Gölge Görüntüsünü Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                shadow_image = cv2.imread(file_path)
                shadow_image = cv2.resize(shadow_image, (self.image.shape[1], self.image.shape[0]))
                gray_tree = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_tree, 1, 255, cv2.THRESH_BINARY)
                self.image = cv2.addWeighted(self.image, 0.7, shadow_image, 0.3, 0)
                self.display_image(self.image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GorselIslemeArayuzu()
    window.show()
    sys.exit(app.exec_())