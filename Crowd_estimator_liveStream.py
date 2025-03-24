"""
تخمین جمعیت آنلاین از استریم ویدیویی
--------------------------------
این برنامه یک استریم ویدیویی را از فایل ورودی دریافت می‌کند و تعداد افراد حاضر در تصویر را
به صورت لحظه‌ای تخمین می‌زند. از DeepFace برای تشخیص چهره استفاده می‌کند.

نویسنده: AI Assistant
تاریخ: 24 مارس 2025

نحوه استفاده:
    python crowd_estimator.py --input stream_url.txt
    python crowd_estimator.py --input stream_url.txt --output results.csv
"""

import cv2
import numpy as np
import requests
import os
import time
import threading
import tempfile
import matplotlib.pyplot as plt
from deepface import DeepFace
import tensorflow as tf
import argparse
import csv
import sys

class CrowdEstimator:
    def __init__(self, stream_url, output_file=None):
        """
        مقداردهی اولیه کلاس تخمین جمعیت
        
        پارامترها:
            stream_url (str): آدرس استریم ویدیویی M3U8
            output_file (str): فایل خروجی برای ذخیره نتایج (اختیاری)
        """
        self.stream_url = stream_url
        self.output_file = output_file
        self.frame = None
        self.processed_frame = None
        self.crowd_count = 0
        self.running = False
        self.history = []
        self.max_history = 100
        
        # بررسی و استفاده از GPU در صورت وجود
        self.device = "CPU"
        if tf.config.list_physical_devices('GPU'):
            self.device = "GPU"
            print(f"GPU detected: {tf.config.list_physical_devices('GPU')}")
        else:
            print("No GPU detected, using CPU")
            
        # تنظیمات تشخیص چهره
        self.face_detector = "retinaface"  # گزینه‌های دیگر: opencv, ssd, dlib, mtcnn
        self.face_detector_model = DeepFace.build_model(self.face_detector)
        
        # تنظیمات نمایشگر
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.line, = self.ax2.plot([], [])
        self.ax2.set_title('Crowd Count History')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('People Count')
        self.ax2.grid(True)
        
        # آماده‌سازی فایل خروجی CSV اگر مشخص شده باشد
        if self.output_file:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'People Count'])

    def download_m3u8_stream(self):
        """
        دانلود و پردازش استریم M3U8
        """
        try:
            # دریافت فایل M3U8
            response = requests.get(self.stream_url)
            if response.status_code != 200:
                print(f"Error downloading M3U8 file: {response.status_code}")
                return
                
            # پردازش فایل M3U8 برای یافتن آدرس پایه
            m3u8_content = response.text
            base_url = self.stream_url.rsplit('/', 1)[0] + '/'
            
            # تنظیم ویدیو کپچر
            cap = cv2.VideoCapture(self.stream_url)
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame, reconnecting...")
                    cap.release()
                    cap = cv2.VideoCapture(self.stream_url)
                    continue
                    
                self.frame = frame
                time.sleep(0.03)  # کاهش بار CPU
                
            cap.release()
            
        except Exception as e:
            print(f"Error in stream download: {e}")

    def process_frames(self):
        """
        پردازش فریم‌ها و تشخیص افراد
        """
        try:
            while self.running:
                if self.frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame = self.frame.copy()
                
                # تشخیص چهره با DeepFace
                faces = []
                try:
                    faces = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend=self.face_detector,
                        enforce_detection=False,
                        align=True
                    )
                except Exception as e:
                    print(f"Face detection error: {e}")
                
                # شمارش تعداد چهره‌های تشخیص داده شده
                self.crowd_count = len(faces)
                
                # اضافه کردن به تاریخچه
                self.history.append(self.crowd_count)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # ذخیره نتایج در فایل CSV
                if self.output_file:
                    with open(self.output_file, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), self.crowd_count])
                
                # رسم مستطیل اطراف چهره‌ها
                for face in faces:
                    if 'facial_area' in face:
                        facial_area = face['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # نمایش تعداد افراد روی تصویر
                cv2.putText(
                    frame,
                    f"People Count: {self.crowd_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                self.processed_frame = frame
                time.sleep(0.1)  # کاهش بار CPU
                
        except Exception as e:
            print(f"Error in frame processing: {e}")

    def update_plot(self):
        """
        به‌روزرسانی نمودار تعداد جمعیت
        """
        try:
            while self.running:
                if self.processed_frame is not None:
                    # نمایش تصویر پردازش شده
                    self.ax1.clear()
                    self.ax1.imshow(cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB))
                    self.ax1.set_title(f'Live Stream - People Count: {self.crowd_count}')
                    self.ax1.axis('off')
                    
                    # به‌روزرسانی نمودار تاریخچه
                    self.line.set_data(range(len(self.history)), self.history)
                    self.ax2.relim()
                    self.ax2.autoscale_view()
                    
                    plt.draw()
                    plt.pause(0.01)
                    
                time.sleep(0.1)  # کاهش بار CPU
                
        except Exception as e:
            print(f"Error in plot update: {e}")

    def start(self):
        """
        شروع پردازش استریم
        """
        self.running = True
        
        # شروع ترد‌های مختلف برای دانلود و پردازش همزمان
        self.download_thread = threading.Thread(target=self.download_m3u8_stream)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.plot_thread = threading.Thread(target=self.update_plot)
        
        self.download_thread.start()
        time.sleep(1)  # اطمینان از شروع دانلود
        self.process_thread.start()
        self.plot_thread.start()
        
        print(f"Crowd estimation started for stream: {self.stream_url}")
        print("Press Ctrl+C to stop.")
        
        try:
            # نگه داشتن برنامه اصلی
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """
        توقف پردازش استریم
        """
        self.running = False
        
        if hasattr(self, 'download_thread'):
            self.download_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        if hasattr(self, 'plot_thread'):
            self.plot_thread.join()
            
        plt.close()
        print("Crowd estimation stopped.")


def read_stream_url_from_file(file_path):
    """
    خواندن آدرس استریم از فایل
    
    پارامترها:
        file_path (str): مسیر فایل حاوی آدرس استریم
        
    برگشت:
        str: آدرس استریم
    """
    try:
        with open(file_path, 'r') as file:
            stream_url = file.readline().strip()
        return stream_url
    except Exception as e:
        print(f"Error reading stream URL from file: {e}")
        return None


def main():
    """
    تابع اصلی برنامه
    """
    # تعریف آرگومان‌های خط فرمان
    parser = argparse.ArgumentParser(description='Crowd Estimation from Video Stream')
    parser.add_argument('--input', '-i', required=True, help='Path to file containing stream URL')
    parser.add_argument('--output', '-o', help='Path to output CSV file for results')
    
    # اگر هیچ آرگومانی داده نشده، راهنما را نمایش بده
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    # خواندن آدرس استریم از فایل
    stream_url = read_stream_url_from_file(args.input)
    
    if not stream_url:
        print("No valid stream URL found in input file.")
        sys.exit(1)
    
    # ایجاد و شروع تخمین‌زننده جمعیت
    estimator = CrowdEstimator(stream_url, args.output)
    estimator.start()


if __name__ == "__main__":
    main()
