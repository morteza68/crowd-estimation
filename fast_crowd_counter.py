"""
تخمین جمعیت آنلاین با پردازش گرافیکی - نسخه بهینه‌سازی شده برای سرعت
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from threading import Thread
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser(description='تخمین جمعیت آنلاین با یادگیری عمیق - بهینه‌سازی شده')
    parser.add_argument('--source', type=str, default='0', help='منبع ورودی: شماره دوربین یا مسیر ویدیو')
    parser.add_argument('--model', type=str, default='yolov8n', help='مدل مورد استفاده: yolov8n, yolov8s')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='آستانه اطمینان تشخیص')
    parser.add_argument('--device', type=str, default='', help='دستگاه مورد استفاده (cpu, cuda:0, 0)')
    parser.add_argument('--view-img', action='store_true', help='نمایش نتایج')
    parser.add_argument('--img-size', type=int, default=320, help='اندازه تصویر ورودی')
    parser.add_argument('--frame-skip', type=int, default=2, help='پرش فریم (هر چند فریم یکبار پردازش شود)')
    return parser.parse_args()

class FastVideoProcessor:
    def __init__(self, source, model, device, conf_thres=0.5, view_img=True, img_size=320, frame_skip=2):
        self.source = source if not source.isdigit() else int(source)
        self.conf_thres = conf_thres
        self.view_img = view_img
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.device = device
        
        # تنظیم ویدیو
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"نمی‌توان منبع ویدیویی {source} را باز کرد")
        
        # تنظیم رزولوشن پایین‌تر برای وبکم
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # بارگیری مدل
        self.model = self.load_model(model, device)
        
        # صف‌های چند نخی
        self.frame_queue = Queue(maxsize=2)  # کاهش اندازه صف برای کاهش تأخیر
        self.result_queue = Queue(maxsize=2)
        self.stopped = False
        
        # شمارنده فریم
        self.frame_counter = 0
        
    def load_model(self, model_path, device):
        try:
            import ultralytics
            from ultralytics import YOLO
            
            print(f"در حال بارگیری مدل {model_path} روی {device}...")
            
            # اگر فقط نام مدل داده شده (نه مسیر فایل)
            if not model_path.endswith('.pt') and not model_path.endswith('.onnx'):
                model_path = f"{model_path}.pt"
            
            model = YOLO(model_path)
            
            # انتقال به دستگاه مناسب
            model.to(device)
            
            # استفاده از FP16 برای سرعت بیشتر در GPU
            if device != 'cpu' and torch.cuda.is_available():
                model.model.half()
            
            print(f"مدل با موفقیت بارگیری شد")
            return model
            
        except ImportError:
            print("در حال نصب ultralytics...")
            os.system("pip install ultralytics")
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(device)
            return model
        except Exception as e:
            print(f"خطا در بارگیری مدل: {e}")
            sys.exit(1)
    
    def read_frames(self):
        """خواندن فریم‌ها از منبع ویدیویی در یک نخ جداگانه"""
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                
                # تغییر اندازه فریم برای سرعت بیشتر
                frame = cv2.resize(frame, (self.img_size, int(self.img_size * self.height / self.width)))
                
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)  # تأخیر کمتر
    
    def process_frames(self):
        """پردازش فریم‌ها با استفاده از مدل در یک نخ جداگانه"""
        while not self.stopped:
            if not self.frame_queue.empty() and not self.result_queue.full():
                frame = self.frame_queue.get()
                self.frame_counter += 1
                
                # پردازش هر چند فریم یکبار
                if self.frame_counter % self.frame_skip == 0:
                    start_time = time.time()
                    
                    # تشخیص افراد با YOLOv8
                    results = self.model(frame, classes=0, conf=self.conf_thres, verbose=False)
                    
                    # زمان پردازش
                    process_time = time.time() - start_time
                    
                    # آماده‌سازی نتایج
                    result_frame = results[0].plot()
                    
                    # شمارش تعداد افراد
                    detections = results[0].boxes
                    people_count = len(detections)
                    
                    # افزودن اطلاعات به تصویر
                    cv2.putText(result_frame, f"People: {people_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(result_frame, f"FPS: {1/process_time:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # قرار دادن نتایج در صف
                    self.result_queue.put((result_frame, people_count, process_time))
                else:
                    # برای فریم‌های پرش شده، فریم اصلی را بدون پردازش ارسال کنید
                    self.result_queue.put((frame, -1, 0))
            else:
                time.sleep(0.001)  # تأخیر کمتر
    
    def display_frames(self):
        """نمایش نتایج پردازش در نخ اصلی"""
        frame_count = 0
        total_people = 0
        total_time = 0
        processed_frames = 0
        start_time = time.time()
        last_people_count = 0
        
        while not self.stopped or not self.result_queue.empty():
            if not self.result_queue.empty():
                result_frame, people_count, process_time = self.result_queue.get()
                frame_count += 1
                
                # اگر این فریم پردازش شده است
                if people_count >= 0:
                    processed_frames += 1
                    total_time += process_time
                    total_people += people_count
                    last_people_count = people_count
                else:
                    # برای فریم‌های پرش شده، از آخرین شمارش استفاده کنید
                    cv2.putText(result_frame, f"People: {last_people_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # محاسبه FPS کلی
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # نمایش FPS کلی
                cv2.putText(result_frame, f"Overall FPS: {fps:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # نمایش تصویر
                if self.view_img:
                    cv2.imshow("Crowd Estimation (Optimized)", result_frame)
                    
                    # خروج با کلید q
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stopped = True
                        break
        
        # نمایش آمار نهایی
        avg_people = total_people / processed_frames if processed_frames > 0 else 0
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        overall_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nآمار نهایی:")
        print(f"تعداد کل فریم‌ها: {frame_count}")
        print(f"تعداد فریم‌های پردازش شده: {processed_frames}")
        print(f"میانگین تعداد افراد: {avg_people:.1f}")
        print(f"میانگین FPS پردازش: {avg_fps:.1f}")
        print(f"میانگین FPS کلی: {overall_fps:.1f}")
        
        # آزادسازی منابع
        self.cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """شروع پردازش ویدیو با استفاده از چند نخ"""
        # ایجاد نخ‌ها
        read_thread = Thread(target=self.read_frames, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        
        # شروع نخ‌ها
        read_thread.start()
        process_thread.start()
        
        # نمایش نتایج در نخ اصلی
        self.display_frames()
        
        # منتظر اتمام نخ‌ها
        self.stopped = True
        read_thread.join()
        process_thread.join()

def main():
    # تنظیم آرگومان‌ها
    args = parse_args()
    
    # تنظیم دستگاه پردازش
    if args.device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"استفاده از دستگاه: {device}")
    print(f"اندازه تصویر: {args.img_size}px")
    print(f"پرش فریم: هر {args.frame_skip} فریم")
    
    # شروع پردازش ویدیو
    processor = FastVideoProcessor(
        source=args.source,
        model=args.model,
        device=device,
        conf_thres=args.conf_thres,
        view_img=args.view_img,
        img_size=args.img_size,
        frame_skip=args.frame_skip
    )
    
    processor.start()

if __name__ == "__main__":
    main()