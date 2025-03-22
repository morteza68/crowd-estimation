"""
تخمین جمعیت آنلاین با پردازش گرافیکی
===================================
این برنامه تعداد افراد را در تصاویر زنده تشخیص می‌دهد و از GPU برای پردازش سریع‌تر استفاده می‌کند.
قابلیت بارگیری مستقیم از گیت‌هاب را دارد.

نویسنده: Team Graphics
تاریخ: 1404/01/01
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import requests
import zipfile
import io
from pathlib import Path
from threading import Thread
from queue import Queue

# تنظیم آرگومان‌های ورودی
def parse_args():
    parser = argparse.ArgumentParser(description='تخمین جمعیت آنلاین با یادگیری عمیق')
    parser.add_argument('--source', type=str, default='0', help='منبع ورودی: شماره دوربین یا مسیر ویدیو')
    parser.add_argument('--model', type=str, default='yolov8n', help='مدل مورد استفاده: yolov8n, yolov8s, yolov8m')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='آستانه اطمینان تشخیص')
    parser.add_argument('--device', type=str, default='', help='دستگاه مورد استفاده (cpu, cuda:0, 0)')
    parser.add_argument('--view-img', action='store_true', help='نمایش نتایج')
    parser.add_argument('--save-img', action='store_true', help='ذخیره نتایج')
    parser.add_argument('--github-repo', type=str, default='ultralytics/assets', help='مخزن گیت‌هاب برای بارگیری مدل')
    parser.add_argument('--github-branch', type=str, default='main', help='شاخه گیت‌هاب')
    return parser.parse_args()

# کلاس برای بارگیری مدل از گیت‌هاب
class GitHubModelLoader:
    def __init__(self, repo, branch='main'):
        self.repo = repo
        self.branch = branch
        self.base_url = f"https://raw.githubusercontent.com/{repo}/{branch}/"
        self.api_url = f"https://api.github.com/repos/{repo}/contents"
        
    def download_file(self, file_path, save_path):
        """دانلود فایل از گیت‌هاب"""
        url = self.base_url + file_path
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"فایل '{file_path}' با موفقیت دانلود شد")
            return save_path
        except Exception as e:
            print(f"خطا در دانلود فایل '{file_path}': {e}")
            return None
    
    def download_model(self, model_name, save_dir='models'):
        """دانلود مدل از گیت‌هاب"""
        print(f"در حال دانلود مدل {model_name} از گیت‌هاب...")
        
        # ایجاد مسیر ذخیره‌سازی
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        
        # بررسی اگر مدل قبلاً دانلود شده است
        if os.path.exists(model_path):
            print(f"مدل {model_name} قبلاً دانلود شده است")
            return model_path
        
        # دانلود مدل
        file_path = f"models/{model_name}.pt"
        return self.download_file(file_path, model_path)

# کلاس برای پردازش ویدیو با استفاده از چند نخ
class VideoProcessor:
    def __init__(self, source, model, device, conf_thres=0.5, view_img=True, save_img=False):
        self.source = source
        self.conf_thres = conf_thres
        self.view_img = view_img
        self.save_img = save_img
        self.device = device
        
        # بررسی اگر منبع یک عدد است (دوربین)
        if source.isdigit():
            self.source = int(source)
        
        # تنظیم ویدیو
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"نمی‌توان منبع ویدیویی {source} را باز کرد")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # بارگیری مدل
        self.model = self.load_model(model, device)
        
        # صف‌های چند نخی
        self.frame_queue = Queue(maxsize=4)
        self.result_queue = Queue(maxsize=4)
        self.stopped = False
        
    def load_model(self, model_path, device):
        """بارگیری مدل YOLOv8"""
        try:
            # استفاده از Ultralytics YOLOv8
            import ultralytics
            from ultralytics import YOLO
            
            print(f"در حال بارگیری مدل {model_path} روی {device}...")
            model = YOLO(model_path)
            model.to(device)
            return model
        except ImportError:
            print("لطفاً کتابخانه ultralytics را نصب کنید: pip install ultralytics")
            sys.exit(1)
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
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)  # کمی صبر کنید تا فضای صف خالی شود
    
    def process_frames(self):
        """پردازش فریم‌ها با استفاده از مدل در یک نخ جداگانه"""
        while not self.stopped:
            if not self.frame_queue.empty() and not self.result_queue.full():
                frame = self.frame_queue.get()
                start_time = time.time()
                
                # تشخیص افراد با YOLOv8
                results = self.model(frame, classes=0, conf=self.conf_thres)  # فقط کلاس 0 (انسان)
                
                # زمان پردازش
                process_time = time.time() - start_time
                
                # آماده‌سازی نتایج
                result_frame = results[0].plot()
                
                # شمارش تعداد افراد
                detections = results[0].boxes
                people_count = len(detections)
                
                # افزودن اطلاعات به تصویر
                cv2.putText(result_frame, f"People: {people_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_frame, f"FPS: {1/process_time:.1f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # قرار دادن نتایج در صف
                self.result_queue.put((result_frame, people_count, process_time))
            else:
                time.sleep(0.01)  # کمی صبر کنید تا فریم جدید دریافت شود
    
    def display_frames(self):
        """نمایش نتایج پردازش در نخ اصلی"""
        frame_count = 0
        total_people = 0
        total_time = 0
        start_time = time.time()
        
        if self.save_img:
            # ایجاد پوشه برای ذخیره نتایج
            save_dir = Path('output')
            save_dir.mkdir(exist_ok=True)
            
            # تنظیم ویدیو نویسنده
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = save_dir / f'result_{int(time.time())}.mp4'
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        while not self.stopped or not self.result_queue.empty():
            if not self.result_queue.empty():
                result_frame, people_count, process_time = self.result_queue.get()
                
                # به‌روزرسانی آمار
                frame_count += 1
                total_people += people_count
                total_time += process_time
                
                # نمایش تصویر
                if self.view_img:
                    cv2.imshow("Crowd Estimation", result_frame)
                    
                    # خروج با کلید q
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stopped = True
                        break
                
                # ذخیره تصویر
                if self.save_img:
                    writer.write(result_frame)
        
        # نمایش آمار نهایی
        elapsed_time = time.time() - start_time
        avg_people = total_people / frame_count if frame_count > 0 else 0
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nآمار نهایی:")
        print(f"تعداد فریم‌ها: {frame_count}")
        print(f"میانگین تعداد افراد: {avg_people:.1f}")
        print(f"میانگین FPS: {avg_fps:.1f}")
        
        # آزادسازی منابع
        if self.save_img:
            writer.release()
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

# تابع اصلی
def main():
    # تنظیم آرگومان‌ها
    args = parse_args()
    
    # بررسی نیازمندی‌ها
    try:
        import torch
        import ultralytics
    except ImportError:
        print("در حال نصب نیازمندی‌ها...")
        os.system("pip install ultralytics")
        import torch
        import ultralytics
    
    # تنظیم دستگاه پردازش
    if args.device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"استفاده از دستگاه: {device}")
    
    # بارگیری مدل از گیت‌هاب
    model_path = f"{args.model}.pt"
    if not os.path.exists(model_path):
        loader = GitHubModelLoader(args.github_repo, args.github_branch)
        model_path = loader.download_model(args.model)
        if model_path is None:
            print(f"خطا در دانلود مدل {args.model}. در حال استفاده از مدل پیش‌فرض...")
            # نصب مدل با ultralytics
            os.system(f"pip install {args.model}")
            model_path = args.model
    
    # شروع پردازش ویدیو
    processor = VideoProcessor(
        source=args.source,
        model=model_path,
        device=device,
        conf_thres=args.conf_thres,
        view_img=args.view_img,
        save_img=args.save_img
    )
    
    processor.start()

# نقطه ورود برنامه
if __name__ == "__main__":
    main()