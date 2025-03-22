"""
تخمین جمعیت آنلاین با پردازش گرافیکی و نمایش پیشرفت با tqdm
=============================================================
این برنامه تعداد افراد را در تصاویر زنده تشخیص می‌دهد، از GPU برای پردازش سریع‌تر استفاده می‌کند،
و با استفاده از tqdm پیشرفت عملیات را نمایش می‌دهد.

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

# اضافه کردن tqdm برای نمایش پیشرفت
from tqdm import tqdm

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
    parser.add_argument('--max-frames', type=int, default=0, help='حداکثر تعداد فریم‌ها برای پردازش (0 = نامحدود)')
    return parser.parse_args()

# کلاس برای بارگیری مدل از گیت‌هاب با نمایش پیشرفت
class GitHubModelLoader:
    def __init__(self, repo, branch='main'):
        self.repo = repo
        self.branch = branch
        self.base_url = f"https://raw.githubusercontent.com/{repo}/{branch}/"
        self.api_url = f"https://api.github.com/repos/{repo}/contents"
        
    def download_file(self, file_path, save_path):
        """دانلود فایل از گیت‌هاب با نمایش پیشرفت"""
        url = self.base_url + file_path
        try:
            # درخواست HEAD برای دریافت اندازه فایل
            head_response = requests.head(url)
            file_size = int(head_response.headers.get('content-length', 0))
            
            print(f"شروع دانلود فایل '{file_path}' با حجم {file_size/1024/1024:.1f} MB")
            
            # دانلود فایل با نمایش پیشرفت
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f, tqdm(
                desc=f"دانلود {os.path.basename(file_path)}",
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            print(f"فایل '{file_path}' با موفقیت دانلود شد")
            return save_path
        except Exception as e:
            print(f"خطا در دانلود فایل '{file_path}': {e}")
            return None
    
    def download_model(self, model_name, save_dir='models'):
        """دانلود مدل از گیت‌هاب"""
        print(f"در حال آماده‌سازی مدل {model_name}...")
        
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

# کلاس برای پردازش ویدیو با استفاده از چند نخ و نمایش پیشرفت
class VideoProcessor:
    def __init__(self, source, model, device, conf_thres=0.5, view_img=True, save_img=False, max_frames=0):
        self.source = source
        self.conf_thres = conf_thres
        self.view_img = view_img
        self.save_img = save_img
        self.device = device
        self.max_frames = max_frames
        
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
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source != int(source) else 0
        
        # تنظیم حداکثر تعداد فریم‌ها
        if self.max_frames > 0 and self.max_frames < self.total_frames:
            self.total_frames = self.max_frames
            
        # بارگیری مدل
        self.model = self.load_model(model, device)
        
        # صف‌های چند نخی
        self.frame_queue = Queue(maxsize=4)
        self.result_queue = Queue(maxsize=4)
        self.stopped = False
        
    def load_model(self, model_path, device):
        """بارگیری مدل YOLOv8 با نمایش پیشرفت"""
        try:
            # استفاده از Ultralytics YOLOv8
            import ultralytics
            from ultralytics import YOLO
            
            print(f"در حال بارگیری مدل {model_path} روی {device}...")
            
            # نمایش نوار پیشرفت برای بارگیری مدل
            with tqdm(desc="بارگیری مدل", total=100) as pbar:
                pbar.update(10)  # شروع بارگیری
                model = YOLO(model_path)
                pbar.update(60)  # مدل بارگیری شد
                model.to(device)
                pbar.update(30)  # انتقال به دستگاه انجام شد
            
            print(f"مدل {model_path} با موفقیت بارگیری شد")
            return model
        except ImportError:
            print("لطفاً کتابخانه ultralytics را نصب کنید: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"خطا در بارگیری مدل: {e}")
            sys.exit(1)
    
    def read_frames(self):
        """خواندن فریم‌ها از منبع ویدیویی در یک نخ جداگانه با نمایش پیشرفت"""
        # ایجاد نوار پیشرفت برای خواندن فریم‌ها
        frame_count = 0
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="خواندن فریم‌ها", position=0)
        
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                    
                self.frame_queue.put(frame)
                frame_count += 1
                
                # به‌روزرسانی نوار پیشرفت
                if pbar is not None:
                    pbar.update(1)
                
                # بررسی حداکثر تعداد فریم‌ها
                if self.max_frames > 0 and frame_count >= self.max_frames:
                    self.stopped = True
                    break
            else:
                time.sleep(0.01)  # کمی صبر کنید تا فضای صف خالی شود
        
        # بستن نوار پیشرفت
        if pbar is not None:
            pbar.close()
    
    def process_frames(self):
        """پردازش فریم‌ها با استفاده از مدل در یک نخ جداگانه با نمایش پیشرفت"""
        # ایجاد نوار پیشرفت برای پردازش فریم‌ها
        frame_count = 0
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="پردازش فریم‌ها", position=1)
        
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
                
                # به‌روزرسانی نوار پیشرفت
                if pbar is not None:
                    pbar.update(1)
                
                frame_count += 1
            else:
                time.sleep(0.01)  # کمی صبر کنید تا فریم جدید دریافت شود
                
                # بررسی اگر کار خواندن فریم‌ها تمام شده و صف خالی است
                if self.stopped and self.frame_queue.empty():
                    break
        
        # بستن نوار پیشرفت
        if pbar is not None:
            pbar.close()
    
    def display_frames(self):
        """نمایش نتایج پردازش در نخ اصلی با نمایش پیشرفت"""
        frame_count = 0
        total_people = 0
        total_time = 0
        start_time = time.time()
        
        # ایجاد نوار پیشرفت برای نمایش نتایج
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="نمایش نتایج", position=2)
        
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
                
                # به‌روزرسانی نوار پیشرفت
                if pbar is not None:
                    pbar.update(1)
            else:
                time.sleep(0.01)
                
                # بررسی اگر همه فریم‌ها پردازش شده‌اند
                if self.stopped and self.result_queue.empty():
                    break
        
        # بستن نوار پیشرفت
        if pbar is not None:
            pbar.close()
        
        # نمایش آمار نهایی
        elapsed_time = time.time() - start_time
        avg_people = total_people / frame_count if frame_count > 0 else 0
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nآمار نهایی:")
        print(f"تعداد فریم‌ها: {frame_count}")
        print(f"میانگین تعداد افراد: {avg_people:.1f}")
        print(f"میانگین FPS: {avg_fps:.1f}")
        print(f"زمان کل پردازش: {elapsed_time:.1f} ثانیه")
        
        # آزادسازی منابع
        if self.save_img:
            writer.release()
        self.cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """شروع پردازش ویدیو با استفاده از چند نخ"""
        print("\nشروع پردازش ویدیو...")
        
        # بررسی حالت ویدیوی زنده یا فایل
        if self.total_frames == 0:
            print("منبع: دوربین یا ویدیوی زنده (تعداد فریم‌ها نامشخص)")
        else:
            print(f"منبع: فایل ویدیو با {self.total_frames} فریم")
            
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
        
        print("پردازش ویدیو با موفقیت به پایان رسید")

# بررسی و نصب نیازمندی‌ها با نمایش پیشرفت
def check_requirements():
    required_packages = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'tqdm': 'tqdm',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    # بررسی هر بسته
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    # نصب بسته‌های مفقود
    if missing_packages:
        print(f"نصب {len(missing_packages)} بسته مورد نیاز...")
        
        for package in tqdm(missing_packages, desc="نصب بسته‌ها"):
            print(f"در حال نصب {package}...")
            os.system(f"pip install {package}")
            print(f"{package} با موفقیت نصب شد")
    
    return True

# تابع اصلی
def main():
    # تنظیم آرگومان‌ها
    args = parse_args()
    
    print("=" * 50)
    print("سیستم تخمین جمعیت آنلاین")
    print("=" * 50)
    
    # بررسی نیازمندی‌ها
    with tqdm(total=1, desc="بررسی نیازمندی‌ها") as pbar:
        check_requirements()
        pbar.update(1)
    
    # وارد کردن کتابخانه‌های لازم
    import torch
    import ultralytics
    from ultralytics import YOLO
    
    # تنظیم دستگاه پردازش
    if args.device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"استفاده از دستگاه: {device}")
    
    # بررسی اطلاعات دستگاه
    if 'cuda' in device:
        with tqdm(total=3, desc="بررسی اطلاعات GPU") as pbar:
            gpu_name = torch.cuda.get_device_name(0)
            pbar.update(1)
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pbar.update(1)
            
            print(f"GPU: {gpu_name} با {gpu_memory:.2f} GB حافظه")
            pbar.update(1)
    
    # بارگیری مدل از گیت‌هاب
    model_path = f"{args.model}.pt"
    if not os.path.exists(model_path):
        with tqdm(total=1, desc="آماده‌سازی مدل") as pbar:
            loader = GitHubModelLoader(args.github_repo, args.github_branch)
            model_path = loader.download_model(args.model)
            pbar.update(1)
            
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
        save_img=args.save_img,
        max_frames=args.max_frames
    )
    
    processor.start()

# نقطه ورود برنامه
if __name__ == "__main__":
    main()