"""
Online Crowd Estimation with GPU Processing and Progress Display
===============================================================
This program detects people in live images, uses GPU for faster processing,
and displays progress using tqdm.

Author: Team Graphics
Date: 2025-03-22
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

# Add tqdm for progress display
from tqdm import tqdm

# Set up input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Online Crowd Estimation with Deep Learning')
    parser.add_argument('--source', type=str, default='0', help='Input source: camera number or video path')
    parser.add_argument('--model', type=str, default='yolov8n', help='Model: yolov8n, yolov8s, yolov8m')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda:0, 0)')
    parser.add_argument('--view-img', action='store_true', help='Display results')
    parser.add_argument('--save-img', action='store_true', help='Save results')
    parser.add_argument('--github-repo', type=str, default='ultralytics/assets', help='GitHub repository for model download')
    parser.add_argument('--github-branch', type=str, default='main', help='GitHub branch')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum frames to process (0 = unlimited)')
    return parser.parse_args()

# Class for loading model from GitHub with progress display
class GitHubModelLoader:
    def __init__(self, repo, branch='main'):
        self.repo = repo
        self.branch = branch
        self.base_url = f"https://raw.githubusercontent.com/{repo}/{branch}/"
        self.api_url = f"https://api.github.com/repos/{repo}/contents"
        
    def download_file(self, file_path, save_path):
        """Download file from GitHub with progress display"""
        url = self.base_url + file_path
        try:
            # HEAD request to get file size
            head_response = requests.head(url)
            file_size = int(head_response.headers.get('content-length', 0))
            
            print(f"Starting download of '{file_path}' with size {file_size/1024/1024:.1f} MB")
            
            # Download file with progress display
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f, tqdm(
                desc=f"Downloading {os.path.basename(file_path)}",
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            print(f"File '{file_path}' downloaded successfully")
            return save_path
        except Exception as e:
            print(f"Error downloading file '{file_path}': {e}")
            return None
    
    def download_model(self, model_name, save_dir='models'):
        """Download model from GitHub"""
        print(f"Preparing model {model_name}...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        
        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model {model_name} already downloaded")
            return model_path
        
        # Download model
        file_path = f"models/{model_name}.pt"
        return self.download_file(file_path, model_path)

# Class for video processing using multiple threads and progress display
class VideoProcessor:
    def __init__(self, source, model, device, conf_thres=0.5, view_img=True, save_img=False, max_frames=0):
        self.source = source
        self.conf_thres = conf_thres
        self.view_img = view_img
        self.save_img = save_img
        self.device = device
        self.max_frames = max_frames
        
        # Check if source is a number (camera)
        if source.isdigit():
            self.source = int(source)
        
        # Set up video
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source != int(source) else 0
        
        # Set maximum number of frames
        if self.max_frames > 0 and self.max_frames < self.total_frames:
            self.total_frames = self.max_frames
            
        # Load model
        self.model = self.load_model(model, device)
        
        # Multi-threading queues
        self.frame_queue = Queue(maxsize=4)
        self.result_queue = Queue(maxsize=4)
        self.stopped = False
        
    def load_model(self, model_path, device):
        """Load YOLOv8 model with progress display"""
        try:
            # Use Ultralytics YOLOv8
            import ultralytics
            from ultralytics import YOLO
            
            print(f"Loading model {model_path} on {device}...")
            
            # Show progress bar for model loading
            with tqdm(desc="Loading model", total=100) as pbar:
                pbar.update(10)  # Start loading
                model = YOLO(model_path)
                pbar.update(60)  # Model loaded
                model.to(device)
                pbar.update(30)  # Transferred to device
            
            print(f"Model {model_path} loaded successfully")
            return model
        except ImportError:
            print("Please install ultralytics: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def read_frames(self):
        """Read frames from video source in a separate thread with progress display"""
        # Create progress bar for reading frames
        frame_count = 0
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="Reading frames", position=0)
        
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                    
                self.frame_queue.put(frame)
                frame_count += 1
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                
                # Check maximum number of frames
                if self.max_frames > 0 and frame_count >= self.max_frames:
                    self.stopped = True
                    break
            else:
                time.sleep(0.01)  # Wait a bit for queue space
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
    
    def process_frames(self):
        """Process frames using the model in a separate thread with progress display"""
        # Create progress bar for processing frames
        frame_count = 0
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="Processing frames", position=1)
        
        while not self.stopped:
            if not self.frame_queue.empty() and not self.result_queue.full():
                frame = self.frame_queue.get()
                start_time = time.time()
                
                # Detect people with YOLOv8
                results = self.model(frame, classes=0, conf=self.conf_thres)  # Only class 0 (person)
                
                # Processing time
                process_time = time.time() - start_time
                
                # Prepare results
                result_frame = results[0].plot()
                
                # Count people
                detections = results[0].boxes
                people_count = len(detections)
                
                # Add information to image
                cv2.putText(result_frame, f"People: {people_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_frame, f"FPS: {1/process_time:.1f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add exit instructions
                cv2.putText(result_frame, "Press 'q' to quit", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Put results in queue
                self.result_queue.put((result_frame, people_count, process_time))
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                
                frame_count += 1
            else:
                time.sleep(0.01)  # Wait a bit for new frame
                
                # Check if frame reading is done and queue is empty
                if self.stopped and self.frame_queue.empty():
                    break
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
    
    def display_frames(self):
        """Display processing results in main thread with progress display"""
        frame_count = 0
        total_people = 0
        total_time = 0
        start_time = time.time()
        
        # Create progress bar for displaying results
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="Displaying results", position=2)
        
        if self.save_img:
            # Create directory for saving results
            save_dir = Path('output')
            save_dir.mkdir(exist_ok=True)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = save_dir / f'result_{int(time.time())}.mp4'
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        while not self.stopped or not self.result_queue.empty():
            if not self.result_queue.empty():
                result_frame, people_count, process_time = self.result_queue.get()
                
                # Update statistics
                frame_count += 1
                total_people += people_count
                total_time += process_time
                
                # Display image
                if self.view_img:
                    cv2.imshow("Crowd Estimation - Press 'q' to quit", result_frame)
                    
                    # Exit with 'q' key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Exit requested by user (q key pressed)")
                        self.stopped = True
                        break
                
                # Save image
                if self.save_img:
                    writer.write(result_frame)
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
            else:
                time.sleep(0.01)
                
                # Check if all frames are processed
                if self.stopped and self.result_queue.empty():
                    break
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
        
        # Display final statistics
        elapsed_time = time.time() - start_time
        avg_people = total_people / frame_count if frame_count > 0 else 0
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("\nFinal Statistics:")
        print(f"Total frames: {frame_count}")
        print(f"Average people count: {avg_people:.1f}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total processing time: {elapsed_time:.1f} seconds")
        
        # Release resources
        if self.save_img:
            writer.release()
        self.cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """Start video processing using multiple threads"""
        print("\nStarting video processing...")
        
        # Check live video or file
        if self.total_frames == 0:
            print("Source: Camera or live video (unknown frame count)")
        else:
            print(f"Source: Video file with {self.total_frames} frames")
            
        # Create threads
        read_thread = Thread(target=self.read_frames, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        
        # Start threads
        read_thread.start()
        process_thread.start()
        
        # Display results in main thread
        self.display_frames()
        
        # Wait for threads to finish
        self.stopped = True
        read_thread.join()
        process_thread.join()
        
        print("Video processing completed successfully")

# Check and install requirements with progress display
def check_requirements():
    required_packages = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'tqdm': 'tqdm',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    # Check each package
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"Installing {len(missing_packages)} required packages...")
        
        for package in tqdm(missing_packages, desc="Installing packages"):
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
            print(f"{package} installed successfully")
    
    return True

# Main function
def main():
    # Set up arguments
    args = parse_args()
    
    print("=" * 50)
    print("ONLINE CROWD ESTIMATION SYSTEM")
    print("=" * 50)
    
    # Check requirements
    with tqdm(total=1, desc="Checking requirements") as pbar:
        check_requirements()
        pbar.update(1)
    
    # Import required libraries
    import torch
    import ultralytics
    from ultralytics import YOLO
    
    # Set up processing device
    if args.device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check device information
    if 'cuda' in device:
        with tqdm(total=3, desc="Checking GPU information") as pbar:
            gpu_name = torch.cuda.get_device_name(0)
            pbar.update(1)
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pbar.update(1)
            
            print(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
            pbar.update(1)
    
    # Load model from GitHub
    model_path = f"{args.model}.pt"
    if not os.path.exists(model_path):
        with tqdm(total=1, desc="Preparing model") as pbar:
            loader = GitHubModelLoader(args.github_repo, args.github_branch)
            model_path = loader.download_model(args.model)
            pbar.update(1)
            
            if model_path is None:
                print(f"Error downloading model {args.model}. Using default model...")
                # Install model with ultralytics
                os.system(f"pip install {args.model}")
                model_path = args.model
    
    # Start video processing
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

# Entry point
if __name__ == "__main__":
    main()