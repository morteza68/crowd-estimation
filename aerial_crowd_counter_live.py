"""
Advanced Crowd Detection with Live Stream Support
===============================================
This program detects people in various input sources including live streams.
Specialized for aerial imagery with enhanced accuracy and visualization.

Features:
- Multiple input sources: camera, video, image, directory, live streams (RTSP, HTTP, etc.)
- Real-time processing with GPU acceleration
- Specialized aerial detection with scale-aware processing
- Live stream buffering and reconnection capabilities

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
import glob
import json
import threading
import queue
import re
from pathlib import Path
from threading import Thread
from queue import Queue
from datetime import datetime
import urllib.parse

# Add tqdm for progress display
from tqdm import tqdm

# Set up input arguments with aerial-specific defaults
def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Crowd Detection with Deep Learning')
    
    # Input sources
    parser.add_argument('--source', type=str, default='0', 
                      help='Input source: camera number, video path, image path, directory path, or stream URL')
    parser.add_argument('--source-type', type=str, default='auto', 
                      choices=['auto', 'camera', 'video', 'image', 'dir', 'stream'],
                      help='Force source type instead of auto-detection')
    
    # Stream-specific options
    parser.add_argument('--stream-buffer', type=int, default=60, 
                      help='Buffer size for stream (in frames)')
    parser.add_argument('--reconnect-attempts', type=int, default=5, 
                      help='Number of reconnection attempts for streams')
    parser.add_argument('--reconnect-delay', type=int, default=5, 
                      help='Delay between reconnection attempts (in seconds)')
    parser.add_argument('--stream-timeout', type=int, default=30, 
                      help='Stream connection timeout (in seconds)')
    
    # Model settings - specialized for aerial imagery
    parser.add_argument('--model', type=str, default='yolov8x', 
                      help='Model: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x')
    parser.add_argument('--conf-thres', type=float, default=0.3, 
                      help='Confidence threshold (lower for aerial views)')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                      help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='', 
                      help='Device (cpu, cuda:0, 0)')
    
    # Aerial-specific options
    parser.add_argument('--img-size', type=int, default=1280, 
                      help='Inference size (pixels) - larger for aerial views')
    parser.add_argument('--augment', action='store_true',
                      help='Augment inference for better detection')
    parser.add_argument('--scale-persons', action='store_true',
                      help='Apply scale-aware person detection')
    parser.add_argument('--density-map', action='store_true',
                      help='Generate crowd density map')
    
    # Display and save options
    parser.add_argument('--view-img', action='store_true', 
                      help='Display results')
    parser.add_argument('--save-img', action='store_true', 
                      help='Save processed images/videos')
    parser.add_argument('--save-results', action='store_true', 
                      help='Save detection results as JSON')
    parser.add_argument('--output-dir', type=str, default='output', 
                      help='Directory to save results')
    
    # Processing options
    parser.add_argument('--max-frames', type=int, default=0, 
                      help='Maximum frames to process (0 = unlimited)')
    parser.add_argument('--batch-size', type=int, default=1, 
                      help='Batch size for image directory processing')
    parser.add_argument('--fps-limit', type=int, default=0, 
                      help='Limit processing FPS (0 = no limit)')
    
    # GitHub options
    parser.add_argument('--github-repo', type=str, default='ultralytics/assets', 
                      help='GitHub repository for model download')
    parser.add_argument('--github-branch', type=str, default='main', 
                      help='GitHub branch')
    
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

# Class for detecting input source type
class SourceDetector:
    @staticmethod
    def is_url(source):
        """Check if source is a URL"""
        try:
            result = urllib.parse.urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def is_stream_url(source):
        """Check if source is a streaming URL"""
        stream_protocols = ['rtsp://', 'rtmp://', 'http://', 'https://']
        stream_extensions = ['.m3u8', '.ts', '.mpd', '.flv']
        
        if any(source.startswith(p) for p in stream_protocols):
            return True
            
        if SourceDetector.is_url(source) and any(source.endswith(e) for e in stream_extensions):
            return True
            
        return False
    
    @staticmethod
    def detect_source_type(source, force_type='auto'):
        """Detect the type of input source"""
        if force_type != 'auto':
            return force_type
            
        # Check if it's a stream URL
        if SourceDetector.is_stream_url(source):
            return 'stream'
            
        # If source is a digit, it's a camera
        if source.isdigit():
            return 'camera'
            
        # Check if it's a file or directory
        if os.path.exists(source):
            # If it's a directory
            if os.path.isdir(source):
                return 'dir'
                
            # Get file extension
            ext = os.path.splitext(source)[1].lower()
            
            # Check if it's an image
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                return 'image'
                
            # Check if it's a video
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                return 'video'
                
        # Check if it's a URL but not specifically identified as stream
        if SourceDetector.is_url(source):
            return 'stream'
                
        # Default to camera if can't determine
        return 'camera'

# Class for aerial crowd density estimation
class CrowdDensityEstimator:
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size
        self.kernel = cv2.getGaussianKernel(kernel_size, -1) * cv2.getGaussianKernel(kernel_size, -1).T
        self.kernel = self.kernel / self.kernel.sum()
        
    def generate_density_map(self, detections, image_shape):
        """Generate density map from detections"""
        h, w = image_shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)
        
        # For each person detection, add a Gaussian blob
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            # Use center of bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Calculate size-adaptive sigma (larger boxes get larger blobs)
            box_width, box_height = x2 - x1, y2 - y1
            box_size = max(box_width, box_height)
            sigma = max(3, box_size / 10)  # Adaptive sigma based on person size
            kernel_size = int(sigma * 6)  # 3-sigma rule
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
                
            if kernel_size > 3:  # Only add if kernel is reasonable size
                kernel = cv2.getGaussianKernel(kernel_size, sigma) * cv2.getGaussianKernel(kernel_size, sigma).T
                kernel = kernel / kernel.sum()  # Normalize to ensure sum is 1
                
                # Calculate region to add the kernel
                x_start = max(0, center_x - kernel_size // 2)
                y_start = max(0, center_y - kernel_size // 2)
                x_end = min(w, center_x + kernel_size // 2 + 1)
                y_end = min(h, center_y + kernel_size // 2 + 1)
                
                # Calculate kernel region
                k_x_start = max(0, kernel_size // 2 - center_x)
                k_y_start = max(0, kernel_size // 2 - center_y)
                k_x_end = kernel_size - max(0, center_x + kernel_size // 2 + 1 - w)
                k_y_end = kernel_size - max(0, center_y + kernel_size // 2 + 1 - h)
                
                # Add kernel to density map
                kernel_region = kernel[k_y_start:k_y_end, k_x_start:k_x_end]
                density_map[y_start:y_end, x_start:x_end] += kernel_region
        
        return density_map

# Base processor class with aerial-specific enhancements
class BaseProcessor:
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output'):
        self.source = source
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.augment = augment
        self.scale_persons = scale_persons
        self.density_map = density_map
        self.view_img = view_img
        self.save_img = save_img
        self.save_results = save_results
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directory
        if self.save_img or self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self.load_model(model, device)
        
        # Create density estimator if needed
        if self.density_map:
            self.density_estimator = CrowdDensityEstimator()
        
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
    
    def detect_people(self, image):
        """Detect people in image with aerial-specific settings"""
        # Process image with model
        results = self.model(
            image, 
            classes=0,  # Only person class
            conf=self.conf_thres,
            iou=self.iou_thres,
            augment=self.augment,
            imgsz=self.img_size
        )
        
        return results
            
    def apply_post_processing(self, detections, image_shape):
        """Apply post-processing to improve aerial detections"""
        if not self.scale_persons:
            return detections
        
        # Get image dimensions
        h, w = image_shape[:2]
        min_dim = min(h, w)
        
        # Filter detections based on size and position
        filtered_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            # Calculate aspect ratio and size relative to image
            aspect_ratio = width / height if height > 0 else 0
            rel_size = (width * height) / (h * w)
            
            # People in aerial views typically have certain characteristics
            # 1. Aspect ratio close to 1:1 or slightly taller (0.5-1.5)
            # 2. Not too large relative to image size
            # 3. Not too small to be noise
            if (0.3 < aspect_ratio < 2.0 and  # Reasonable aspect ratio for aerial view
                rel_size < 0.1 and  # Not too large
                min(width, height) > min_dim * 0.01):  # Not too small
                filtered_detections.append(det)
        
        return filtered_detections
    
    def save_detection_results(self, results, filename, raw_image=None):
        """Save detection results to JSON file with aerial-specific information"""
        if not self.save_results:
            return
            
        output_path = self.output_dir / f"{filename}_detections.json"
        
        # Extract detection data
        detections = []
        total_people = 0
        
        for i, result in enumerate(results):
            boxes = result.boxes
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                if class_id == 0:  # Person class
                    total_people += 1
                    
                    detections.append({
                        "id": total_people,
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
        
        # Apply scale-aware post-processing if enabled
        if self.scale_persons and raw_image is not None:
            detections = self.apply_post_processing(detections, raw_image.shape)
            total_people = len(detections)
        
        # Create JSON data
        json_data = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "confidence_threshold": self.conf_thres,
                "iou_threshold": self.iou_thres,
                "image_size": self.img_size,
                "augmented": self.augment,
                "scale_aware": self.scale_persons
            },
            "total_detections": total_people,
            "people_count": total_people,
            "detections": detections
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Detection results saved to {output_path}")
        return detections

# Stream Buffer class for handling network streams with reconnection capability
class StreamBuffer:
    def __init__(self, stream_url, buffer_size=60, reconnect_attempts=5, reconnect_delay=5, timeout=30):
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout
        
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.status_queue = queue.Queue()
        self.running = False
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self.connection_ok = False
        
        # Stream properties
        self.width = 0
        self.height = 0
        self.stream_fps = 0
        
    def start(self):
        """Start the stream buffer in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._buffer_frames, daemon=True)
        self.thread.start()
        
        # Wait for initial connection
        try:
            status = self.status_queue.get(timeout=self.timeout)
            if status["status"] == "connected":
                self.connection_ok = True
                self.width = status["width"]
                self.height = status["height"]
                self.stream_fps = status["fps"]
                print(f"Connected to stream: {self.width}x{self.height} at {self.stream_fps} FPS")
                return True
            else:
                print(f"Failed to connect to stream: {status['message']}")
                self.stop()
                return False
        except queue.Empty:
            print(f"Timeout connecting to stream after {self.timeout} seconds")
            self.stop()
            return False
            
    def _buffer_frames(self):
        """Buffer frames from stream in a separate thread"""
        attempt = 0
        
        while self.running and attempt < self.reconnect_attempts:
            try:
                # Open the stream
                cap = cv2.VideoCapture(self.stream_url)
                if not cap.isOpened():
                    attempt += 1
                    self.status_queue.put({
                        "status": "error",
                        "message": f"Failed to open stream (attempt {attempt}/{self.reconnect_attempts})"
                    })
                    time.sleep(self.reconnect_delay)
                    continue
                
                # Get stream properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Reset attempt counter on successful connection
                attempt = 0
                self.status_queue.put({
                    "status": "connected",
                    "width": width,
                    "height": height,
                    "fps": fps
                })
                
                # Start reading frames
                frame_time = time.time()
                frame_count_for_fps = 0
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        # Stream ended or error
                        break
                    
                    # Calculate FPS
                    current_time = time.time()
                    frame_count_for_fps += 1
                    if current_time - frame_time >= 1.0:
                        self.fps = frame_count_for_fps / (current_time - frame_time)
                        frame_time = current_time
                        frame_count_for_fps = 0
                    
                    # Add frame to buffer, drop oldest if full
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_buffer.put((frame, self.frame_count))
                    self.frame_count += 1
                    self.last_frame_time = current_time
                
                # Stream ended normally
                cap.release()
                
                if self.running:
                    # If we're still running, this was an unexpected disconnection
                    print("Stream ended unexpectedly. Attempting to reconnect...")
                    time.sleep(self.reconnect_delay)
            
            except Exception as e:
                # Handle connection errors
                print(f"Stream error: {e}")
                attempt += 1
                if attempt < self.reconnect_attempts:
                    print(f"Reconnecting in {self.reconnect_delay} seconds (attempt {attempt}/{self.reconnect_attempts})...")
                    time.sleep(self.reconnect_delay)
                else:
                    self.status_queue.put({
                        "status": "error",
                        "message": f"Failed after {self.reconnect_attempts} attempts: {str(e)}"
                    })
                    break
        
        print("Stream buffer thread exiting")
        
    def read(self):
        """Read the next frame from the buffer"""
        if not self.running:
            return False, None
            
        try:
            frame, index = self.frame_buffer.get(timeout=self.timeout)
            self.current_frame = frame
            return True, frame
        except queue.Empty:
            # No frames received within timeout
            time_since_last = time.time() - self.last_frame_time
            if time_since_last > self.timeout and self.last_frame_time > 0:
                print(f"Stream timeout: No frames received for {time_since_last:.1f} seconds")
                return False, None
            return False, self.current_frame  # Return the last valid frame
            
    def stop(self):
        """Stop the stream buffer"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
        # Clear buffers
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break

# Live Stream Processor for handling network video streams
class LiveStreamProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 max_frames=0, buffer_size=60, reconnect_attempts=5, reconnect_delay=5, 
                 stream_timeout=30, fps_limit=0):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir)
                        
        self.max_frames = max_frames
        self.fps_limit = fps_limit
        
        # Create stream buffer
        self.stream_buffer = StreamBuffer(
            stream_url=source,
            buffer_size=buffer_size,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay=reconnect_delay,
            timeout=stream_timeout
        )
        
        # Initialize video writer
        self.writer = None
        
        # Get timestamp for output files
        self.timestamp = int(time.time())
        
        # Status tracking
        self.frame_count = 0
        self.total_people = 0
        self.process_times = []
        
    def process(self):
        """Process the live stream"""
        print(f"Connecting to stream: {self.source}")
        
        # Start the stream buffer
        if not self.stream_buffer.start():
            print("Failed to start stream processing")
            return 0, 0
        
        # Set up video writer if needed
        if self.save_img:
            # Generate a safe filename from the URL
            safe_filename = re.sub(r'[^\w]', '_', self.source)
            safe_filename = safe_filename[:50]  # Limit length
            output_filename = f"stream_{safe_filename}_{self.timestamp}.mp4"
            output_path = str(self.output_dir / output_filename)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                max(self.stream_buffer.stream_fps, 25),  # Use stream FPS or default to 25
                (self.stream_buffer.width, self.stream_buffer.height)
            )
            print(f"Recording stream to: {output_path}")
        
        # Process frames
        start_time = time.time()
        last_frame_time = time.time()
        processing = True
        
        print(f"Processing stream with resolution: {self.stream_buffer.width}x{self.stream_buffer.height}")
        
        try:
            while processing:
                # Control processing rate if needed
                if self.fps_limit > 0:
                    elapsed = time.time() - last_frame_time
                    target_time = 1.0 / self.fps_limit
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                
                # Read frame from buffer
                ret, frame = self.stream_buffer.read()
                if not ret:
                    # Check if we're still connected
                    if not self.stream_buffer.connection_ok:
                        print("Lost connection to stream")
                        break
                    # No new frame yet, continue
                    time.sleep(0.01)
                    continue
                
                last_frame_time = time.time()
                self.frame_count += 1
                
                # Process frame
                process_start = time.time()
                
                # Detect people
                results = self.detect_people(frame)
                
                # Processing time
                process_time = time.time() - process_start
                self.process_times.append(process_time)
                
                # Save detection results periodically
                if self.save_results and self.frame_count % 30 == 0:  # Every 30 frames
                    detections = self.save_detection_results(
                        results, 
                        f"stream_{self.timestamp}_{self.frame_count:06d}", 
                        frame
                    )
                else:
                    # Extract detections anyway for counting
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            
                            if class_id == 0:  # Person class
                                detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": confidence,
                                    "width": x2 - x1,
                                    "height": y2 - y1
                                })
                
                # Apply scale-aware post-processing if enabled
                if self.scale_persons:
                    detections = self.apply_post_processing(detections, frame.shape)
                
                # Count people
                people_count = len(detections)
                self.total_people += people_count
                
                # Prepare visualization
                result_img = results[0].plot()
                
                # Generate density map if requested
                if self.density_map and len(detections) > 0:
                    density_map = self.density_estimator.generate_density_map(detections, frame.shape)
                    
                    # Normalize and colorize density map
                    density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
                    
                    # Blend with original image
                    alpha = 0.5
                    result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
                
                # Calculate average FPS (both stream FPS and processing FPS)
                stream_fps = self.stream_buffer.fps
                avg_process_time = sum(self.process_times[-100:]) / min(len(self.process_times), 100)
                process_fps = 1 / avg_process_time if avg_process_time > 0 else 0
                
                # Add information to image
                cv2.putText(result_img, f"People: {people_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_img, f"Process FPS: {process_fps:.1f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_img, f"Stream FPS: {stream_fps:.1f}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add exit instructions and stream info
                cv2.putText(result_img, "Press 'q' to quit", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_img, f"Stream: {self.source.split('/')[-1]}", 
                            (10, self.stream_buffer.height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Save processed frame if needed
                if self.save_img and self.writer is not None:
                    self.writer.write(result_img)
                
                # Display frame if needed
                if self.view_img:
                    # Resize if too large for display
                    display_img = result_img.copy()
                    screen_res = 1920, 1080  # Assumed max screen resolution
                    scale_width = screen_res[0] / display_img.shape[1]
                    scale_height = screen_res[1] / display_img.shape[0]
                    scale = min(scale_width, scale_height, 1)  # Don't upscale
                    
                    if scale < 1:
                        new_width = int(display_img.shape[1] * scale)
                        new_height = int(display_img.shape[0] * scale)
                        display_img = cv2.resize(display_img, (new_width, new_height))
                    
                    cv2.imshow("Live Stream Analysis - Press 'q' to quit", display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user")
                        break
                
                # Check if we've processed enough frames
                if self.max_frames > 0 and self.frame_count >= self.max_frames:
                    print(f"Reached maximum frame count ({self.max_frames})")
                    break
                
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        except Exception as e:
            print(f"Error during stream processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up
            self.stream_buffer.stop()
            
            if self.save_img and self.writer is not None:
                self.writer.release()
                
            cv2.destroyAllWindows()
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            avg_people = self.total_people / self.frame_count if self.frame_count > 0 else 0
            avg_process_time = sum(self.process_times) / len(self.process_times) if self.process_times else 0
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\nStream Processing Complete:")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total people detected: {self.total_people}")
            print(f"Average people per frame: {avg_people:.1f}")
            print(f"Average processing time: {avg_process_time:.3f}s per frame")
            print(f"Average processing FPS: {1/avg_process_time:.1f}" if avg_process_time > 0 else "N/A")
            print(f"Total elapsed time: {elapsed_time:.1f}s")
            
            return self.total_people, avg_process_time

# Class for processing aerial images
class AerialImageProcessor(BaseProcessor):
    def __init__(self, source, model, device, **kwargs):
        super().__init__(source, model, device, **kwargs)
        
        # Check if source exists
        if not os.path.isfile(source):
            raise ValueError(f"Image file not found: {source}")
            
    def process(self):
        """Process a single aerial image"""
        print(f"Processing aerial image: {self.source}")
        
        # Read image
        img = cv2.imread(self.source)
        if img is None:
            raise ValueError(f"Failed to read image: {self.source}")
            
        # Get image name without extension
        img_name = os.path.splitext(os.path.basename(self.source))[0]
        
        # Start timing
        start_time = time.time()
        
        # Process image with model
        results = self.detect_people(img)
        
        # Processing time
        process_time = time.time() - start_time
        
        # Save detection results if needed
        detections = self.save_detection_results(results, img_name, img)
        
        # Get processed people count (after potential filtering)
        if self.scale_persons and detections is not None:
            people_count = len(detections)
        else:
            people_count = len(results[0].boxes)
        
        # Prepare visualization
        result_img = results[0].plot()
        
        # Generate density map if requested
        if self.density_map and detections is not None:
            density_map = self.density_estimator.generate_density_map(detections, img.shape)
            
            # Normalize and colorize density map
            density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
            
            # Blend with original image
            alpha = 0.5
            result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
            
            # Save density map
            if self.save_img:
                density_path = str(self.output_dir / f"{img_name}_density.jpg")
                cv2.imwrite(density_path, density_colored)
                print(f"Density map saved to: {density_path}")
        
        # Add information to image
        cv2.putText(result_img, f"People: {people_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_img, f"Process time: {process_time:.3f}s", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.scale_persons:
            cv2.putText(result_img, "Scale-aware detection active", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save processed image if needed
        if self.save_img:
            output_path = str(self.output_dir / f"{img_name}_processed.jpg")
            cv2.imwrite(output_path, result_img)
            print(f"Processed image saved to: {output_path}")
        
        # Display image if needed
        if self.view_img:
            # Resize if too large for display
            display_img = result_img.copy()
            screen_res = 1920, 1080  # Assumed max screen resolution
            scale_width = screen_res[0] / display_img.shape[1]
            scale_height = screen_res[1] / display_img.shape[0]
            scale = min(scale_width, scale_height, 1)  # Don't upscale
            
            if scale < 1:
                new_width = int(display_img.shape[1] * scale)
                new_height = int(display_img.shape[0] * scale)
                display_img = cv2.resize(display_img, (new_width, new_height))
            
            cv2.imshow("Aerial Crowd Detection - Press any key to continue", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"Image processing completed: {people_count} people detected in {process_time:.3f}s")
        
        return people_count, process_time

# Class for processing multiple aerial images in a directory
class AerialDirectoryProcessor(BaseProcessor):
    def __init__(self, source, model, device, batch_size=1, max_frames=0, **kwargs):
        super().__init__(source, model, device, **kwargs)
        
        self.batch_size = batch_size
        self.max_frames = max_frames
        
        # Check if source is a directory
        if not os.path.isdir(source):
            raise ValueError(f"Directory not found: {source}")
            
        # Get all image files in directory
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
            self.image_files.extend(glob.glob(os.path.join(source, ext)))
            self.image_files.extend(glob.glob(os.path.join(source, ext.upper())))
        
        # Sort image files
        self.image_files.sort()
        
        # Apply max_frames limit if needed
        if self.max_frames > 0 and self.max_frames < len(self.image_files):
            self.image_files = self.image_files[:self.max_frames]
            
        print(f"Found {len(self.image_files)} aerial images in {source}")
        
    def process(self):
        """Process all aerial images in the directory"""
        print(f"Processing {len(self.image_files)} aerial images...")
        
        total_people = 0
        total_time = 0
        
        # Create progress bar
        with tqdm(total=len(self.image_files), desc="Processing aerial images") as pbar:
            for i, img_path in enumerate(self.image_files):
                # Get image name without extension
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Failed to read image: {img_path}")
                    continue
                
                # Start timing
                start_time = time.time()
                
                # Process image with model
                results = self.detect_people(img)
                
                # Processing time
                process_time = time.time() - start_time
                
                # Save detection results if needed
                detections = self.save_detection_results(results, img_name, img)
                
                # Get processed people count (after potential filtering)
                if self.scale_persons and detections is not None:
                    people_count = len(detections)
                else:
                    people_count = len(results[0].boxes)
                
                # Update totals
                total_people += people_count
                total_time += process_time
                
                # Prepare visualization
                result_img = results[0].plot()
                
                # Generate density map if requested
                if self.density_map and detections is not None:
                    density_map = self.density_estimator.generate_density_map(detections, img.shape)
                    
                    # Normalize and colorize density map
                    density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
                    
                    # Blend with original image
                    alpha = 0.5
                    result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
                    
                    # Save density map
                    if self.save_img:
                        density_path = str(self.output_dir / f"{img_name}_density.jpg")
                        cv2.imwrite(density_path, density_colored)
                
                # Add information to image
                cv2.putText(result_img, f"People: {people_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_img, f"Process time: {process_time:.3f}s", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if self.scale_persons:
                    cv2.putText(result_img, "Scale-aware detection active", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save processed image if needed
                if self.save_img:
                    output_path = str(self.output_dir / f"{img_name}_processed.jpg")
                    cv2.imwrite(output_path, result_img)
                
                # Display image if needed
                if self.view_img:
                    # Resize if too large for display
                    display_img = result_img.copy()
                    screen_res = 1920, 1080  # Assumed max screen resolution
                    scale_width = screen_res[0] / display_img.shape[1]
                    scale_height = screen_res[1] / display_img.shape[0]
                    scale = min(scale_width, scale_height, 1)  # Don't upscale
                    
                    if scale < 1:
                        new_width = int(display_img.shape[1] * scale)
                        new_height = int(display_img.shape[0] * scale)
                        display_img = cv2.resize(display_img, (new_width, new_height))
                    
                    cv2.imshow("Aerial Crowd Detection - Press 'q' to quit, any other key to continue", display_img)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user")
                        break
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"people": people_count, "time": f"{process_time:.3f}s"})
        
        # Close any open windows
        cv2.destroyAllWindows()
        
        # Calculate averages
        avg_people = total_people / len(self.image_files) if self.image_files else 0
        avg_time = total_time / len(self.image_files) if self.image_files else 0
        
        print("\nAerial Directory Processing Complete:")
        print(f"Total images processed: {len(self.image_files)}")
        print(f"Total people detected: {total_people}")
        print(f"Average people per image: {avg_people:.1f}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        return total_people, total_time

# Check and install requirements with progress display
def check_requirements():
    required_packages = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'requests': 'requests'
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
    print("ADVANCED CROWD DETECTION SYSTEM")
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
    
    # Detect source type
    source_type = SourceDetector.detect_source_type(args.source, args.source_type)
    print(f"Detected source type: {source_type}")
    
    # Common arguments for processors
    processor_args = {
        'model': model_path,
        'device': device,
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'img_size': args.img_size,
        'augment': args.augment,
        'scale_persons': args.scale_persons,
        'density_map': args.density_map,
        'view_img': args.view_img,
        'save_img': args.save_img,
        'save_results': args.save_results,
        'output_dir': args.output_dir
    }
    
    # Process based on source type
    if source_type == 'image':
        processor = AerialImageProcessor(
            source=args.source,
            **processor_args
        )
        processor.process()
        
    elif source_type == 'dir':
        processor = AerialDirectoryProcessor(
            source=args.source,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            **processor_args
        )
        processor.process()
        
    elif source_type == 'stream':
        processor = LiveStreamProcessor(
            source=args.source,
            buffer_size=args.stream_buffer,
            reconnect_attempts=args.reconnect_attempts,
            reconnect_delay=args.reconnect_delay,
            stream_timeout=args.stream_timeout,
            max_frames=args.max_frames,
            fps_limit=args.fps_limit,
            **processor_args
        )
        processor.process()
        
    else:
        print(f"Source type '{source_type}' not yet implemented for aerial analysis")
        print(f"Please use image, directory, or stream input")

# Entry point
if __name__ == "__main__":
    main()