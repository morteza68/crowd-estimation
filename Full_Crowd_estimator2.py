"""
Advanced Crowd Detection System
==============================
Comprehensive crowd detection for multiple input sources including webcams, videos, images,
directories, and live streams with enhanced false positive reduction.

Features:
- Multiple input sources: webcam, video, image, directory, live streams (RTSP, HTTP)
- False positive filtering to prevent picture frames and static objects from being counted
- Real-time processing with GPU acceleration
- Enhanced live stream processing with temporal smoothing and adaptive confidence
- Density map generation for crowd visualization
- Object tracking for consistent detection

Author: Team Graphics
Date: 2025-03-24
Version: 2.1
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
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import urllib.parse
from collections import deque

# Add tqdm for progress display
try:
    from tqdm import tqdm
except ImportError:
    # Define minimal tqdm fallback
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Set up input arguments with comprehensive options
def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Crowd Detection System')
    
    # Input sources
    parser.add_argument('--source', type=str, default='0', 
                      help='Input source: camera number, video path, image path, directory path, or stream URL')
    parser.add_argument('--source-type', type=str, default='auto', 
                      choices=['auto', 'camera', 'video', 'image', 'dir', 'stream'],
                      help='Force source type instead of auto-detection')
    
    # Stream-specific options
    parser.add_argument('--stream-buffer', type=int, default=90, 
                      help='Buffer size for stream (in frames)')
    parser.add_argument('--reconnect-attempts', type=int, default=10, 
                      help='Number of reconnection attempts for streams')
    parser.add_argument('--reconnect-delay', type=int, default=3, 
                      help='Delay between reconnection attempts (in seconds)')
    parser.add_argument('--stream-timeout', type=int, default=30, 
                      help='Stream connection timeout (in seconds)')
    parser.add_argument('--stream-retry-interval', type=int, default=5,
                      help='Seconds between reconnection attempts')
    parser.add_argument('--stream-jitter-buffer', type=int, default=15,
                      help='Jitter buffer size for smoothing stream processing')
    
    # Enhanced processing
    parser.add_argument('--temporal-smoothing', type=int, default=5,
                      help='Number of frames for temporal smoothing (0 to disable)')
    parser.add_argument('--preprocess-stream', action='store_true',
                      help='Apply preprocessing to enhance stream quality')
    parser.add_argument('--adaptive-conf', action='store_true',
                      help='Use adaptive confidence thresholds based on image quality')
    parser.add_argument('--motion-based', action='store_true',
                      help='Use motion detection to optimize processing')
    parser.add_argument('--track-objects', action='store_true',
                      help='Track objects across frames for better consistency')
    
    # False positive reduction
    parser.add_argument('--filter-static', action='store_true',
                      help='Filter out static objects like picture frames')
    parser.add_argument('--static-threshold', type=int, default=20,
                      help='Frames an object must be static to be filtered (with --filter-static)')
    parser.add_argument('--min-person-height', type=int, default=50,
                      help='Minimum height in pixels for a person detection')
    parser.add_argument('--aspect-ratio-check', action='store_true',
                      help='Use aspect ratio checking for better person detection')
    
    # Model settings
    parser.add_argument('--model', type=str, default='yolov8x', 
                      help='Model: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x')
    parser.add_argument('--conf-thres', type=float, default=0.3, 
                      help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                      help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='', 
                      help='Device (cpu, cuda:0, 0)')
    
    # Aerial-specific options
    parser.add_argument('--img-size', type=int, default=1280, 
                      help='Inference size (pixels)')
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

# Image Preprocessing for Stream Enhancement
class StreamPreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def enhance_image(self, image):
        """Apply image enhancement techniques for better detection"""
        try:
            # Check if image is valid
            if image is None or image.size == 0:
                return None
                
            # Create a copy to avoid modifying original
            enhanced = image.copy()
            
            # Convert to YUV color space
            yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
            
            # Apply CLAHE to the Y channel (luminance)
            yuv[:,:,0] = self.clahe.apply(yuv[:,:,0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Apply slight noise reduction
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in image enhancement: {e}")
            return image  # Return original image if enhancement fails
    
    def detect_blur(self, image):
        """Detect if image is blurry"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 100  # Default to non-blurry if detection fails
    
    def detect_lighting(self, image):
        """Detect lighting conditions"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            brightness = hsv[:,:,2].mean()
            return brightness
        except:
            return 127  # Default to medium brightness

# Advanced Object Tracker with false positive filtering
class ObjectTracker:
    def __init__(self, max_disappeared=10, max_distance=50, 
                 filter_static=True, static_threshold=20, 
                 min_person_height=50, aspect_ratio_check=True):
        self.next_object_id = 0
        self.objects = {}  # Dictionary of tracked objects
        self.disappeared = {}  # Number of consecutive frames object has been lost
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # False positive filtering
        self.filter_static = filter_static
        self.static_threshold = static_threshold
        self.min_person_height = min_person_height
        self.aspect_ratio_check = aspect_ratio_check
        
        # For tracking static objects (potential picture frames)
        self.static_count = {}  # Count of frames object hasn't moved significantly
        self.static_objects = set()  # IDs of objects classified as static (non-person)
        
        # For movement tracking
        self.movement_threshold = 3  # Pixels of movement to be considered non-static
        
    def register(self, centroid, bbox, confidence):
        """Register a new object"""
        # Apply initial filtering
        if not self._passes_filters(bbox):
            return None  # Don't register objects that don't pass filters
            
        object_id = self.next_object_id
        self.objects[object_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "confidence": confidence,
            "history": [centroid],
            "size_history": [bbox]  # Track bbox size over time
        }
        self.disappeared[object_id] = 0
        self.static_count[object_id] = 0
        self.next_object_id += 1
        
        return object_id
        
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.static_count:
            del self.static_count[object_id]
        if object_id in self.static_objects:
            self.static_objects.remove(object_id)
        
    def _passes_filters(self, bbox):
        """Check if detection passes basic filters"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Height filter - reject very small detections
        if height < self.min_person_height:
            return False
            
        # Aspect ratio check for persons (if enabled)
        if self.aspect_ratio_check:
            aspect_ratio = height / width if width > 0 else 0
            # Typical human aspect ratios are between 1.5 and 4.0
            if not (1.2 < aspect_ratio < 4.5):
                return False
                
        return True
        
    def _update_static_status(self, object_id):
        """Update static status of an object based on movement history"""
        if not self.filter_static or len(self.objects[object_id]["history"]) < 5:
            return
            
        # Get recent movement
        recent_positions = self.objects[object_id]["history"][-5:]
        max_movement = 0
        
        # Calculate maximum movement in recent history
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            movement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            max_movement = max(max_movement, movement)
        
        # Update static counter based on movement
        if max_movement < self.movement_threshold:
            self.static_count[object_id] += 1
        else:
            # Reset static counter if significant movement detected
            self.static_count[object_id] = max(0, self.static_count[object_id] - 2)
        
        # Classify as static if counter exceeds threshold
        if self.static_count[object_id] > self.static_threshold:
            self.static_objects.add(object_id)
        elif object_id in self.static_objects and self.static_count[object_id] < self.static_threshold // 2:
            # Remove from static objects if enough movement detected
            self.static_objects.remove(object_id)
    
    def is_static_object(self, object_id):
        """Check if object is classified as static (like picture frame)"""
        return object_id in self.static_objects
            
    def update(self, detections):
        """Update tracked objects with new detections"""
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Filter detections before processing
        filtered_detections = []
        for detection in detections:
            bbox = detection["bbox"]
            if self._passes_filters(bbox):
                filtered_detections.append(detection)
        
        # If no filtered detections, handle like no detections
        if len(filtered_detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # If no existing objects, register all filtered detections
        if len(self.objects) == 0:
            for detection in filtered_detections:
                bbox = detection["bbox"]
                centroid = (
                    int((bbox[0] + bbox[2]) / 2), 
                    int((bbox[1] + bbox[3]) / 2)
                )
                self.register(centroid, bbox, detection["confidence"])
            return self.objects
        
        # Match existing objects with new detections
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[i]["centroid"] for i in object_ids]
        
        # Calculate centroids for new detections
        detection_centroids = []
        for detection in filtered_detections:
            bbox = detection["bbox"]
            centroid = (
                int((bbox[0] + bbox[2]) / 2), 
                int((bbox[1] + bbox[3]) / 2)
            )
            detection_centroids.append({
                "centroid": centroid,
                "bbox": bbox,
                "confidence": detection["confidence"]
            })
        
        # Calculate distances between all objects and detections
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det in enumerate(detection_centroids):
                det_centroid = det["centroid"]
                distances[i, j] = np.sqrt(
                    (obj_centroid[0] - det_centroid[0]) ** 2 +
                    (obj_centroid[1] - det_centroid[1]) ** 2
                )
        
        # Find best matches using Hungarian algorithm if scipy is available
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(distances)
        except ImportError:
            # Fallback to greedy matching
            row_indices, col_indices = [], []
            for i in range(min(distances.shape)):
                j = np.argmin(distances[i, :])
                if distances[i, j] <= self.max_distance:
                    row_indices.append(i)
                    col_indices.append(j)
        
        # Update matched objects
        used_rows = set()
        used_cols = set()
        
        for row, col in zip(row_indices, col_indices):
            # Only update if distance is within threshold
            if distances[row, col] > self.max_distance:
                continue
                
            object_id = object_ids[row]
            det = detection_centroids[col]
            
            self.objects[object_id]["centroid"] = det["centroid"]
            self.objects[object_id]["bbox"] = det["bbox"]
            self.objects[object_id]["confidence"] = det["confidence"]
            self.objects[object_id]["history"].append(det["centroid"])
            self.objects[object_id]["size_history"].append(det["bbox"])
            
            # Limit history length
            if len(self.objects[object_id]["history"]) > 60:
                self.objects[object_id]["history"] = self.objects[object_id]["history"][-60:]
                self.objects[object_id]["size_history"] = self.objects[object_id]["size_history"][-60:]
                
            self.disappeared[object_id] = 0
            
            # Update static status for false positive filtering
            self._update_static_status(object_id)
            
            used_rows.add(row)
            used_cols.add(col)
        
        # Check for unmatched rows (disappeared objects)
        for row in range(len(object_centroids)):
            if row in used_rows:
                continue
                
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # Register new detections
        for col in range(len(detection_centroids)):
            if col in used_cols:
                continue
                
            det = detection_centroids[col]
            self.register(det["centroid"], det["bbox"], det["confidence"])
        
        return self.objects

# Enhanced Stream Buffer class for more reliable streaming
class EnhancedStreamBuffer:
    def __init__(self, stream_url, buffer_size=90, reconnect_attempts=10, 
                 reconnect_delay=3, timeout=30, retry_interval=5, jitter_buffer=15):
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.jitter_buffer = jitter_buffer
        
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
        
        # Enhanced properties
        self.last_frame_received = time.time()
        self.frame_history = deque(maxlen=jitter_buffer)
        self.connection_lock = Lock()
        self.reconnection_thread = None
        self.stream_health = 100  # Stream health percentage
        
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
    
    def _check_stream_health(self, cap):
        """Check if stream is healthy"""
        if not cap.isOpened():
            return False
            
        # Check time since last frame
        time_since_last = time.time() - self.last_frame_received
        if time_since_last > self.timeout and self.last_frame_received > 0:
            print(f"Stream timeout: No frames received for {time_since_last:.1f} seconds")
            return False
            
        return True
    
    def _reconnect_stream(self):
        """Reconnect to stream in a separate thread"""
        with self.connection_lock:
            if self.reconnection_thread is not None and self.reconnection_thread.is_alive():
                return  # Already reconnecting
                
            self.reconnection_thread = threading.Thread(
                target=self._reconnect_stream_worker,
                daemon=True
            )
            self.reconnection_thread.start()
    
    def _reconnect_stream_worker(self):
        """Worker thread for reconnection"""
        print(f"Attempting to reconnect to stream: {self.stream_url}")
        
        for attempt in range(self.reconnect_attempts):
            try:
                # Try to open the stream
                cap = cv2.VideoCapture(self.stream_url)
                if cap.isOpened():
                    # Get stream properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Check if we got valid properties
                    if width > 0 and height > 0:
                        self.width = width
                        self.height = height
                        self.stream_fps = fps if fps > 0 else 30  # Default to 30 if unknown
                        
                        print(f"Successfully reconnected to stream after {attempt+1} attempts")
                        self.connection_ok = True
                        self.stream_health = 100
                        
                        # Start reading frames again
                        self._read_frames_from_cap(cap)
                        return
                    
                cap.release()
            except Exception as e:
                print(f"Reconnection attempt {attempt+1} failed: {e}")
            
            # Wait before next attempt
            print(f"Waiting {self.retry_interval}s before next reconnection attempt...")
            time.sleep(self.retry_interval)
        
        print(f"Failed to reconnect after {self.reconnect_attempts} attempts")
        self.connection_ok = False
        self.stream_health = 0
    
    def _read_frames_from_cap(self, cap):
        """Read frames from an opened capture object"""
        frame_time = time.time()
        frame_count_for_fps = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"Stream read failed {consecutive_failures} times in a row")
                        break
                    time.sleep(0.01)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                self.last_frame_received = time.time()
                
                # Calculate FPS
                current_time = time.time()
                frame_count_for_fps += 1
                if current_time - frame_time >= 1.0:
                    self.fps = frame_count_for_fps / (current_time - frame_time)
                    frame_time = current_time
                    frame_count_for_fps = 0
                
                # Update stream health based on FPS
                expected_fps = max(self.stream_fps, 10)  # At least 10 FPS expected
                health_factor = min(self.fps / expected_fps, 1.0)
                self.stream_health = int(health_factor * 100)
                
                # Add frame to buffer, drop oldest if full
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add to frame history for jitter buffer
                self.frame_history.append(frame)
                
                # Only add to processing buffer if we have enough history frames
                if len(self.frame_history) >= min(5, self.jitter_buffer):
                    self.frame_buffer.put((frame.copy(), self.frame_count))
                    self.frame_count += 1
                
            except Exception as e:
                print(f"Error reading from stream: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
        
        # Stream ended or failed
        cap.release()
        self.connection_ok = False
        print("Stream connection lost")
        
        # Try to reconnect
        self._reconnect_stream()
            
    def _buffer_frames(self):
        """Buffer frames from stream in a separate thread"""
        try:
            # Open the stream
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                self.status_queue.put({
                    "status": "error",
                    "message": "Failed to open stream"
                })
                return
            
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Check if we got valid properties
            if width <= 0 or height <= 0:
                # Try to read a frame to get properties
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    # Add this frame to history
                    self.frame_history.append(frame)
                else:
                    self.status_queue.put({
                        "status": "error",
                        "message": "Could not determine stream dimensions"
                    })
                    cap.release()
                    return
            
            self.width = width
            self.height = height
            self.stream_fps = fps if fps > 0 else 30  # Default to 30 if unknown
            
            # Signal successful connection
            self.status_queue.put({
                "status": "connected",
                "width": width,
                "height": height,
                "fps": self.stream_fps
            })
            
            # Start reading frames
            self._read_frames_from_cap(cap)
            
        except Exception as e:
            self.status_queue.put({
                "status": "error",
                "message": f"Stream initialization error: {str(e)}"
            })
    
    def read(self):
        """Read the next frame from the buffer with enhanced error handling"""
        if not self.running:
            return False, None
            
        try:
            # Check stream health
            if self.stream_health < 30:
                print(f"Warning: Stream health is low ({self.stream_health}%)")
                if not self.connection_ok:
                    self._reconnect_stream()
            
            # Try to get a frame with timeout
            frame, index = self.frame_buffer.get(timeout=self.timeout)
            self.current_frame = frame
            return True, frame
            
        except queue.Empty:
            # No frames received within timeout
            time_since_last = time.time() - self.last_frame_received
            if time_since_last > self.timeout and self.last_frame_received > 0:
                print(f"Stream timeout: No frames received for {time_since_last:.1f} seconds")
                self._reconnect_stream()
                return False, None
                
            # Return last valid frame if available
            if self.current_frame is not None:
                return True, self.current_frame
            
            # Try to get a frame from history buffer
            if len(self.frame_history) > 0:
                self.current_frame = self.frame_history[-1].copy()
                return True, self.current_frame
                
            return False, None
            
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
                
        # Clear frame history
        self.frame_history.clear()

# Base processor class with common functionality
class BaseProcessor:
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 filter_static=True, static_threshold=20, min_person_height=50,
                 aspect_ratio_check=True):
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
        self.output_dir = Path(output_dir)
        
        # False positive filtering parameters
        self.filter_static = filter_static
        self.static_threshold = static_threshold
        self.min_person_height = min_person_height
        self.aspect_ratio_check = aspect_ratio_check
        
        # Make sure output directory exists
        if save_img or save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model: {model}")
        try:
            self.model = torch.hub.load('ultralytics/yolov8', model, pretrained=True, device=device)
            self.model.conf = conf_thres  # Set confidence threshold
            self.model.iou = iou_thres    # Set IoU threshold
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting alternative model loading method...")
            try:
                # Try to load using alternative method
                from ultralytics import YOLO
                self.model = YOLO(f"{model}.pt")
                print(f"Model loaded successfully using YOLO")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                sys.exit(1)
        
        # Initialize density estimator if needed
        if self.density_map:
            self.density_estimator = CrowdDensityEstimator()
    
    def apply_post_processing(self, detections, image_shape):
        """Apply post-processing to detections, including scale-awareness"""
        if not self.scale_persons:
            return detections
            
        # Get image dimensions
        img_h, img_w = image_shape[:2]
        processed_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Skip tiny detections (likely false positives)
            if box_height < self.min_person_height:
                continue
                
            # Check aspect ratio for person-like objects
            if self.aspect_ratio_check:
                aspect_ratio = box_height / box_width if box_width > 0 else 0
                if not (1.2 < aspect_ratio < 4.5):  # Typical human aspect ratios
                    continue
            
            # Adjust confidence based on position in image (for aerial views)
            position_factor = 1.0
            if y1 < img_h * 0.1:  # Objects near top of frame (far away in aerial)
                position_factor = 0.9  # Slightly reduce confidence
            
            # Adjust confidence based on size (larger people usually more reliable)
            size_factor = min(1.0, max(0.8, box_height / 100))
            
            # Apply adjustments
            adjusted_confidence = confidence * position_factor * size_factor
            
            # Add to processed detections if still above threshold
            if adjusted_confidence >= self.conf_thres:
                det["confidence"] = adjusted_confidence
                processed_detections.append(det)
                
        return processed_detections
    
    def save_detection_results(self, results, filename_base, original_image=None):
        """Save detection results as JSON and optionally the image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.output_dir / f"{filename_base}_{timestamp}.json"
        
        # Extract detection data
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
                        "class": "person"
                    })
        
        # Create result data
        result_data = {
            "timestamp": timestamp,
            "source": str(self.source),
            "total_detections": len(detections),
            "detections": detections
        }
        
        # Save JSON
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Save image if provided
        if original_image is not None and self.save_img:
            img_path = self.output_dir / f"{filename_base}_{timestamp}.jpg"
            cv2.imwrite(str(img_path), original_image)
        
        return result_path

# Webcam Processor with enhanced false positive filtering
class WebcamProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 max_frames=0, fps_limit=0, temporal_smoothing=5, preprocess_stream=True,
                 filter_static=True, static_threshold=20, min_person_height=50, 
                 aspect_ratio_check=True, track_objects=True):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir,
                        filter_static, static_threshold, min_person_height,
                        aspect_ratio_check)
        
        self.max_frames = max_frames
        self.fps_limit = fps_limit
        self.temporal_smoothing = temporal_smoothing
        self.preprocess_stream = preprocess_stream
        self.track_objects = track_objects
        
        # Create video capture
        try:
            # If source is a string digit, convert to int
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open webcam {source}")
                
            # Set properties for webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Get actual properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # Default FPS if not detected
                
            print(f"Connected to webcam: {self.width}x{self.height} at {self.fps} FPS")
        except Exception as e:
            print(f"Error opening webcam: {e}")
            raise
        
        # Initialize video writer if needed
        self.writer = None
        if self.save_img:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = int(time.time())
            output_filename = f"webcam_{timestamp}.mp4"
            output_path = str(self.output_dir / output_filename)
            
            self.writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                self.fps,
                (self.width, self.height)
            )
            print(f"Recording webcam to: {output_path}")
        
        # For temporal smoothing
        self.people_count_history = deque(maxlen=max(30, temporal_smoothing))
        
        # Initialize object tracker if needed
        if self.track_objects:
            self.tracker = ObjectTracker(
                filter_static=filter_static,
                static_threshold=static_threshold,
                min_person_height=min_person_height,
                aspect_ratio_check=aspect_ratio_check
            )
        
        # Initialize preprocessor if needed
        if self.preprocess_stream:
            self.preprocessor = StreamPreprocessor()
    
    def _get_smoothed_count(self, current_count):
        """Apply temporal smoothing to people count"""
        if self.temporal_smoothing <= 1:
            return current_count
            
        # Add current count to history
        self.people_count_history.append(current_count)
        
        # Calculate smoothed count
        if len(self.people_count_history) < 3:
            return current_count
            
        # Use median filter for robustness against outliers
        return int(np.median(list(self.people_count_history)[-self.temporal_smoothing:]))
    
    def process(self):
        """Process the webcam stream with enhanced person detection"""
        print(f"Starting webcam processing...")
        
        # Process frames
        start_time = time.time()
        frame_count = 0
        total_people = 0
        process_times = []
        last_frame_time = time.time()
        
        try:
            while True:
                # Control processing rate if needed
                if self.fps_limit > 0:
                    elapsed = time.time() - last_frame_time
                    target_time = 1.0 / self.fps_limit
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                
                last_frame_time = time.time()
                frame_count += 1
                
                # Process frame
                process_start = time.time()
                
                # Preprocess image if enabled
                if self.preprocess_stream:
                    enhanced_frame = self.preprocessor.enhance_image(frame)
                    if enhanced_frame is not None:
                        frame_for_detection = enhanced_frame
                    else:
                        frame_for_detection = frame
                else:
                    frame_for_detection = frame
                
                # Detect people
                results = self.model(
                    frame_for_detection,
                    classes=0,  # Only person class
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    augment=self.augment,
                    imgsz=self.img_size
                )
                
                # Processing time
                process_time = time.time() - process_start
                process_times.append(process_time)
                
                # Extract detections
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
                
                # Track objects across frames if enabled
                if self.track_objects:
                    tracked_objects = self.tracker.update(detections)
                    
                    # Use tracked objects for visualization and counting
                    result_img = frame.copy()
                    active_person_count = 0
                    
                    for object_id, obj in tracked_objects.items():
                        # Skip static objects if filtering is enabled
                        if self.filter_static and self.tracker.is_static_object(object_id):
                            # Draw in red to indicate filtered object
                            bbox = obj["bbox"]
                            cv2.rectangle(
                                result_img, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 0, 255),  # Red for static objects (like picture frames)
                                2
                            )
                            
                            # Add "STATIC" label
                            cv2.putText(
                                result_img,
                                "STATIC",
                                (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2
                            )
                            continue
                        
                        # Count as active person
                        active_person_count += 1
                        
                        # Draw bounding box for active person
                        bbox = obj["bbox"]
                        cv2.rectangle(
                            result_img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0),  # Green for active persons
                            2
                        )
                        
                        # Draw ID and confidence
                        label = f"ID:{object_id} {obj['confidence']:.2f}"
                        cv2.putText(
                            result_img,
                            label,
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        
                        # Draw trajectory
                        if len(obj["history"]) > 1:
                            for i in range(1, len(obj["history"])):
                                cv2.line(
                                    result_img,
                                    obj["history"][i-1],
                                    obj["history"][i],
                                    (0, 255, 255),
                                    1
                                )
                    
                    # Count people excluding static objects
                    people_count = active_person_count
                else:
                    # Use regular detections
                    result_img = results[0].plot()
                    people_count = len(detections)
                
                # Apply temporal smoothing to people count
                smoothed_count = self._get_smoothed_count(people_count)
                total_people += smoothed_count
                
                # Generate density map if requested
                if self.density_map and len(detections) > 0:
                    density_map = self.density_estimator.generate_density_map(detections, frame.shape)
                    
                    # Normalize and colorize density map
                    density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
                    
                    # Blend with result image
                    alpha = 0.5
                    result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
                
                # Calculate FPS
                avg_process_time = sum(process_times[-100:]) / min(len(process_times), 100)
                process_fps = 1 / avg_process_time if avg_process_time > 0 else 0
                
                # Add information to image
                cv2.putText(result_img, f"People: {smoothed_count} (raw: {people_count})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_img, f"FPS: {process_fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add false positive filtering status
                if self.filter_static:
                    filter_status = "ON"
                    if self.track_objects:
                        static_count = sum(1 for obj_id in self.tracker.objects if self.tracker.is_static_object(obj_id))
                        filter_status = f"ON ({static_count} filtered)"
                    cv2.putText(result_img, f"Static Filter: {filter_status}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add exit instructions
                cv2.putText(result_img, "Press 'q' to quit", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save detection results periodically
                if self.save_results and frame_count % 30 == 0:  # Every 30 frames
                    timestamp = int(time.time())
                    self.save_detection_results(
                        results, 
                        f"webcam_{timestamp}_{frame_count:06d}", 
                        frame
                    )
                
                # Save processed frame if needed
                if self.save_img and self.writer is not None:
                    self.writer.write(result_img)
                
                # Display frame if needed
                if self.view_img:
                    cv2.imshow("Webcam Analysis - Press 'q' to quit", result_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user")
                        break
                
                # Check if we've processed enough frames
                if self.max_frames > 0 and frame_count >= self.max_frames:
                    print(f"Reached maximum frame count ({self.max_frames})")
                    break
                
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        except Exception as e:
            print(f"Error during webcam processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up
            self.cap.release()
            
            if self.save_img and self.writer is not None:
                self.writer.release()
                
            cv2.destroyAllWindows()
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            avg_people = total_people / frame_count if frame_count > 0 else 0
            avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\nWebcam Processing Complete:")
            print(f"Total frames processed: {frame_count}")
            print(f"Total people detected: {total_people}")
            print(f"Average people per frame: {avg_people:.1f}")
            print(f"Average processing time: {avg_process_time:.3f}s per frame")
            print(f"Average processing FPS: {1/avg_process_time:.1f}" if avg_process_time > 0 else "N/A")
            print(f"Total elapsed time: {elapsed_time:.1f}s")
            
            return total_people, avg_process_time

# Enhanced Live Stream Processor for reliable crowd detection
class EnhancedLiveStreamProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 max_frames=0, buffer_size=90, reconnect_attempts=10, reconnect_delay=3, 
                 stream_timeout=30, fps_limit=0, stream_retry_interval=5, stream_jitter_buffer=15,
                 temporal_smoothing=5, preprocess_stream=True, adaptive_conf=True, 
                 motion_based=False, track_objects=True, filter_static=True,
                 static_threshold=20, min_person_height=50, aspect_ratio_check=True):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir,
                        filter_static, static_threshold, min_person_height,
                        aspect_ratio_check)
                        
        self.max_frames = max_frames
        self.fps_limit = fps_limit
        self.temporal_smoothing = temporal_smoothing
        self.preprocess_stream = preprocess_stream
        self.adaptive_conf = adaptive_conf
        self.motion_based = motion_based
        self.track_objects = track_objects
        
        # Create enhanced stream buffer
        self.stream_buffer = EnhancedStreamBuffer(
            stream_url=source,
            buffer_size=buffer_size,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay=reconnect_delay,
            timeout=stream_timeout,
            retry_interval=stream_retry_interval,
            jitter_buffer=stream_jitter_buffer
        )
        
        # Initialize video writer
        self.writer = None
        
        # Get timestamp for output files
        self.timestamp = int(time.time())
        
        # Status tracking
        self.frame_count = 0
        self.total_people = 0
        self.process_times = []
        
        # Create preprocessor if needed
        if self.preprocess_stream:
            self.preprocessor = StreamPreprocessor()
        
        # Create object tracker if needed
        if self.track_objects:
            self.tracker = ObjectTracker(
                filter_static=filter_static,
                static_threshold=static_threshold,
                min_person_height=min_person_height,
                aspect_ratio_check=aspect_ratio_check
            )
        
        # For temporal smoothing
        self.people_count_history = deque(maxlen=max(30, temporal_smoothing))
        
        # For motion detection
        self.prev_frame = None
        self.motion_threshold = 0.02  # Percentage of pixels that need to change
        
        # For adaptive confidence
        self.base_conf_thres = conf_thres
        self.min_conf_thres = max(0.1, conf_thres - 0.2)
        
        # For debug information
        self.debug_info = {
            "adaptive_conf": 0.0,
            "blur_level": 0.0,
            "brightness": 0.0,
            "motion_detected": False,
            "stream_health": 100,
            "tracking_objects": 0
        }
    
    def _detect_motion(self, current_frame):
        """Detect significant motion between frames"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return True
            
        # Convert current frame to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold to difference
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of pixels that changed
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_percentage = motion_pixels / total_pixels
        
        # Update previous frame
        self.prev_frame = gray
        
        # Return True if motion percentage exceeds threshold
        return motion_percentage > self.motion_threshold
    
    def _get_adaptive_confidence(self, image):
        """Calculate adaptive confidence threshold based on image quality"""
        if not self.adaptive_conf:
            return self.conf_thres
            
        # Detect blur level
        blur_level = self.preprocessor.detect_blur(image)
        self.debug_info["blur_level"] = blur_level
        
        # Detect lighting conditions
        brightness = self.preprocessor.detect_lighting(image)
        self.debug_info["brightness"] = brightness
        
        # Adjust confidence based on image quality
        conf_thres = self.base_conf_thres
        
        # Adjust for blur (lower confidence for blurry images)
        if blur_level < 100:
            blur_factor = max(0.5, min(1.0, blur_level / 100))
            conf_thres *= blur_factor
        
        # Adjust for extreme lighting
        if brightness < 50 or brightness > 200:
            # Lower confidence for poor lighting
            lighting_factor = 0.8
            conf_thres *= lighting_factor
        
        # Get stream health
        stream_health = self.stream_buffer.stream_health / 100
        conf_thres *= max(0.7, stream_health)
        
        # Ensure confidence doesn't go too low
        conf_thres = max(self.min_conf_thres, conf_thres)
        
        self.debug_info["adaptive_conf"] = conf_thres
        return conf_thres
    
    def _get_smoothed_count(self, current_count):
        """Apply temporal smoothing to people count"""
        if self.temporal_smoothing <= 1:
            return current_count
            
        # Add current count to history
        self.people_count_history.append(current_count)
        
        # Calculate smoothed count
        if len(self.people_count_history) < 3:
            return current_count
            
        # Use median filter for robustness against outliers
        return int(np.median(list(self.people_count_history)[-self.temporal_smoothing:]))
    
    def process(self):
        """Process the live stream with enhanced reliability"""
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
        last_detection_time = time.time()
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
                        # Wait briefly before checking again
                        time.sleep(0.5)
                        continue
                    # No new frame yet, continue
                    time.sleep(0.01)
                    continue
                
                last_frame_time = time.time()
                self.frame_count += 1
                
                # Update debug info
                self.debug_info["stream_health"] = self.stream_buffer.stream_health
                
                # Skip processing if using motion-based optimization and no motion detected
                skip_detection = False
                if self.motion_based and self.frame_count > 1:
                    motion_detected = self._detect_motion(frame)
                    self.debug_info["motion_detected"] = motion_detected
                    
                    if not motion_detected and time.time() - last_detection_time < 2.0:
                        skip_detection = True
                
                # Process frame
                process_start = time.time()
                
                # Preprocess image if enabled
                if self.preprocess_stream:
                    frame_for_detection = self.preprocessor.enhance_image(frame)
                    if frame_for_detection is None:
                        print("Error: Frame preprocessing failed")
                        frame_for_detection = frame  # Fallback to original frame
                else:
                    frame_for_detection = frame
                
                # Detect people (skip if motion-based and no motion)
                if not skip_detection:
                    # Get adaptive confidence threshold
                    if self.adaptive_conf:
                        current_conf = self._get_adaptive_confidence(frame)
                    else:
                        current_conf = self.conf_thres
                    
                    # Detect people
                    results = self.model(
                        frame_for_detection, 
                        classes=0,  # Only person class
                        conf=current_conf,
                        iou=self.iou_thres,
                        augment=self.augment,
                        imgsz=self.img_size
                    )
                    
                    # Update last detection time
                    last_detection_time = time.time()
                else:
                    # Use previous results if skipping detection
                    if hasattr(self, 'last_results'):
                        results = self.last_results
                    else:
                        # If no previous results, do detection anyway
                        results = self.model(
                            frame_for_detection, 
                            classes=0,
                            conf=self.conf_thres,
                            iou=self.iou_thres,
                            augment=self.augment,
                            imgsz=self.img_size
                        )
                
                # Save last results
                self.last_results = results
                
                # Processing time
                process_time = time.time() - process_start
                self.process_times.append(process_time)
                
                # Extract detections
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
                
                # Track objects across frames if enabled
                if self.track_objects:
                    tracked_objects = self.tracker.update(detections)
                    self.debug_info["tracking_objects"] = len(tracked_objects)
                    
                    # Use tracked objects for visualization
                    result_img = frame.copy()
                    active_person_count = 0
                    
                    for object_id, obj in tracked_objects.items():
                        # Skip static objects (like picture frames) if filtering is enabled
                        if self.filter_static and self.tracker.is_static_object(object_id):
                            # Draw in red to indicate filtered object
                            bbox = obj["bbox"]
                            cv2.rectangle(
                                result_img, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 0, 255),  # Red for static objects
                                2
                            )
                            
                            # Add "STATIC" label
                            cv2.putText(
                                result_img,
                                "STATIC",
                                (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2
                            )
                            continue
                        
                        # Count as active person
                        active_person_count += 1
                        
                        # Draw bounding box
                        bbox = obj["bbox"]
                        cv2.rectangle(
                            result_img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0),  # Green for active persons
                            2
                        )
                        
                        # Draw ID and confidence
                        label = f"ID:{object_id} {obj['confidence']:.2f}"
                        cv2.putText(
                            result_img,
                            label,
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        
                        # Draw trajectory
                        if len(obj["history"]) > 1:
                            for i in range(1, len(obj["history"])):
                                cv2.line(
                                    result_img,
                                    obj["history"][i-1],
                                    obj["history"][i],
                                    (0, 255, 255),
                                    1
                                )
                    
                    # Count people from active tracked objects
                    people_count = active_person_count
                else:
                    # Use regular detections
                    result_img = results[0].plot()
                    people_count = len(detections)
                
                # Apply temporal smoothing to people count
                smoothed_count = self._get_smoothed_count(people_count)
                self.total_people += smoothed_count
                
                # Generate density map if requested
                if self.density_map and len(detections) > 0:
                    density_map = self.density_estimator.generate_density_map(detections, frame.shape)
                    
                    # Normalize and colorize density map
                    density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
                    
                    # Blend with result image
                    alpha = 0.5
                    result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
                
                # Calculate average FPS (both stream FPS and processing FPS)
                stream_fps = self.stream_buffer.fps
                avg_process_time = sum(self.process_times[-100:]) / min(len(self.process_times), 100)
                process_fps = 1 / avg_process_time if avg_process_time > 0 else 0
                
                # Add information to image
                cv2.putText(result_img, f"People: {smoothed_count} (raw: {people_count})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_img, f"Process FPS: {process_fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_img, f"Stream FPS: {stream_fps:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add stream health info
                health_color = (0, 255, 0)  # Green for good health
                if self.stream_buffer.stream_health < 70:
                    health_color = (0, 165, 255)  # Orange for moderate health
                if self.stream_buffer.stream_health < 40:
                    health_color = (0, 0, 255)  # Red for poor health
                    
                cv2.putText(result_img, f"Stream Health: {self.stream_buffer.stream_health}%", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, health_color, 2)
                
                # Add filter status if enabled
                if self.filter_static:
                    filter_status = "ON"
                    if self.track_objects:
                        static_count = sum(1 for obj_id in self.tracker.objects if self.tracker.is_static_object(obj_id))
                        filter_status = f"ON ({static_count} filtered)"
                    cv2.putText(result_img, f"Static Filter: {filter_status}", (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add debug information if adaptive confidence is enabled
                if self.adaptive_conf:
                    cv2.putText(result_img, f"Conf: {self.debug_info['adaptive_conf']:.2f}", 
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Add exit instructions and stream info
                cv2.putText(result_img, "Press 'q' to quit", (10, 210), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add stream URL (shortened for privacy/display)
                url_parts = urllib.parse.urlparse(self.source)
                display_url = f"{url_parts.scheme}://{url_parts.netloc[:20]}..."
                cv2.putText(result_img, f"Stream: {display_url}", 
                            (10, self.stream_buffer.height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Save detection results periodically
                if self.save_results and self.frame_count % 30 == 0:  # Every 30 frames
                    self.save_detection_results(
                        results, 
                        f"stream_{self.timestamp}_{self.frame_count:06d}", 
                        frame
                    )
                
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
                    
                    cv2.imshow("Enhanced Live Stream Analysis - Press 'q' to quit", display_img)
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

# Video processor with enhanced detection
class VideoProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 max_frames=0, fps_limit=0, temporal_smoothing=5, preprocess_stream=False,
                 track_objects=True, filter_static=True, static_threshold=20,
                 min_person_height=50, aspect_ratio_check=True):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir,
                        filter_static, static_threshold, min_person_height,
                        aspect_ratio_check)
        
        self.max_frames = max_frames
        self.fps_limit = fps_limit
        self.temporal_smoothing = temporal_smoothing
        self.preprocess_stream = preprocess_stream
        self.track_objects = track_objects
        
        # Open video file
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open video file: {source}")
                
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video opened: {self.width}x{self.height} at {self.fps} FPS, {self.total_frames} frames")
        except Exception as e:
            print(f"Error opening video: {e}")
            raise
            
        # Initialize video writer if needed
        self.writer = None
        if self.save_img:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = int(time.time())
            video_name = os.path.basename(source).split('.')[0]
            output_filename = f"{video_name}_processed_{timestamp}.mp4"
            output_path = str(self.output_dir / output_filename)
            
            self.writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                self.fps,
                (self.width, self.height)
            )
            print(f"Recording processed video to: {output_path}")
        
        # For temporal smoothing
        self.people_count_history = deque(maxlen=max(30, temporal_smoothing))
        
        # Initialize preprocessor if needed
        if self.preprocess_stream:
            self.preprocessor = StreamPreprocessor()
            
        # Initialize object tracker if needed
        if self.track_objects:
            self.tracker = ObjectTracker(
                filter_static=filter_static,
                static_threshold=static_threshold,
                min_person_height=min_person_height,
                aspect_ratio_check=aspect_ratio_check
            )
    
    def _get_smoothed_count(self, current_count):
        """Apply temporal smoothing to people count"""
        if self.temporal_smoothing <= 1:
            return current_count
            
        # Add current count to history
        self.people_count_history.append(current_count)
        
        # Calculate smoothed count
        if len(self.people_count_history) < 3:
            return current_count
            
        # Use median filter for robustness against outliers
        return int(np.median(list(self.people_count_history)[-self.temporal_smoothing:]))
    
    def process(self):
        """Process the video file with enhanced person detection"""
        print(f"Processing video: {self.source}")
        
        # Initialize progress bar
        if self.max_frames > 0:
            total_frames = min(self.max_frames, self.total_frames)
        else:
            total_frames = self.total_frames
            
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        # Process frames
        start_time = time.time()
        frame_count = 0
        total_people = 0
        process_times = []
        last_frame_time = time.time()
        
        try:
            while True:
                # Control processing rate if needed
                if self.fps_limit > 0:
                    elapsed = time.time() - last_frame_time
                    target_time = 1.0 / self.fps_limit
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                last_frame_time = time.time()
                frame_count += 1
                pbar.update(1)
                
                # Process frame
                process_start = time.time()
                
                # Preprocess image if enabled
                if self.preprocess_stream:
                    frame_for_detection = self.preprocessor.enhance_image(frame)
                    if frame_for_detection is None:
                        frame_for_detection = frame
                else:
                    frame_for_detection = frame
                
                # Detect people
                results = self.model(
                    frame_for_detection,
                    classes=0,  # Only person class
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    augment=self.augment,
                    imgsz=self.img_size
                )
                
                # Processing time
                process_time = time.time() - process_start
                process_times.append(process_time)
                
                # Extract detections
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
                
                # Track objects across frames if enabled
                if self.track_objects:
                    tracked_objects = self.tracker.update(detections)
                    
                    # Use tracked objects for visualization
                    result_img = frame.copy()
                    active_person_count = 0
                    
                    for object_id, obj in tracked_objects.items():
                        # Skip static objects if filtering is enabled
                        if self.filter_static and self.tracker.is_static_object(object_id):
                            # Draw in red to indicate filtered object
                            bbox = obj["bbox"]
                            cv2.rectangle(
                                result_img, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 0, 255),  # Red for static objects
                                2
                            )
                            
                            # Add "STATIC" label
                            cv2.putText(
                                result_img,
                                "STATIC",
                                (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2
                            )
                            continue
                            
                        # Count as active person
                        active_person_count += 1
                        
                        # Draw bounding box
                        bbox = obj["bbox"]
                        cv2.rectangle(
                            result_img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0),  # Green for active persons
                            2
                        )
                        
                        # Draw ID and confidence
                        label = f"ID:{object_id} {obj['confidence']:.2f}"
                        cv2.putText(
                            result_img,
                            label,
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        
                        # Draw trajectory
                        if len(obj["history"]) > 1:
                            for i in range(1, len(obj["history"])):
                                cv2.line(
                                    result_img,
                                    obj["history"][i-1],
                                    obj["history"][i],
                                    (0, 255, 255),
                                    1
                                )
                    
                    # Count people from active tracked objects
                    people_count = active_person_count
                else:
                    # Use regular detections
                    result_img = results[0].plot()
                    people_count = len(detections)
                
                # Apply temporal smoothing to people count
                smoothed_count = self._get_smoothed_count(people_count)
                total_people += smoothed_count
                
                # Generate density map if requested
                if self.density_map and len(detections) > 0:
                    density_map = self.density_estimator.generate_density_map(detections, frame.shape)
                    
                    # Normalize and colorize density map
                    density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
                    
                    # Blend with result image
                    alpha = 0.5
                    result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
                
                # Calculate FPS
                avg_process_time = sum(process_times[-100:]) / min(len(process_times), 100)
                process_fps = 1 / avg_process_time if avg_process_time > 0 else 0
                
                # Add information to image
                cv2.putText(result_img, f"People: {smoothed_count} (raw: {people_count})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_img, f"FPS: {process_fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_img, f"Frame: {frame_count}/{total_frames}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add filter status if enabled
                if self.filter_static:
                    filter_status = "ON"
                    if self.track_objects:
                        static_count = sum(1 for obj_id in self.tracker.objects if self.tracker.is_static_object(obj_id))
                        filter_status = f"ON ({static_count} filtered)"
                    cv2.putText(result_img, f"Static Filter: {filter_status}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save detection results periodically
                if self.save_results and frame_count % 30 == 0:  # Every 30 frames
                    timestamp = int(time.time())
                    self.save_detection_results(
                        results, 
                        f"video_{timestamp}_{frame_count:06d}", 
                        frame
                    )
                
                # Save processed frame if needed
                if self.save_img and self.writer is not None:
                    self.writer.write(result_img)
                
                # Display frame if needed
                if self.view_img:
                    cv2.imshow("Video Analysis - Press 'q' to quit", result_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user")
                        break
                
                # Check if we've processed enough frames
                if self.max_frames > 0 and frame_count >= self.max_frames:
                    print(f"Reached maximum frame count ({self.max_frames})")
                    break
                
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Close progress bar
            pbar.close()
            
            # Clean up
            self.cap.release()
            
            if self.save_img and self.writer is not None:
                self.writer.release()
                
            cv2.destroyAllWindows()
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            avg_people = total_people / frame_count if frame_count > 0 else 0
            avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\nVideo Processing Complete:")
            print(f"Total frames processed: {frame_count}")
            print(f"Total people detected: {total_people}")
            print(f"Average people per frame: {avg_people:.1f}")
            print(f"Average processing time: {avg_process_time:.3f}s per frame")
            print(f"Average processing FPS: {1/avg_process_time:.1f}" if avg_process_time > 0 else "N/A")
            print(f"Total elapsed time: {elapsed_time:.1f}s")
            
            return total_people, avg_process_time

# Image processor with enhanced detection
class ImageProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 filter_static=False, min_person_height=30, aspect_ratio_check=True):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir,
                        filter_static, 0, min_person_height, aspect_ratio_check)
    
    def process(self):
        """Process a single image with person detection"""
        print(f"Processing image: {self.source}")
        
        try:
            # Load image
            img = cv2.imread(str(self.source))
            if img is None:
                print(f"Error loading image: {self.source}")
                return 0, 0
                
            # Get image dimensions
            height, width = img.shape[:2]
            print(f"Image dimensions: {width}x{height}")
            
            # Process start time
            process_start = time.time()
            
            # Detect people
            results = self.model(
                img,
                classes=0,  # Only person class
                conf=self.conf_thres,
                iou=self.iou_thres,
                augment=self.augment,
                imgsz=self.img_size
            )
            
            # Processing time
            process_time = time.time() - process_start
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    if class_id == 0:  # Person class
                        # Apply basic filtering
                        box_height = y2 - y1
                        box_width = x2 - x1
                        
                        # Skip tiny detections (likely false positives)
                        if box_height < self.min_person_height:
                            continue
                            
                        # Check aspect ratio for person-like objects
                        if self.aspect_ratio_check:
                            aspect_ratio = box_height / box_width if box_width > 0 else 0
                            if not (1.2 < aspect_ratio < 4.5):  # Typical human aspect ratios
                                continue
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "width": box_width,
                            "height": box_height
                        })
            
            # Apply scale-aware post-processing if enabled
            if self.scale_persons:
                detections = self.apply_post_processing(detections, img.shape)
            
            # Count people
            people_count = len(detections)
            
            # Create result image
            result_img = img.copy()
            
            # Draw bounding boxes
            for det in detections:
                bbox = det["bbox"]
                confidence = det["confidence"]
                
                # Draw bounding box
                cv2.rectangle(
                    result_img, 
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    (0, 255, 0), 
                    2
                )
                
                # Draw confidence
                label = f"{confidence:.2f}"
                cv2.putText(
                    result_img,
                    label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Generate density map if requested
            if self.density_map and len(detections) > 0:
                density_map = self.density_estimator.generate_density_map(detections, img.shape)
                
                # Normalize and colorize density map
                density_colored = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
                
                # Blend with result image
                alpha = 0.5
                result_img = cv2.addWeighted(result_img, 1-alpha, density_colored, alpha, 0)
            
            # Add information to image
            cv2.putText(result_img, f"People: {people_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_img, f"Process time: {process_time:.3f}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save detection results
            if self.save_results:
                timestamp = int(time.time())
                img_name = os.path.basename(self.source).split('.')[0]
                self.save_detection_results(
                    results, 
                    f"{img_name}_{timestamp}", 
                    img
                )
            
            # Save processed image
            if self.save_img:
                timestamp = int(time.time())
                img_name = os.path.basename(self.source).split('.')[0]
                output_filename = f"{img_name}_processed_{timestamp}.jpg"
                output_path = str(self.output_dir / output_filename)
                cv2.imwrite(output_path, result_img)
                print(f"Saved processed image to: {output_path}")
            
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
                
                cv2.imshow("Image Analysis - Press any key to exit", display_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(f"\nImage Processing Complete:")
            print(f"People detected: {people_count}")
            print(f"Processing time: {process_time:.3f}s")
            
            return people_count, process_time
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

# Directory processor for batch processing
class DirectoryProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 max_frames=0, batch_size=1, filter_static=False, 
                 min_person_height=30, aspect_ratio_check=True):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir,
                        filter_static, 0, min_person_height, aspect_ratio_check)
        
        self.max_frames = max_frames
        self.batch_size = batch_size
        
        # Get list of image files in directory
        self.image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
            self.image_files.extend(glob.glob(os.path.join(source, f"*.{ext}")))
            self.image_files.extend(glob.glob(os.path.join(source, f"*.{ext.upper()}")))
        
        # Sort files for consistent processing
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in directory: {source}")
    
    def process(self):
        """Process all images in the directory"""
        print(f"Starting batch processing of {len(self.image_files)} images...")
        
        # Limit number of files if max_frames specified
        if self.max_frames > 0 and self.max_frames < len(self.image_files):
            self.image_files = self.image_files[:self.max_frames]
            print(f"Limited to first {self.max_frames} images")
        
        # Process images
        total_people = 0
        total_time = 0
        processed_count = 0
        
        # Create image processor for individual images
        image_processor = ImageProcessor(
            source="",  # Will be set for each image
            model=self.model,
            device="",  # Uses the same device as already loaded
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            img_size=self.img_size,
            augment=self.augment,
            scale_persons=self.scale_persons,
            density_map=self.density_map,
            view_img=self.view_img,
            save_img=self.save_img,
            save_results=self.save_results,
            output_dir=self.output_dir,
            min_person_height=self.min_person_height,
            aspect_ratio_check=self.aspect_ratio_check
        )
        
        # Process each image
        for i, img_path in enumerate(tqdm(self.image_files, desc="Processing images")):
            # Set source for current image
            image_processor.source = img_path
            
            # Process image
            people_count, process_time = image_processor.process()
            
            # Update statistics
            total_people += people_count
            total_time += process_time
            processed_count += 1
        
        # Calculate average statistics
        avg_people = total_people / processed_count if processed_count > 0 else 0
        avg_time = total_time / processed_count if processed_count > 0 else 0
        
        print("\nDirectory Processing Complete:")
        print(f"Total images processed: {processed_count}")
        print(f"Total people detected: {total_people}")
        print(f"Average people per image: {avg_people:.1f}")
        print(f"Average processing time: {avg_time:.3f}s per image")
        
        return total_people, avg_time

# Main function with updated processor selection
def main():
    # Set up arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    if args.device:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get model path - for simplicity, let's use direct loading from torch hub
    model_name = args.model
    if not model_name.startswith('yolov8'):
        model_name = 'yolov8' + model_name
    
    # Detect source type
    source_type = SourceDetector.detect_source_type(args.source, args.source_type)
    print(f"Detected source type: {source_type}")
    
    # Common arguments for processors
    processor_args = {
        'model': model_name,
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
        'output_dir': args.output_dir,
        'filter_static': args.filter_static,
        'static_threshold': args.static_threshold,
        'min_person_height': args.min_person_height,
        'aspect_ratio_check': args.aspect_ratio_check
    }
    
    # Process based on source type
    if source_type == 'image':
        processor = ImageProcessor(
            source=args.source,
            **processor_args
        )
        processor.process()
        
    elif source_type == 'dir':
        processor = DirectoryProcessor(
            source=args.source,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            **processor_args
        )
        processor.process()
        
    elif source_type == 'video':
        processor = VideoProcessor(
            source=args.source,
            max_frames=args.max_frames,
            fps_limit=args.fps_limit,
            temporal_smoothing=args.temporal_smoothing,
            preprocess_stream=args.preprocess_stream,
            track_objects=args.track_objects,
            **processor_args
        )
        processor.process()
        
    elif source_type == 'camera':
        processor = WebcamProcessor(
            source=args.source,
            max_frames=args.max_frames,
            fps_limit=args.fps_limit,
            temporal_smoothing=args.temporal_smoothing,
            preprocess_stream=args.preprocess_stream,
            track_objects=args.track_objects,
            **processor_args
        )
        processor.process()
        
    elif source_type == 'stream':
        # Use enhanced stream processor for better reliability
        processor = EnhancedLiveStreamProcessor(
            source=args.source,
            buffer_size=args.stream_buffer,
            reconnect_attempts=args.reconnect_attempts,
            reconnect_delay=args.reconnect_delay,
            stream_timeout=args.stream_timeout,
            stream_retry_interval=args.stream_retry_interval,
            stream_jitter_buffer=args.stream_jitter_buffer,
            max_frames=args.max_frames,
            fps_limit=args.fps_limit,
            temporal_smoothing=args.temporal_smoothing,
            preprocess_stream=args.preprocess_stream,
            adaptive_conf=args.adaptive_conf,
            motion_based=args.motion_based,
            track_objects=args.track_objects,
            **processor_args
        )
        processor.process()
        
    else:
        print(f"Source type '{source_type}' not supported")

if __name__ == "__main__":
    main()