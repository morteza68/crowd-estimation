"""
Advanced Crowd Estimation with GPU Processing
============================================
This program detects people in live camera feeds, video files, or images.
It uses GPU for faster processing and displays progress using tqdm.

Features:
- Multiple input sources: camera, video, image, directory
- Real-time processing with GPU acceleration
- Progress tracking with tqdm
- Export results to various formats

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
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import urllib.parse
from collections import deque

# Add tqdm for progress display
from tqdm import tqdm

# Set up input arguments
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
    
    # Enhanced stream processing
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
    
# Base processor class
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

# Class for processing video (camera or file)
class VideoProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.5, img_size=640,
                 view_img=True, save_img=False, save_results=False, 
                 output_dir='output', max_frames=0):
        super().__init__(source, model, device, conf_thres, img_size, 
                        view_img, save_img, save_results, output_dir)
        
        self.max_frames = max_frames
        
        # Check if source is a number (camera)
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        
        # Set up video
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(self.source, str) else 0
        
        # Set maximum number of frames
        if self.max_frames > 0 and self.total_frames > 0:
            self.total_frames = min(self.max_frames, self.total_frames)
        
        # Multi-threading queues
        self.frame_queue = Queue(maxsize=4)
        self.result_queue = Queue(maxsize=4)
        self.stopped = False
        
        # Video writer
        self.writer = None
        if self.save_img:
            timestamp = int(time.time())
            source_name = os.path.splitext(os.path.basename(str(source)))[0] if isinstance(source, str) else "camera"
            output_filename = f"{source_name}_{timestamp}.mp4"
            output_path = str(self.output_dir / output_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"Video will be saved to: {output_path}")
    
    def read_frames(self):
        """Read frames from video source in a separate thread with progress display"""
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
                    
                self.frame_queue.put((frame, frame_count))
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
        frame_count = 0
        pbar = None
        
        if self.total_frames > 0:
            pbar = tqdm(total=self.total_frames, desc="Processing frames", position=1)
        
        while not self.stopped:
            if not self.frame_queue.empty() and not self.result_queue.full():
                frame, frame_idx = self.frame_queue.get()
                start_time = time.time()
                
                # Detect people with YOLOv8
                results = self.model(frame, classes=0, conf=self.conf_thres)
                
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
                
                # Save detection results if needed
                if self.save_results and frame_idx % 10 == 0:  # Save every 10th frame to avoid too many files
                    self.save_detection_results(results, f"frame_{frame_idx:06d}")
                
                # Put results in queue
                self.result_queue.put((result_frame, people_count, process_time, frame_idx))
                
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
        
        while not self.stopped or not self.result_queue.empty():
            if not self.result_queue.empty():
                result_frame, people_count, process_time, frame_idx = self.result_queue.get()
                
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
                if self.save_img and self.writer is not None:
                    self.writer.write(result_frame)
                
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
        if self.save_img and self.writer is not None:
            self.writer.release()
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

# Class for processing aerial images
class ImageProcessor(BaseProcessor):
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

# Class for processing multiple images in a directory
class DirectoryProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.5, img_size=640,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 batch_size=1, max_frames=0):
        super().__init__(source, model, device, conf_thres, img_size, 
                        view_img, save_img, save_results, output_dir)
        
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
            
        print(f"Found {len(self.image_files)} images in {source}")
        
    def process(self):
        """Process all images in the directory"""
        print(f"Processing {len(self.image_files)} images...")
        
        total_people = 0
        total_time = 0
        
        # Create progress bar
        with tqdm(total=len(self.image_files), desc="Processing images") as pbar:
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
                results = self.model(img, classes=0, conf=self.conf_thres)
                
                # Processing time
                process_time = time.time() - start_time
                
                # Prepare results
                result_img = results[0].plot()
                
                # Count people
                detections = results[0].boxes
                people_count = len(detections)
                
                # Update totals
                total_people += people_count
                total_time += process_time
                
                # Add information to image
                cv2.putText(result_img, f"People: {people_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_img, f"Process time: {process_time:.3f}s", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Save detection results if needed
                if self.save_results:
                    self.save_detection_results(results, img_name)
                
                # Save processed image if needed
                if self.save_img:
                    output_path = str(self.output_dir / f"{img_name}_processed.jpg")
                    cv2.imwrite(output_path, result_img)
                
                # Display image if needed
                if self.view_img:
                    cv2.imshow("Crowd Estimation - Press 'q' to quit, any other key to continue", result_img)
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
        
        print("\nDirectory Processing Complete:")
        print(f"Total images processed: {len(self.image_files)}")
        print(f"Total people detected: {total_people}")
        print(f"Average people per image: {avg_people:.1f}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        return total_people, total_time

# Enhanced Live Stream Processor for reliable crowd detection
class EnhancedLiveStreamProcessor(BaseProcessor):
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 max_frames=0, buffer_size=90, reconnect_attempts=10, reconnect_delay=3, 
                 stream_timeout=30, fps_limit=0, stream_retry_interval=5, stream_jitter_buffer=15,
                 temporal_smoothing=5, preprocess_stream=True, adaptive_conf=True, 
                 motion_based=False, track_objects=True):
        super().__init__(source, model, device, conf_thres, iou_thres, img_size,
                        augment, scale_persons, density_map,
                        view_img, save_img, save_results, output_dir)
                        
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
            self.tracker = ObjectTracker()
        
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
                    frame = self.preprocessor.enhance_image(frame)
                    if frame is None:
                        print("Error: Frame preprocessing failed")
                        continue
                
                # Detect people (skip if motion-based and no motion)
                if not skip_detection:
                    # Get adaptive confidence threshold
                    if self.adaptive_conf:
                        current_conf = self._get_adaptive_confidence(frame)
                    else:
                        current_conf = self.conf_thres
                    
                    # Detect people
                    results = self.model(
                        frame, 
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
                            frame, 
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
                    
                    for object_id, obj in tracked_objects.items():
                        bbox = obj["bbox"]
                        confidence = obj["confidence"]
                        
                        # Draw bounding box
                        cv2.rectangle(
                            result_img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 
                            2
                        )
                        
                        # Draw ID and confidence
                        label = f"ID:{object_id} {confidence:.2f}"
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
                    
                    # Count people from tracked objects
                    people_count = len(tracked_objects)
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
                
                # Add debug information if adaptive confidence is enabled
                if self.adaptive_conf:
                    cv2.putText(result_img, f"Conf: {self.debug_info['adaptive_conf']:.2f}", 
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Add exit instructions and stream info
                cv2.putText(result_img, "Press 'q' to quit", (10, 180), 
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
                    frame = self.preprocessor.enhance_image(frame)
                    if frame is None:
                        print("Error: Frame preprocessing failed")
                        continue
                
                # Detect people (skip if motion-based and no motion)
                if not skip_detection:
                    # Get adaptive confidence threshold
                    if self.adaptive_conf:
                        current_conf = self._get_adaptive_confidence(frame)
                    else:
                        current_conf = self.conf_thres
                    
                    # Detect people
                    results = self.model(
                        frame, 
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
                            frame, 
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
                    
                    for object_id, obj in tracked_objects.items():
                        bbox = obj["bbox"]
                        confidence = obj["confidence"]
                        
                        # Draw bounding box
                        cv2.rectangle(
                            result_img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 
                            2
                        )
                        
                        # Draw ID and confidence
                        label = f"ID:{object_id} {confidence:.2f}"
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
                    
                    # Count people from tracked objects
                    people_count = len(tracked_objects)
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
                
                # Add debug information if adaptive confidence is enabled
                if self.adaptive_conf:
                    cv2.putText(result_img, f"Conf: {self.debug_info['adaptive_conf']:.2f}", 
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Add exit instructions and stream info
                cv2.putText(result_img, "Press 'q' to quit", (10, 180), 
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
    print("ADVANCED CROWD ESTIMATION SYSTEM")
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
    if source_type == 'camera' or source_type == 'video':
        processor = VideoProcessor(
            source=args.source,
            model=model_path,
            device=device,
            conf_thres=args.conf_thres,
            img_size=args.img_size,
            view_img=args.view_img,
            save_img=args.save_img,
            save_results=args.save_results,
            output_dir=args.output_dir,
            max_frames=args.max_frames
        )
        processor.start() 
        # Process based on source type
    elif source_type == 'image':
        processor = ImageProcessor(
            source=args.source,
            **kwargs
        )
        processor.process()    
        
    elif source_type == 'dir':
        
        processor = DirectoryProcessor(
            source=args.source,
            model=model_path,
            device=device,
            conf_thres=args.conf_thres,
            img_size=args.img_size,
            view_img=args.view_img,
            save_img=args.save_img,
            save_results=args.save_results,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_frames=args.max_frames
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
        print(f"Source type '{source_type}' not yet implemented for aerial analysis")
        print(f"Please use image, directory, or stream input")

# Entry point
if __name__ == "__main__":
    main()