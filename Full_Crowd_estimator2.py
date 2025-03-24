"""
Advanced Crowd Detection System
==============================
Comprehensive crowd detection for multiple input sources including live streams.
Optimized for aerial imagery with enhanced accuracy and visualization.

Features:
- Multiple input sources: camera, video, image, directory, live streams (RTSP, HTTP, etc.)
- Real-time processing with GPU acceleration
- Specialized aerial detection with scale-aware processing
- Enhanced live stream processing with temporal smoothing and adaptive confidence
- Density map generation for crowd visualization
- Object tracking for consistent detection
- Webcam-specific people counting with photo frame filtering

Author: Team Graphics
Date: 2025-03-24
Version: 2.0
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
    
    # Webcam people counting options
    parser.add_argument('--filter-photo-frames', action='store_true',
                      help='Filter out photo frames from people counting')
    parser.add_argument('--min-person-size', type=float, default=0.05,
                      help='Minimum person size relative to frame size (0-1)')
    parser.add_argument('--max-person-size', type=float, default=0.9,
                      help='Maximum person size relative to frame size (0-1)')
    parser.add_argument('--person-movement-threshold', type=float, default=0.02,
                      help='Minimum movement to consider as real person (relative to frame size)')
    
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

# Object Tracker for consistent detection across frames
class ObjectTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # Dictionary of tracked objects
        self.disappeared = {}  # Number of consecutive frames object has been lost
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox, confidence):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "confidence": confidence,
            "history": [centroid]
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        """Update tracked objects with new detections"""
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for detection in detections:
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
        for detection in detections:
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
            
            # Limit history length
            if len(self.objects[object_id]["history"]) > 30:
                self.objects[object_id]["history"] = self.objects[object_id]["history"][-30:]
                
            self.disappeared[object_id] = 0
            
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

# New class for photo frame detection and filtering
class PhotoFrameDetector:
    def __init__(self):
        # Initialize parameters for photo frame detection
        self.movement_history = {}  # Track movement of detected objects
        self.movement_threshold = 0.02  # Minimum movement to consider as real person (relative to frame size)
        self.frame_history = deque(maxlen=10)  # Store recent frames for movement analysis
        self.edge_detector = cv2.createCanny(100, 200, 3, False)
        self.last_frame_gray = None
        
    def set_movement_threshold(self, threshold):
        """Set the movement threshold for real person detection"""
        self.movement_threshold = threshold
        
    def is_photo_frame(self, detection, frame_shape, current_frame):
        """Determine if a detection is likely a photo frame rather than a real person"""
        # Get detection details
        bbox = detection["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        object_id = detection.get("id", -1)
        
        # Calculate relative size
        frame_height, frame_width = frame_shape[:2]
        bbox_width, bbox_height = x2 - x1, y2 - y1
        relative_width = bbox_width / frame_width
        relative_height = bbox_height / frame_height
        
        # Convert current frame to grayscale
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
            
        # Initialize last frame if needed
        if self.last_frame_gray is None:
            self.last_frame_gray = current_gray
            self.frame_history.append(current_gray)
            return False  # Can't determine on first frame
            
        # Calculate region of interest
        roi_x1, roi_y1 = max(0, x1 - 10), max(0, y1 - 10)
        roi_x2, roi_y2 = min(frame_width, x2 + 10), min(frame_height, y2 + 10)
        
        # Check for movement
        if object_id in self.movement_history:
            prev_centroid = self.movement_history[object_id]["centroid"]
            current_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Calculate movement as fraction of frame size
            movement_x = abs(current_centroid[0] - prev_centroid[0]) / frame_width
            movement_y = abs(current_centroid[1] - prev_centroid[1]) / frame_height
            movement = max(movement_x, movement_y)
            
            # Update history
            self.movement_history[object_id] = {
                "centroid": current_centroid,
                "movement": movement,
                "history": self.movement_history[object_id]["history"] + [movement]
            }
            
            # Keep history length reasonable
            if len(self.movement_history[object_id]["history"]) > 30:
                self.movement_history[object_id]["history"] = self.movement_history[object_id]["history"][-30:]
                
            # Calculate average movement
            avg_movement = sum(self.movement_history[object_id]["history"]) / len(self.movement_history[object_id]["history"])
            
            # If consistent movement, likely a real person
            if avg_movement > self.movement_threshold:
                return False
        else:
            # First time seeing this object, initialize tracking
            self.movement_history[object_id] = {
                "centroid": ((x1 + x2) // 2, (y1 + y2) // 2),
                "movement": 0,
                "history": [0]
            }
            
        # Extract region of interest for additional analysis
        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
            roi_current = current_gray[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Check for frame-like edges
            edges = self.edge_detector.detect(roi_current)
            edge_ratio = np.sum(edges > 0) / (roi_current.size)
            
            # Photo frames often have strong rectangular edges
            if edge_ratio > 0.2:  # High edge content
                # Check for rectangular pattern
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Reasonably sized contour
                        # Check if contour is rectangular
                        peri = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                        if len(approx) == 4:  # Rectangular shape
                            return True  # Likely a photo frame
        
        # Update last frame
        self.last_frame_gray = current_gray
        self.frame_history.append(current_gray)
        
        # Default: if we can't determine, assume it's not a photo frame
        return False
    
    def filter_photo_frames(self, detections, frame):
        """Filter out detections that are likely photo frames"""
        if frame is None:
            return detections
            
        filtered_detections = []
        frame_shape = frame.shape
        
        for detection in detections:
            if not self.is_photo_frame(detection, frame_shape, frame):
                filtered_detections.append(detection)
                
        return filtered_detections
    
    def cleanup_old_tracks(self, active_ids):
        """Clean up tracking history for objects no longer detected"""
        for object_id in list(self.movement_history.keys()):
            if object_id not in active_ids:
                del self.movement_history[object_id]

# Base processor class with aerial-specific enhancements
class BaseProcessor:
    def __init__(self, source, model, device, conf_thres=0.3, iou_thres=0.45, img_size=1280,
                 augment=False, scale_persons=False, density_map=False,
                 view_img=True, save_img=False, save_results=False, output_dir='output',
                 filter_photo_frames=False, min_person_size=0.05, max_person_size=0.9,
                 person_movement_threshold=0.02):
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
        self.output_dir = output_dir
        
        # Initialize photo frame filtering if enabled
        self.filter_photo_frames = filter_photo_frames
        self.min_person_size = min_person_size
        self.max_person_size = max_person_size
        self.photo_frame_detector = PhotoFrameDetector()
        self.photo_frame_detector.set_movement_threshold(person_movement_threshold)
        
        # Ensure output directory exists
        if save_img or save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # Half precision only on CUDA
        
        # Load model
        self.model = attempt_load(model, map_location=self.device)
        self.stride = int(self.model.stride.max())
        if self.half:
            self.model.half()
            
        # Initialize density map estimator if needed
        if self.density_map:
            self.density_estimator = CrowdDensityEstimator()
            
        # Initialize object tracker
        self.object_tracker = ObjectTracker()
        
        # Get class names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Initialize webcam-specific variables
        self.is_webcam = False
        self.frame_count = 0
        self.webcam_fps = 0
        self.webcam_start_time = time.time()
        
        # Initialize colors for visualization
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
    def preprocess(self, img0):
        """Preprocess image for inference"""
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Add batch dimension if needed
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
    
    def detect(self, img0):
        """Run detection on the image"""
        # Preprocess
        img = self.preprocess(img0)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]
            
        # Apply NMS
        pred = non_max_suppression(
            pred, 
            self.conf_thres, 
            self.iou_thres
        )
        
        # Process detections
        results = []
        for i, det in enumerate(pred):  # Per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Results
                for *xyxy, conf, cls in reversed(det):
                    # Only include person class (usually class 0)
                    if int(cls) == 0:  # Person class
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        
                        # Calculate relative size
                        img_height, img_width = img0.shape[:2]
                        rel_width = (x2 - x1) / img_width
                        rel_height = (y2 - y1) / img_height
                        
                        # Filter by size if needed (for webcam)
                        if self.is_webcam:
                            if rel_width < self.min_person_size or rel_height < self.min_person_size:
                                continue  # Too small
                            if rel_width > self.max_person_size or rel_height > self.max_person_size:
                                continue  # Too large
                        
                        results.append({
                            "class": int(cls),
                            "class_name": self.names[int(cls)],
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf),
                            "rel_size": max(rel_width, rel_height)
                        })
                        
        # Update object tracker if needed
        if hasattr(self, 'object_tracker'):
            tracked_objects = self.object_tracker.update(results)
            
            # Add tracking IDs to results
            for result in results:
                bbox = result["bbox"]
                centroid = (
                    int((bbox[0] + bbox[2]) / 2), 
                    int((bbox[1] + bbox[3]) / 2)
                )
                
                # Find closest tracked object
                min_dist = float('inf')
                closest_id = None
                
                for obj_id, obj in tracked_objects.items():
                    obj_centroid = obj["centroid"]
                    dist = ((centroid[0] - obj_centroid[0]) ** 2 + 
                           (centroid[1] - obj_centroid[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = obj_id
                        
                if closest_id is not None and min_dist < 50:  # Threshold for association
                    result["id"] = closest_id
                    result["tracking_history"] = tracked_objects[closest_id]["history"]
        
        # Filter out photo frames if enabled
        if self.filter_photo_frames:
            active_ids = [r.get("id", -1) for r in results]
            results = self.photo_frame_detector.filter_photo_frames(results, img0)
            self.photo_frame_detector.cleanup_old_tracks(active_ids)
                    
        return results
    
    def visualize(self, img0, results):
        """Visualize detection results on the image"""
        # Create a copy for visualization
        vis_img = img0.copy()
        
        # Generate density map if needed
        if self.density_map:
            density = self.density_estimator.generate_density_map(results, img0.shape)
            # Normalize and colorize density map
            if density.max() > 0:
                density_norm = (density / density.max() * 255).astype(np.uint8)
                density_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
                # Blend with original image
                alpha = 0.5
                vis_img = cv2.addWeighted(vis_img, 1-alpha, density_color, alpha, 0)
        
        # Draw bounding boxes
        for result in results:
            bbox = result["bbox"]
            x1, y1, x2, y2 = bbox
            cls = result["class"]
            conf = result["confidence"]
            
            # Get color for this class
            color = self.colors[cls]
            
            # Draw box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.names[cls]} {conf:.2f}"
            if "id" in result:
                label += f" ID:{result['id']}"
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_img, (x1, y1-text_size[1]-4), (x1+text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_img, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Draw tracking history if available
            if "tracking_history" in result and len(result["tracking_history"]) > 1:
                history = result["tracking_history"]
                for i in range(1, len(history)):
                    cv2.line(vis_img, history[i-1], history[i], color, 2)
        
        # Draw total count
        person_count = len(results)
        cv2.putText(vis_img, f"People count: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw FPS if webcam
        if self.is_webcam:
            # Calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.webcam_start_time
            if elapsed >= 1.0:
                self.webcam_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.webcam_start_time = time.time()
                
            cv2.putText(vis_img, f"FPS: {self.webcam_fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_img
    
    def save_results_to_file(self, results, frame_idx, img_shape):
        """Save detection results to a JSON file"""
        if not self.save_results:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"results_{timestamp}_{frame_idx}.json")
        
        # Prepare data
        data = {
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "image_shape": {
                "height": img_shape[0],
                "width": img_shape[1],
                "channels": img_shape[2] if len(img_shape) > 2 else 1
            },
            "detections": []
        }
        
        # Add each detection
        for result in results:
            detection = {
                "class": result["class"],
                "class_name": result["class_name"],
                "bbox": result["bbox"],
                "confidence": result["confidence"]
            }
            
            # Add tracking ID if available
            if "id" in result:
                detection["id"] = result["id"]
                
            data["detections"].append(detection)
            
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
    
    def process_webcam(self):
        """Process webcam input"""
        self.is_webcam = True
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam {self.source}")
            return
            
        print(f"Processing webcam feed from {self.source}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from webcam")
                break
                
            # Run detection
            results = self.detect(frame)
            
            # Visualize results
            vis_frame = self.visualize(frame, results)
            
            # Display
            if self.view_img:
                cv2.imshow("Webcam People Counter", vis_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) == ord('q'):
                    break
                    
            # Save results
            if self.save_results:
                self.save_results_to_file(results, frame_idx, frame.shape)
                
            # Save image
            if self.save_img:
                output_path = os.path.join(self.output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(output_path, vis_frame)
                
            frame_idx += 1
            
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self):
        """Process video input"""
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.source}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} at {fps} FPS, {total_frames} frames")
        
        # Create video writer if saving
        if self.save_img:
            output_path = os.path.join(self.output_dir, f"output_{Path(self.source).stem}.mp4")
            writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (width, height)
            )
            
        # Process frames
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection
            results = self.detect(frame)
            
            # Visualize results
            vis_frame = self.visualize(frame, results)
            
            # Display
            if self.view_img:
                cv2.imshow("Video People Counter", vis_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) == ord('q'):
                    break
                    
            # Save results
            if self.save_results:
                self.save_results_to_file(results, frame_idx, frame.shape)
                
            # Save frame to output video
            if self.save_img:
                writer.write(vis_frame)
                
            frame_idx += 1
            pbar.update(1)
            
        # Release resources
        cap.release()
        if self.save_img:
            writer.release()
        cv2.destroyAllWindows()
        pbar.close()
    
    def process_image(self):
        """Process a single image"""
        # Read image
        img0 = cv2.imread(self.source)
        
        if img0 is None:
            print(f"Error: Could not read image {self.source}")
            return
            
        print(f"Processing image: {self.source}")
        
        # Run detection
        results = self.detect(img0)
        
        # Visualize results
        vis_img = self.visualize(img0, results)
        
        # Display
        if self.view_img:
            cv2.imshow("Image People Counter", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        # Save results
        if self.save_results:
            self.save_results_to_file(results, 0, img0.shape)
            
        # Save image
        if self.save_img:
            output_path = os.path.join(self.output_dir, f"output_{Path(self.source).stem}.jpg")
            cv2.imwrite(output_path, vis_img)
    
    def process_directory(self):
        """Process a directory of images"""
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.source, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(self.source, f"*{ext.upper()}")))
            
        if not image_files:
            print(f"Error: No images found in directory {self.source}")
            return
            
        print(f"Processing {len(image_files)} images from directory: {self.source}")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            # Read image
            img0 = cv2.imread(image_path)
            
            if img0 is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            # Run detection
            results = self.detect(img0)
            
            # Visualize results
            vis_img = self.visualize(img0, results)
            
            # Display
            if self.view_img:
                cv2.imshow("Image People Counter", vis_img)
                if cv2.waitKey(1) == ord('q'):
                    break
                    
            # Save results
            if self.save_results:
                self.save_results_to_file(results, idx, img0.shape)
                
            # Save image
            if self.save_img:
                output_path = os.path.join(self.output_dir, f"output_{Path(image_path).stem}.jpg")
                cv2.imwrite(output_path, vis_img)
                
        # Close any open windows
        if self.view_img:
            cv2.destroyAllWindows()
    
    def process_stream(self):
        """Process a video stream"""
        # Create enhanced stream buffer
        stream_buffer = EnhancedStreamBuffer(
            self.source,
            buffer_size=90,
            reconnect_attempts=10,
            reconnect_delay=3,
            timeout=30,
            retry_interval=5,
            jitter_buffer=15
        )
        
        # Start buffer
        if not stream_buffer.start():
            print(f"Error: Could not connect to stream {self.source}")
            return
            
        print(f"Processing stream: {self.source}")
        
        # Create video writer if saving
        if self.save_img:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"stream_{timestamp}.mp4")
            writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                30,  # Default to 30 FPS
                (stream_buffer.width, stream_buffer.height)
            )
            
        # Process frames
        frame_idx = 0
        
        while True:
            # Read frame from buffer
            ret, frame = stream_buffer.read()
            
            if not ret:
                print("Warning: Failed to grab frame from stream")
                time.sleep(0.1)
                continue
                
            # Run detection
            results = self.detect(frame)
            
            # Visualize results
            vis_frame = self.visualize(frame, results)
            
            # Display
            if self.view_img:
                cv2.imshow("Stream People Counter", vis_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) == ord('q'):
                    break
                    
            # Save results
            if self.save_results:
                self.save_results_to_file(results, frame_idx, frame.shape)
                
            # Save frame to output video
            if self.save_img:
                writer.write(vis_frame)
                
            frame_idx += 1
            
        # Release resources
        stream_buffer.stop()
        if self.save_img:
            writer.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Run the processor based on source type"""
        # Detect source type
        source_type = SourceDetector.detect_source_type(self.source)
        
        print(f"Detected source type: {source_type}")
        
        # Process based on source type
        if source_type == 'camera':
            self.process_webcam()
        elif source_type == 'video':
            self.process_video()
        elif source_type == 'image':
            self.process_image()
        elif source_type == 'dir':
            self.process_directory()
        elif source_type == 'stream':
            self.process_stream()
        else:
            print(f"Error: Unknown source type {source_type}")

# Helper functions for model loading and preprocessing
def select_device(device=''):
    # Select device (CPU or GPU)
    if device.lower() == 'cpu':
        return torch.device('cpu')
    elif device.lower().startswith(('cuda', '0')):
        return torch.device('cuda:0')
    else:
        # Auto-select
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def attempt_load(weights, map_location=None):
    # Attempt to load model from weights file
    # This is a placeholder - in a real implementation, this would load YOLOv8
    # For this example, we'll create a dummy model
    class DummyModel:
        def __init__(self):
            self.stride = torch.tensor([32.0])
            self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
            
        def __call__(self, img, augment=False):
            # Return dummy predictions
            batch_size = img.shape[0]
            return [torch.rand(batch_size, 10, 6)]  # [batch, num_detections, (x1,y1,x2,y2,conf,cls)]
            
        def half(self):
            # Dummy half precision
            return self
            
    print(f"Loading model from {weights}...")
    return DummyModel()

def letterbox(img, new_shape=(640, 640), stride=32, auto=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    # Divide padding into 2 sides
    dw /= 2
    dh /= 2
    
    # Resize
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return img, r, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    # Calculate gain
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    
    # Calculate padding
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    # Apply transformations
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clip bounding xyxy bounding boxes to image shape
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    
    return coords

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    # Non-Maximum Suppression to filter detections
    # This is a simplified placeholder implementation
    # In a real implementation, this would perform actual NMS
    
    # Filter by confidence
    mask = prediction[..., 4] > conf_thres
    filtered = []
    
    for i, pred in enumerate(prediction):
        det = pred[mask[i]]
        if len(det):
            filtered.append(det)
        else:
            filtered.append(torch.zeros((0, 6)))
            
    return filtered

def main():
    """Main function to run the crowd detection system"""
    # Parse arguments
    args = parse_args()
    
    # Prepare model
    if args.model.startswith('yolov'):
        # Download model if needed
        loader = GitHubModelLoader(args.github_repo, args.github_branch)
        model_path = loader.download_model(args.model)
    else:
        model_path = args.model
        
    # Create processor
    processor = BaseProcessor(
        source=args.source,
        model=model_path,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        augment=args.augment,
        scale_persons=args.scale_persons,
        density_map=args.density_map,
        view_img=args.view_img,
        save_img=args.save_img,
        save_results=args.save_results,
        output_dir=args.output_dir,
        filter_photo_frames=args.filter_photo_frames,
        min_person_size=args.min_person_size,
        max_person_size=args.max_person_size,
        person_movement_threshold=args.person_movement_threshold
    )
    
    # Run processor
    processor.run()

if __name__ == "__main__":
    main()