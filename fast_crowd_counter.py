"""
Fast Crowd Estimation with GPU Processing - Optimized for Speed
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
    parser = argparse.ArgumentParser(description='Fast Crowd Estimation with Deep Learning - Optimized')
    parser.add_argument('--source', type=str, default='0', help='Input source: camera number or video path')
    parser.add_argument('--model', type=str, default='yolov8n', help='Model to use: yolov8n, yolov8s')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu, cuda:0, 0)')
    parser.add_argument('--view-img', action='store_true', help='Display results')
    parser.add_argument('--img-size', type=int, default=320, help='Input image size')
    parser.add_argument('--frame-skip', type=int, default=2, help='Process every n frames')
    return parser.parse_args()

class FastVideoProcessor:
    def __init__(self, source, model, device, conf_thres=0.5, view_img=True, img_size=320, frame_skip=2):
        self.source = source if not source.isdigit() else int(source)
        self.conf_thres = conf_thres
        self.view_img = view_img
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.device = device
        
        # Set up video
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source {source}")
        
        # Set lower resolution for webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Load model
        self.model = self.load_model(model, device)
        
        # Multi-threaded queues
        self.frame_queue = Queue(maxsize=2)  # Smaller queue size to reduce latency
        self.result_queue = Queue(maxsize=2)
        self.stopped = False
        
        # Frame counter
        self.frame_counter = 0
        
    def load_model(self, model_path, device):
        try:
            import ultralytics
            from ultralytics import YOLO
            
            print(f"Loading model {model_path} on {device}...")
            
            # If only model name is given (not a file path)
            if not model_path.endswith('.pt') and not model_path.endswith('.onnx'):
                model_path = f"{model_path}.pt"
            
            model = YOLO(model_path)
            
            # Move to appropriate device
            model.to(device)
            
            # Use FP16 for faster speed on GPU
            if device != 'cpu' and torch.cuda.is_available():
                model.model.half()
            
            print(f"Model loaded successfully")
            return model
            
        except ImportError:
            print("Installing ultralytics...")
            os.system("pip install ultralytics")
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def read_frames(self):
        """Read frames from the video source in a separate thread"""
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (self.img_size, int(self.img_size * self.height / self.width)))
                
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)  # Less delay
    
    def process_frames(self):
        """Process frames using the model in a separate thread"""
        while not self.stopped:
            if not self.frame_queue.empty() and not self.result_queue.full():
                frame = self.frame_queue.get()
                self.frame_counter += 1
                
                # Process every few frames
                if self.frame_counter % self.frame_skip == 0:
                    start_time = time.time()
                    
                    # Detect people with YOLOv8
                    results = self.model(frame, classes=0, conf=self.conf_thres, verbose=False)
                    
                    # Processing time
                    process_time = time.time() - start_time
                    
                    # Prepare results
                    result_frame = results[0].plot()
                    
                    # Count number of people
                    detections = results[0].boxes
                    people_count = len(detections)
                    
                    # Add info to image
                    cv2.putText(result_frame, f"People: {people_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(result_frame, f"FPS: {1/process_time:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Put results in queue
                    self.result_queue.put((result_frame, people_count, process_time))
                else:
                    # For skipped frames, send the original frame without processing
                    self.result_queue.put((frame, -1, 0))
            else:
                time.sleep(0.001)  # Less delay
    
    def display_frames(self):
        """Display processed results in the main thread"""
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
                
                # If this frame was processed
                if people_count >= 0:
                    processed_frames += 1
                    total_time += process_time
                    total_people += people_count
                    last_people_count = people_count
                else:
                    # For skipped frames, use the last count
                    cv2.putText(result_frame, f"People: {last_people_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Calculate overall FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Display overall FPS
                cv2.putText(result_frame, f"Overall FPS: {fps:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display image
                if self.view_img:
                    cv2.imshow("Crowd Estimation (Optimized)", result_frame)
                    
                    # Exit with q key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stopped = True
                        break
        
        # Display final stats
        avg_people = total_people / processed_frames if processed_frames > 0 else 0
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        overall_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nFinal Statistics:")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_frames}")
        print(f"Average people count: {avg_people:.1f}")
        print(f"Average processing FPS: {avg_fps:.1f}")
        print(f"Average overall FPS: {overall_fps:.1f}")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """Start video processing using multiple threads"""
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

def main():
    # Set up arguments
    args = parse_args()
    
    # Set up processing device
    if args.device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Image size: {args.img_size}px")
    print(f"Frame skip: every {args.frame_skip} frames")
    
    # Start video processing
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