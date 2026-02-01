#!/usr/bin/env python3
import cv2
import os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

rtsp_url = "rtsp://172.23.28.195:8554/feeder"

def main():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Cannot open stream")
        return
    
    print("Testing stream smoothness - Press 'q' to quit")
    
    frame_count = 0
    start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Calculate actual FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start
            fps = 30 / elapsed
            print(f"Receiving FPS: {fps:.1f}")
            start = time.time()
        
        cv2.imshow('Stream Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()