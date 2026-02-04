#!/usr/bin/env python3
import cv2
import socket
import time
import threading
import requests
from imutils.video import VideoStream
from options import Options

# --- Configuration ---
PI_IP = "100.103.111.233"
RTSP_URL = "rtsp://100.103.111.233:8554/feeder"
UDP_PORT = 5005
MIN_CONFIDENCE = 0.6
RESIZE_WIDTH = 480  # Reduced slightly for significantly faster API response

# Shared Resources
frame_lock = threading.Lock()
latest_frame = None
cached_results = []
is_running = True

# Networking
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
session = requests.Session()
opts = Options()
current_led_state = None

def control_led(should_be_on):
    global current_led_state
    if current_led_state != should_be_on:
        try:
            sock.sendto(b'LED_ON' if should_be_on else b'LED_OFF', (PI_IP, UDP_PORT))
            current_led_state = should_be_on
        except: pass

def processing_thread():
    """ Background thread: Dedicated only to API calls """
    global latest_frame, cached_results, is_running
    
    while is_running:
        frame_to_proc = None
        
        # 1. Grab the freshest frame available
        with frame_lock:
            if latest_frame is not None:
                frame_to_proc = latest_frame.copy()
        
        if frame_to_proc is not None:
            try:
                h, w = frame_to_proc.shape[:2]
                # Fast Resize for Detection
                small_frame = cv2.resize(frame_to_proc, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))
                _, encoded = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # Low quality = high speed
                
                # API Call
                detect_response = session.post(opts.endpoint("vision/face"), 
                                               files={"image": encoded.tobytes()}, 
                                               timeout=0.8).json()
                
                predictions = detect_response.get('predictions', [])
                new_results = []
                
                if predictions:
                    # Calculate scale once
                    scale_x = w / RESIZE_WIDTH
                    scale_y = h / (h * RESIZE_WIDTH / w)
                    
                    for pred in predictions:
                        x_min, y_min = int(pred['x_min'] * scale_x), int(pred['y_min'] * scale_y)
                        x_max, y_max = int(pred['x_max'] * scale_x), int(pred['y_max'] * scale_y)
                        
                        # Crop & Recognize
                        face = frame_to_proc[y_min:y_max, x_min:x_max]
                        if face.size == 0: continue
                        
                        _, f_enc = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        rec_res = session.post(opts.endpoint("vision/face/recognize"), 
                                               files={"image": f_enc.tobytes()}, 
                                               data={"min_confidence": MIN_CONFIDENCE}, 
                                               timeout=0.8).json()
                        
                        name = "Unknown"
                        conf = 0
                        if rec_res.get("predictions"):
                            name = rec_res["predictions"][0].get("userid", "Unknown")
                            conf = rec_res["predictions"][0].get("confidence", 0)
                        
                        new_results.append({'bbox': (x_min, y_min, x_max, y_max), 'name': name, 'confidence': conf})

                # Update shared results and LED
                cached_results = new_results
                control_led(any(r['name'] != "Unknown" for r in new_results) if new_results else False)
                
            except Exception as e:
                print(f"Sync Error: {e}")
        
        # Small sleep to prevent CPU saturation while waiting for next frame
        time.sleep(0.01)

def main():
    global latest_frame, is_running
    
    # Use higher FPS RTSP handling
    cap = VideoStream(RTSP_URL).start()
    time.sleep(2.0)
    
    # Launch worker thread
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    
    print("Syncing recognition with video. Press 'q' to quit.")

    while True:
        frame = cap.read()
        if frame is None: continue
        
        # Atomic update of the frame
        with frame_lock:
            latest_frame = frame
        
        # Draw results on the LIVE frame
        # Note: Bounding boxes will update as fast as the API allows, 
        # but the video will never stutter.
        for result in cached_results:
            x_min, y_min, x_max, y_max = result['bbox']
            name, conf = result['name'], result['confidence']
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Fast Sync Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    is_running = False
    t.join()
    control_led(False)
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()