#!/usr/bin/env python3
import cv2
import socket
import time
import threading
import requests
from imutils.video import VideoStream
from options import Options

# Configuration
PI_IP = "100.103.111.233"
RTSP_URL = "rtsp://100.103.111.233:8554/feeder"
UDP_PORT = 5005
MIN_CONFIDENCE = 0.6
RESIZE_WIDTH = 640

# Global state for threading
latest_frame = None
cached_results = []
is_running = True
new_frame_available = False

# Socket & Session setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
session = requests.Session()  # Connection pooling for lower latency
opts = Options()
current_led_state = None

def control_led(should_be_on):
    global current_led_state
    if current_led_state != should_be_on:
        try:
            sock.sendto(b'LED_ON' if should_be_on else b'LED_OFF', (PI_IP, UDP_PORT))
            current_led_state = should_be_on
        except Exception as e:
            print(f"LED Error: {e}")

def processing_thread():
    """ Background thread for API calls """
    global latest_frame, cached_results, is_running, new_frame_available
    
    while is_running:
        if latest_frame is not None and new_frame_available:
            new_frame_available = False
            frame_to_proc = latest_frame.copy()
            
            try:
                # 1. Resize and Encode
                h, w = frame_to_proc.shape[:2]
                small_frame = cv2.resize(frame_to_proc, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))
                _, encoded = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                # 2. API Detection
                detect_response = session.post(
                    opts.endpoint("vision/face"),
                    files={"image": encoded.tobytes()},
                    timeout=1
                ).json()
                
                predictions = detect_response.get('predictions', [])
                results = []
                
                if predictions:
                    scale_x = w / RESIZE_WIDTH
                    scale_y = h / (h * RESIZE_WIDTH / w)
                    
                    for pred in predictions:
                        x_min, y_min = int(pred['x_min'] * scale_x), int(pred['y_min'] * scale_y)
                        x_max, y_max = int(pred['x_max'] * scale_x), int(pred['y_max'] * scale_y)
                        
                        face = frame_to_proc[y_min:y_max, x_min:x_max]
                        if face.size == 0: continue
                        
                        # 3. API Recognition
                        _, f_enc = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        rec_res = session.post(
                            opts.endpoint("vision/face/recognize"),
                            files={"image": f_enc.tobytes()},
                            data={"min_confidence": MIN_CONFIDENCE},
                            timeout=1
                        ).json()
                        
                        name = "Unknown"
                        conf = 0
                        if rec_res.get("predictions"):
                            user = rec_res["predictions"][0]
                            name = user.get("userid", "Unknown")
                            conf = user.get("confidence", 0)
                        
                        results.append({'bbox': (x_min, y_min, x_max, y_max), 'name': name, 'confidence': conf})
                
                # Update shared results and LED
                cached_results = results
                any_recognized = any(r['name'] != "Unknown" for r in results) if results else False
                control_led(any_recognized)
                
            except Exception as e:
                print(f"Proc Error: {e}")
        else:
            time.sleep(0.01) # Prevent CPU hogging

def main():
    global latest_frame, is_running, new_frame_available
    
    cap = VideoStream(RTSP_URL).start()
    time.sleep(2.0)
    
    # Start the background worker
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    
    print("System Live. Press 'q' to quit.")

    while True:
        frame = cap.read()
        if frame is None: continue
        
        # Update frame for the processing thread
        latest_frame = frame
        new_frame_available = True
        
        # Draw the latest known results (non-blocking)
        for result in cached_results:
            x_min, y_min, x_max, y_max = result['bbox']
            name, conf = result['name'], result['confidence']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{name} {conf:.2f}" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Face Recognition (High Speed)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    is_running = False
    t.join()
    control_led(False)
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()