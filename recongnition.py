#!/usr/bin/env python3
import os
import cv2
import requests
import datetime
from options import Options
from imutils.video import VideoStream
import socket
import time

# Configuration
PI_IP = "192.168.55.100"
RTSP_URL = "rtsp://172.28.182.91:8554/feeder"
UDP_PORT = 5005
MIN_CONFIDENCE = 0.6
SKIP_FRAMES = 5
RESIZE_WIDTH = 640

# Socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
opts = Options()

# State tracking
current_led_state = None


def control_led(should_be_on):
    """Control LED only when state changes."""
    global current_led_state
    if current_led_state != should_be_on:
        try:
            sock.sendto(b'LED_ON' if should_be_on else b'LED_OFF', (PI_IP, UDP_PORT))
            current_led_state = should_be_on
            print(f"LED: {'ON' if should_be_on else 'OFF'}")  # Debug
        except Exception as e:
            print(f"LED Error: {e}")


def detect_and_recognize_fast(frame):
    """
    Combined detection and recognition in one go.
    Resizes frame before sending to API for speed.
    """
    try:
        # Resize frame for faster API processing
        small_frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))
        
        # Encode with lower quality for speed
        _, encoded = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_bytes = encoded.tobytes()
        
        # Detect faces
        detect_response = requests.post(
            opts.endpoint("vision/face"),
            files={"image": img_bytes},
            timeout=1
        ).json()
        
        predictions = detect_response.get('predictions', [])
        if not predictions:
            return []
        
        # Scale coordinates back to original frame size
        scale_x = frame.shape[1] / RESIZE_WIDTH
        scale_y = frame.shape[0] / (frame.shape[0] * RESIZE_WIDTH / frame.shape[1])
        
        results = []
        for pred in predictions:
            # Scale coordinates
            x_min = int(pred['x_min'] * scale_x)
            y_min = int(pred['y_min'] * scale_y)
            x_max = int(pred['x_max'] * scale_x)
            y_max = int(pred['y_max'] * scale_y)
            
            # Crop face from original frame
            face = frame[y_min:y_max, x_min:x_max]
            if face.size == 0:
                continue
            
            # Recognize this face
            _, face_encoded = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            recog_response = requests.post(
                opts.endpoint("vision/face/recognize"),
                files={"image": face_encoded.tobytes()},
                data={"min_confidence": MIN_CONFIDENCE},
                timeout=1
            ).json()
            
            # Extract result
            recognized_id = "Unknown"
            confidence = 0
            if recog_response and "predictions" in recog_response and recog_response["predictions"]:
                user = recog_response["predictions"][0]
                recognized_id = user.get("userid", "Unknown")
                confidence = user.get("confidence", 0)
                if recognized_id == "unknown":
                    recognized_id = "Unknown"
            
            results.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'name': recognized_id,
                'confidence': confidence
            })
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    print("Starting Face Recognition...")
    print(f"Stream: {RTSP_URL}")
    print("Press 'q' to quit\n")
    
    # Start stream
    cap = VideoStream(RTSP_URL).start()
    time.sleep(2)
    
    frame_count = 0
    cached_results = []
    
    while True:
        # Grab frame
        frame = cap.read()
        if frame is None:
            continue
        
        frame_count += 1
        
        # Process only every Nth frame
        if frame_count % SKIP_FRAMES == 0:
            cached_results = detect_and_recognize_fast(frame)
            
            # LED Control - FIXED
            if cached_results:
                # We have detected faces
                any_recognized = any(r['name'] != "Unknown" for r in cached_results)
                control_led(any_recognized)
            else:
                # No faces detected - turn LED off
                control_led(False)
        
        # Draw results
        for result in cached_results:
            x_min, y_min, x_max, y_max = result['bbox']
            name = result['name']
            conf = result['confidence']
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label with background
            label = f"{name}" if name == "Unknown" else f"{name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x_min, y_min - label_size[1] - 10), 
                         (x_min + label_size[0], y_min), color, -1)
            cv2.putText(frame, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Face Recognition', frame)
        
        # Check quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    control_led(False)
    cap.stop()
    cv2.destroyAllWindows()
    sock.close()
    print("Done!")


if __name__ == "__main__":
    main()