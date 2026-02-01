
# recognition_telegram.py
import os
import cv2
import requests
import datetime
import threading
import queue
import time
from options import Options
from imutils.video import VideoStream

# ==============================================================================
# ⚠️ LAB ALERT: NETWORK & TELEGRAM SETTINGS
# ==============================================================================
# 1. NETWORK (IP ADDRESS) - Change if in Lab
RTSP_URL = "rtsp://192.168.0.10:8554/feeder"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# 2. TELEGRAM SETTINGS (FILL THESE IN!)
TELEGRAM_BOT_TOKEN = "8504218202:AAF1yt3Aq8PmfkStI1gQhY7WaFP9qdSW9sI"  # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = "8032542939"      # e.g. "123456789"
# ==============================================================================

opts = Options()
last_telegram_time = 0  
TELEGRAM_COOLDOWN = 15  # Seconds to wait before sending another alert

if not getattr(opts, "imageDir", "") or opts.imageDir.strip() == "":
    opts.imageDir = "images"
os.makedirs(opts.imageDir, exist_ok=True)

# Settings
SKIP_FRAME = 2
MIN_RECOG_CONFIDENCE = 0.60
PAD = 20

# We will resize the FINAL display to this size so it looks big on screen
DISPLAY_SIZE = (800, 800) 

def send_telegram_alert(image_bytes):
    """Sends the captured image to Telegram in a background thread"""
    def _send():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': image_bytes}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': "⚠️ Alert: Unknown Face Detected!"}
            requests.post(url, files=files, data=data, timeout=10)
            print("✅ Telegram alert sent!")
        except Exception as e:
            print(f"❌ Telegram failed: {e}")
    
    # Run in thread so it doesn't freeze the video
    threading.Thread(target=_send, daemon=True).start()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def detect_faces(jpg_bytes: bytes) -> dict:
    try:
        return requests.post(
            opts.endpoint("vision/face"),
            files={"image": jpg_bytes},
            timeout=3
        ).json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "predictions": []}

def recognize_face(jpg_bytes: bytes) -> dict:
    try:
        return requests.post(
            opts.endpoint("vision/face/recognize"),
            files={"image": jpg_bytes},
            data={"min_confidence": MIN_RECOG_CONFIDENCE},
            timeout=4
        ).json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "predictions": []}

def worker(in_q: queue.Queue, out_q: queue.Queue, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            item = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        raw_frame, h, w, frame_id = item
        ok, jpg = cv2.imencode(".jpg", raw_frame)
        if not ok: continue

        # Detect
        det = detect_faces(jpg.tobytes())
        detections = det.get("predictions", []) if isinstance(det, dict) else []
        err = det.get("error") if isinstance(det, dict) else None

        results = []
        unknown_found = False

        if not err and detections:
            for d in detections:
                x1 = clamp(int(d["x_min"]), 0, w - 1)
                y1 = clamp(int(d["y_min"]), 0, h - 1)
                x2 = clamp(int(d["x_max"]), 0, w - 1)
                y2 = clamp(int(d["y_max"]), 0, h - 1)

                # Add padding for better recognition context
                x1p = clamp(x1 - PAD, 0, w - 1)
                y1p = clamp(y1 - PAD, 0, h - 1)
                x2p = clamp(x2 + PAD, 0, w - 1)
                y2p = clamp(y2 + PAD, 0, h - 1)

                face_crop = raw_frame[y1p:y2p, x1p:x2p]
                if face_crop.size == 0 or (x2p - x1p) < 40:
                    results.append({"box": (x1, y1, x2, y2), "label": "Unknown"})
                    unknown_found = True
                    continue

                ok2, face_jpg = cv2.imencode(".jpg", face_crop)
                if not ok2: continue

                # Recognize
                rec = recognize_face(face_jpg.tobytes())
                preds = rec.get("predictions", []) if isinstance(rec, dict) else []

                label = "Unknown"
                if preds:
                    top = preds[0]
                    userid = top.get("userid", "unknown")
                    conf = float(top.get("confidence", top.get("score", 0.0)) or 0.0)
                    if userid and str(userid).lower() != "unknown" and conf >= MIN_RECOG_CONFIDENCE:
                        label = f"{userid} ({conf:.2f})"
                    else:
                        unknown_found = True
                else:
                    unknown_found = True

                results.append({"box": (x1, y1, x2, y2), "label": label})

        payload = {
            "frame_id": frame_id,
            "error": err,
            "results": results,
            "unknown_found": unknown_found,
            "raw_frame_copy": raw_frame if unknown_found else None
        }

        try:
            while True:
                out_q.get_nowait()
        except queue.Empty:
            pass
        out_q.put(payload)

def main():
    global last_telegram_time

    print(f"Connecting to RTSP Stream: {RTSP_URL}...")
    cap = VideoStream(RTSP_URL).start()
    time.sleep(1.0) 

    stop_evt = threading.Event()
    in_q = queue.Queue(maxsize=1)
    out_q = queue.Queue(maxsize=1)

    t = threading.Thread(target=worker, args=(in_q, out_q, stop_evt), daemon=True)
    t.start()

    frame_index = 0
    latest = {"frame_id": -1, "error": None, "results": [], "unknown_found": False}

    while True:
        frame = cap.read()
        if frame is None:
            continue

        # 1. SQUARE CROP LOGIC (Maximized)
        # We grab the largest possible square from the center
        h, w = frame.shape[:2]
        min_dim = min(h, w) # This will be the size of the square
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        
        # Crop the square
        square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Resize to standard size for AI processing (keep it fast)
        ai_frame = cv2.resize(square_frame, (640, 640))
        
        frame_index += 1

        # Send to AI worker
        if SKIP_FRAME <= 1 or (frame_index % SKIP_FRAME == 0):
            try:
                # We send the SQUARE frame to AI, so boxes match the view
                in_q.put_nowait((ai_frame, 640, 640, frame_index))
            except queue.Full:
                try:
                    in_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    in_q.put_nowait((ai_frame, 640, 640, frame_index))
                except queue.Full:
                    pass

        # Get results
        try:
            latest = out_q.get_nowait()
        except queue.Empty:
            pass

        # 2. TELEGRAM ALERT LOGIC
        if latest.get("unknown_found"):
            current_time = time.time()
            if (current_time - last_telegram_time) > TELEGRAM_COOLDOWN:
                print("🚨 Unknown detected! Sending Telegram...")
                
                # Get the frame that triggered the alert
                alert_frame = latest.get("raw_frame_copy")
                if alert_frame is not None:
                    ok, alert_jpg = cv2.imencode(".jpg", alert_frame)
                    if ok:
                        send_telegram_alert(alert_jpg.tobytes())
                        last_telegram_time = current_time

        # Draw UI on the AI frame
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(ai_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        if latest.get("error"):
            cv2.putText(ai_frame, f"Error: {latest['error']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for item in latest.get("results", []):
            x1, y1, x2, y2 = item["box"]
            label = item["label"]
            color = (0, 255, 0) # Green for known
            if "Unknown" in label:
                color = (0, 0, 255) # Red for unknown
            
            cv2.rectangle(ai_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(ai_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 3. BIG DISPLAY
        # Resize the square frame to be BIG (e.g. 800x800) so it's not "too small"
        final_display = cv2.resize(ai_frame, DISPLAY_SIZE)
        cv2.imshow("Recognition (Big Square)", final_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stop capturing...")
            break

    stop_evt.set()
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()