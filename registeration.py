#!/usr/bin/env python3
"""
registration_rtsp_auto_threaded.py

- Auto registration: ask name once, capture N samples automatically
- Uses threaded detection so UI doesn't freeze
- Works NOW on laptop webcam
- Later: switch to Raspberry Pi RTSP over 5G by setting USE_WEBCAM=False and PI_IP

Keys:
  q = quit
"""

import os
import cv2
import time
import queue
import threading
import datetime
import requests
from options import Options

# -------------------- SETTINGS --------------------
USE_WEBCAM = True               # ✅ NOW: True (webcam). Later: False (RTSP)
PI_IP = "172.23.28.195"         # later replace with your Pi's IP
RTSP_URL = f"rtsp://{PI_IP}:8554/feeder"

# Low-latency RTSP hints (used only when USE_WEBCAM=False)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0"
)

# Speed tuning
RESIZE_W, RESIZE_H = 640, 480
JPEG_QUALITY = 65
DETECT_EVERY_N_FRAMES = 3

# Registration tuning
DEFAULT_SAMPLES = 10
CAPTURE_COOLDOWN = 0.8  # seconds between captures

opts = Options()

# Ensure imageDir exists
if not getattr(opts, "imageDir", "") or opts.imageDir.strip() == "":
    opts.imageDir = "images"
os.makedirs(opts.imageDir, exist_ok=True)

session = requests.Session()

def post_json(url: str, *, files=None, data=None, timeout=2.0):
    try:
        r = session.post(url, files=files, data=data, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def encode_jpg(frame):
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return ok, (jpg.tobytes() if ok else b"")

def detect_faces_bytes(jpg_bytes: bytes) -> dict:
    return post_json(opts.endpoint("vision/face"), files={"image": jpg_bytes}, timeout=1.2)

def register_face_bytes(jpg_bytes: bytes, user_id: str) -> dict:
    return post_json(
        opts.endpoint("vision/face/register"),
        files={"image": jpg_bytes},
        data={"userid": user_id},
        timeout=2.5
    )

def detector_worker(in_q: queue.Queue, out_q: queue.Queue, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            jpg_bytes = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        det = detect_faces_bytes(jpg_bytes)
        preds = det.get("predictions", []) if isinstance(det, dict) else []
        err = det.get("error") if isinstance(det, dict) else None

        # keep only newest output
        try:
            while True:
                out_q.get_nowait()
        except queue.Empty:
            pass
        out_q.put({"predictions": preds, "error": err})

def open_capture():
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    user_name = input("Enter the name to register (e.g., Amy): ").strip()
    if not user_name:
        print("[ERROR] Name cannot be empty.")
        return

    try:
        target_samples = int(input(f"How many samples? (default {DEFAULT_SAMPLES}): ").strip() or str(DEFAULT_SAMPLES))
    except ValueError:
        target_samples = DEFAULT_SAMPLES

    print("\n=== Auto Registration (Threaded) ===")
    print("Source:", "Webcam (0)" if USE_WEBCAM else RTSP_URL)
    print(f"Name: {user_name}")
    print(f"Target samples: {target_samples}")
    print("Show ONLY ONE face clearly. Auto-capture will happen.")
    print("q = quit\n")

    cap = open_capture()
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    stop_evt = threading.Event()
    in_q = queue.Queue(maxsize=1)
    out_q = queue.Queue(maxsize=1)

    t = threading.Thread(target=detector_worker, args=(in_q, out_q, stop_evt), daemon=True)
    t.start()

    frame_id = 0
    latest_preds = []
    latest_err = None
    saved = 0
    last_capture = 0.0

    while True:
        if not cap.grab():
            continue
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        raw = frame.copy()
        frame_id += 1

        # send to detector
        if DETECT_EVERY_N_FRAMES <= 1 or (frame_id % DETECT_EVERY_N_FRAMES == 0):
            ok, jpg_bytes = encode_jpg(raw)
            if ok:
                try:
                    in_q.put_nowait(jpg_bytes)
                except queue.Full:
                    try:
                        in_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        in_q.put_nowait(jpg_bytes)
                    except queue.Full:
                        pass

        # read latest detection
        try:
            out = out_q.get_nowait()
            latest_preds = out.get("predictions", [])
            latest_err = out.get("error")
        except queue.Empty:
            pass

        num_faces = len(latest_preds)

        # UI
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Registering: {user_name} | Saved: {saved}/{target_samples} | q=Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # boxes
        for p in latest_preds:
            cv2.rectangle(frame, (p["x_min"], p["y_min"]), (p["x_max"], p["y_max"]), (0, 0, 255), 2)

        # message
        if latest_err:
            cv2.putText(frame, f"Detect error: {latest_err}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        elif num_faces != 1:
            cv2.putText(frame, "Show ONLY ONE face clearly!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "OK: 1 face detected (auto capture soon)", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Registration (Threaded)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # auto capture + async register
        now = time.time()
        if num_faces == 1 and saved < target_samples and (now - last_capture) >= CAPTURE_COOLDOWN:
            last_capture = now

            filename = f"{user_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{saved+1}.jpg"
            path = os.path.join(opts.imageDir, filename)
            cv2.imwrite(path, raw)
            print(f"[SAVE] {path}")

            ok2, jpg2 = encode_jpg(raw)
            if ok2:
                def do_register(name, bytes_):
                    resp = register_face_bytes(bytes_, name)
                    print(f"[REGISTER] {name} -> {resp}")
                threading.Thread(target=do_register, args=(user_name, jpg2), daemon=True).start()

            saved += 1

            if saved >= target_samples:
                print(f"[DONE] Captured {saved} samples for {user_name}.")
                break

    stop_evt.set()
    cap.release()
    cv2.destroyAllWindows()
    session.close()

if __name__ == "__main__":
    main()