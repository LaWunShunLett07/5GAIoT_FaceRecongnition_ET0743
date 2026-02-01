# registration_auto_threaded.py
# Auto-register: ask name once -> auto capture/save/register N samples
# q = quit anytime

import os
import cv2
import requests
import datetime
import threading
import queue
import time
from options import Options
from imutils.video import VideoStream

opts = Options()

# ✅ make sure image directory exists
if not getattr(opts, "imageDir", "") or opts.imageDir.strip() == "":
    opts.imageDir = "images"
os.makedirs(opts.imageDir, exist_ok=True)

# ---------- HTTP helpers ----------
def register_face_bytes(image_bytes: bytes, user_id: str) -> dict:
    """Register using bytes (no need to re-read from disk)."""
    try:
        return requests.post(
            opts.endpoint("vision/face/register"),
            files={"image": image_bytes},
            data={"userid": user_id},
            timeout=5
        ).json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def detect_faces_bytes(image_bytes: bytes) -> dict:
    """Detect faces using CodeProject.AI."""
    try:
        return requests.post(
            opts.endpoint("vision/face"),
            files={"image": image_bytes},
            timeout=3
        ).json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "predictions": []}

# ---------- Worker thread ----------
def detector_worker(in_q: queue.Queue, out_q: queue.Queue, stop_evt: threading.Event):
    """
    Reads latest jpg bytes from in_q and outputs predictions to out_q.
    maxsize=1 avoids backlog.
    """
    while not stop_evt.is_set():
        try:
            jpg_bytes = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        resp = detect_faces_bytes(jpg_bytes)
        preds = resp.get("predictions", []) if isinstance(resp, dict) else []
        err = resp.get("error") if isinstance(resp, dict) else None

        # keep only latest output
        try:
            while True:
                out_q.get_nowait()
        except queue.Empty:
            pass

        out_q.put({"predictions": preds, "error": err})

# ---------- Main ----------
def main():
    # 1) Ask for name
    user_name = input("Enter the name to register (e.g., Amy): ").strip()
    if not user_name:
        print("[ERROR] Name cannot be empty.")
        return

    # Optional: ask how many samples
    try:
        target_samples = int(input("How many samples to capture? (recommended 10): ").strip() or "10")
    except ValueError:
        target_samples = 10

    # gap between captures so it doesn't spam (seconds)
    CAPTURE_COOLDOWN = 0.8

    print("\n=== Auto Registration Mode (Threaded) ===")
    print(f"Name: {user_name}")
    print(f"Target samples: {target_samples}")
    print("Show ONLY ONE face clearly. The system will auto-capture.")
    print("Press q to quit.\n")

    cap = VideoStream(0).start()

    # speed settings
    skip_frame = 3
    frame_index = 0

    # thread queues
    in_q = queue.Queue(maxsize=1)
    out_q = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    t = threading.Thread(target=detector_worker, args=(in_q, out_q, stop_evt), daemon=True)
    t.start()

    latest_preds = []
    latest_error = None

    saved_count = 0
    last_capture_time = 0.0

    while True:
        frame = cap.read()
        if frame is None:
            print("Camera closed")
            break

        # resize (faster AI)
        frame = cv2.resize(frame, (640, 480))
        raw_frame = frame.copy()  # clean frame for saving/registering

        frame_index += 1

        # Send frame to detector periodically
        if skip_frame <= 1 or (frame_index % skip_frame == 0):
            ok, jpg = cv2.imencode(".jpg", raw_frame)
            if ok:
                try:
                    in_q.put_nowait(jpg.tobytes())
                except queue.Full:
                    try:
                        in_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        in_q.put_nowait(jpg.tobytes())
                    except queue.Full:
                        pass

        # Read detection result if available
        try:
            result = out_q.get_nowait()
            latest_preds = result.get("predictions", [])
            latest_error = result.get("error")
        except queue.Empty:
            pass

        num_faces = len(latest_preds)

        # UI overlay
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Registering: {user_name} | Saved: {saved_count}/{target_samples} | q=Quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw boxes
        for p in latest_preds:
            cv2.rectangle(frame,
                          (p["x_min"], p["y_min"]),
                          (p["x_max"], p["y_max"]),
                          (0, 0, 255), 2)

        # Messages
        if latest_error:
            cv2.putText(frame, f"Detect error: {latest_error}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        elif num_faces != 1:
            cv2.putText(frame, "Show ONLY ONE face clearly!", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "OK: 1 face detected (auto capture soon)", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Image Viewer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit.")
            break

        # 2) Auto-capture when exactly 1 face detected + cooldown passed
        now = time.time()
        if (num_faces == 1) and (saved_count < target_samples) and (now - last_capture_time >= CAPTURE_COOLDOWN):
            last_capture_time = now

            # Save clean image
            filename = f"{user_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{saved_count+1}.jpg"
            abs_path = os.path.join(opts.imageDir, filename)
            cv2.imwrite(abs_path, raw_frame)
            print(f"[SAVE] {abs_path}")

            # Register in background (doesn't freeze UI)
            ok2, jpg2 = cv2.imencode(".jpg", raw_frame)
            if ok2:
                img_bytes = jpg2.tobytes()

                def do_register(user, bytes_):
                    resp = register_face_bytes(bytes_, user)
                    print(f"[REGISTER] {user} -> {resp}")

                threading.Thread(target=do_register, args=(user_name, img_bytes), daemon=True).start()

            saved_count += 1

            if saved_count >= target_samples:
                print(f"[DONE] Captured {saved_count} samples for {user_name}.")
                break

    # cleanup
    stop_evt.set()
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()