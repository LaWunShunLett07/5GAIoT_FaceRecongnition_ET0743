# recognition.py (MULTI-FACE)
# Recognize multiple faces in one frame using CodeProject.AI
# q = quit

import os
import cv2
import requests
import datetime
from options import Options
from imutils.video import VideoStream

opts = Options()
rtsp_url = "rtsp://169.254.15.175:8554/feeder"
# Fix imageDir empty issue
if not getattr(opts, "imageDir", "") or opts.imageDir.strip() == "":
    opts.imageDir = "images"
os.makedirs(opts.imageDir, exist_ok=True)

SKIP_FRAME = 5
MIN_RECOG_CONFIDENCE = 0.70   # raise to 0.80 if mislabels happen

def recognize_face_bytes(image_bytes: bytes) -> dict:
    """Send bytes (cropped face) to server for recognition."""
    try:
        return requests.post(
            opts.endpoint("vision/face/recognize"),
            files={"image": image_bytes},
            data={"min_confidence": MIN_RECOG_CONFIDENCE}
        ).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    cap = VideoStream(rtsp_url).start()
    frame_index = 0

    while True:
        # quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stop capturing...")
            break

        frame = cap.read()
        if frame is None:
            print("Camera closed")
            break

        frame_index += 1
        if SKIP_FRAME > 1 and (frame_index % SKIP_FRAME != 0):
            continue

        raw_frame = frame.copy()
        h, w = raw_frame.shape[:2]

        # Timestamp (display)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1, cv2.LINE_AA)

        # Detect faces (get bounding boxes)
        ok, jpg = cv2.imencode(".jpg", raw_frame)
        if not ok:
            cv2.imshow("Image Viewer", frame)
            continue

        try:
            detect_resp = requests.post(
                opts.endpoint("vision/face"),
                files={"image": jpg}
            ).json()
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        detections = detect_resp.get("predictions", [])

        if len(detections) == 0:
            cv2.putText(frame, "No face detected", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Image Viewer", frame)
            continue

        # For each detected face -> crop -> recognize -> label
        for d in detections:
            x1 = clamp(int(d["x_min"]), 0, w - 1)
            y1 = clamp(int(d["y_min"]), 0, h - 1)
            x2 = clamp(int(d["x_max"]), 0, w - 1)
            y2 = clamp(int(d["y_max"]), 0, h - 1)

            # Add padding (helps recognition)
            pad = 20
            x1p = clamp(x1 - pad, 0, w - 1)
            y1p = clamp(y1 - pad, 0, h - 1)
            x2p = clamp(x2 + pad, 0, w - 1)
            y2p = clamp(y2 + pad, 0, h - 1)

            face_crop = raw_frame[y1p:y2p, x1p:x2p]

            # If crop too small, skip
            if face_crop.size == 0 or (x2p - x1p) < 40 or (y2p - y1p) < 40:
                continue

            # Encode crop to jpg bytes
            ok2, face_jpg = cv2.imencode(".jpg", face_crop)
            if not ok2:
                continue

            # Recognize this face
            result = recognize_face_bytes(face_jpg.tobytes())
            preds = result.get("predictions", [])

            # Default label
            label = "Unknown"
            conf = 0.0

            if len(preds) > 0:
                top = preds[0]
                userid = top.get("userid", "unknown")
                conf = float(top.get("confidence", top.get("score", 0.0)) or 0.0)

                if userid and str(userid).lower() != "unknown" and conf >= MIN_RECOG_CONFIDENCE:
                    label = f"{userid} ({conf:.2f})"

            # Draw rectangle + label for THIS face only
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image Viewer", frame)

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()