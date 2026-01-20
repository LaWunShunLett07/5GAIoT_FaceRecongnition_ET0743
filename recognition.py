# recognition.py
# Recognize faces (Amy, Shun, Shafiqa, etc.) already registered in CodeProject.AI
# Controls:
#   q = quit

import os
import cv2
import requests
import datetime
from options import Options
from imutils.video import VideoStream

opts = Options()
rtsp_url = "rtsp://169.254.15.175:8554/feeder"
# ✅ Fix imageDir empty issue
if not getattr(opts, "imageDir", "") or opts.imageDir.strip() == "":
    opts.imageDir = "images"
os.makedirs(opts.imageDir, exist_ok=True)

# Tuning
SKIP_FRAME = 5
MIN_RECOG_CONFIDENCE = 0.60  # increase to 0.70–0.85 if mislabels still happen

def recognize_face(img_path: str) -> dict:
    """Send an image (inside opts.imageDir) to CodeProject.AI for recognition."""
    filepath = os.path.join(opts.imageDir, img_path)

    with open(filepath, "rb") as f:
        image_data = f.read()

    try:
        return requests.post(
            opts.endpoint("vision/face/recognize"),
            files={"image": image_data},
            data={"min_confidence": MIN_RECOG_CONFIDENCE}
        ).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

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

        # Keep a clean copy for recognition (no rectangles/text drawn)
        raw_frame = frame.copy()

        # Timestamp overlay (display only)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1, cv2.LINE_AA)

        # Detect faces (bounding boxes)
        ok, new_frame = cv2.imencode(".jpg", raw_frame)
        if not ok:
            cv2.imshow("Image Viewer", frame)
            continue

        try:
            detect_resp = requests.post(
                opts.endpoint("vision/face"),
                files={"image": new_frame}
            ).json()
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        detections = detect_resp.get("predictions", [])
        num_faces = len(detections)

        if num_faces >= 1:
            # Save the clean frame for recognition
            save_path = os.path.join(opts.imageDir, "image.jpg")
            cv2.imwrite(save_path, raw_frame)

            # Recognize (who is it?)
            result = recognize_face("image.jpg")
            recog_list = result.get("predictions", [])

            # Draw boxes and label each detection with the best match (if any)
            # If CodeProject.AI returns multiple predictions, we take the top one.
            best_userid = "Unknown"
            best_conf = 0.0
            if len(recog_list) > 0:
                top = recog_list[0]
                best_userid = top.get("userid", "Unknown")
                best_conf = float(top.get("confidence", top.get("score", 0.0)) or 0.0)

            for d in detections:
                x1, y1, x2, y2 = d["x_min"], d["y_min"], d["x_max"], d["y_max"]

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Label logic
                if best_userid and str(best_userid).lower() != "unknown" and best_conf >= MIN_RECOG_CONFIDENCE:
                    label = f"{best_userid} ({best_conf:.2f})"
                    print(f"Recognized as: {best_userid} (confidence={best_conf:.2f})")
                else:
                    label = "Unknown"
                    # (optional) print only sometimes to reduce spam
                    print("No confident match")

                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            print("Make sure you are showing your face CLEARLY...")

        cv2.imshow("Image Viewer", frame)

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()