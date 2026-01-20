# Import necessary modules and packages
import os
import cv2
import requests
import datetime
from options import Options
from imutils.video import VideoStream

opts = Options()
rtsp_url = "rtsp://169.254.15.175:8554/feeder"
# ✅ make sure image directory exists
if not opts.imageDir or opts.imageDir.strip() == "":
    opts.imageDir = "images"
os.makedirs(opts.imageDir, exist_ok=True)

# Two users you want to register
USERS = {
    "1": "Amy",
    "2": "Shun",
    "3": "Shafiqa"
}


def register_face(abs_img_path, user_id):
    """Send image + userid to CodeProject.AI to register."""
    image_data = open(abs_img_path, "rb").read()
    try:
        response = requests.post(
            opts.endpoint("vision/face/register"),
            files={"image": image_data},
            data={"userid": user_id}
        ).json()
        print(f"[REGISTER] {user_id} -> {response}")
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

def main():
    print("=== Registration Mode ===")
    print("Press 1 = Amy")
    print("Press 2 = Shun")
    print("Press 3 = Shafiqa")
    print("Press r = Register selected person")
    print("Press q = Quit")
    print("IMPORTANT: show ONLY ONE face clearly when registering.\n")

    cap = VideoStream(rtsp_url).start()

    frame_index = 0
    skip_frame = 5
    selected_user = None

    while True:
        frame = cap.read()
        if frame is None:
            print("Camera closed")
            break

        frame_index += 1
        if skip_frame > 1 and frame_index % skip_frame != 0:
            cv2.imshow("Image Viewer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # UI text
        info = f"Selected: {selected_user if selected_user else 'None'} | 1=Amy 2=Shun 3=Shafiqa r=Register q=Quit"

        cv2.putText(frame, info, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # detect faces
        retval, new_frame = cv2.imencode(".jpg", frame)
        try:
            response = requests.post(
                opts.endpoint("vision/face"),
                files={"image": new_frame}
            ).json()
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        predictions = response.get("predictions", [])
        num_faces = len(predictions)

        # draw boxes
        for p in predictions:
            frame = cv2.rectangle(
                frame,
                (p["x_min"], p["y_min"]),
                (p["x_max"], p["y_max"]),
                (0, 0, 255), 2
            )

        if num_faces != 1:
            cv2.putText(frame, "Show ONLY ONE face clearly!",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image Viewer", frame)

        key = cv2.waitKey(1) & 0xFF

        # quit
        if key == ord("q"):
            print("Stop capturing...")
            break

        # select user
        if key in (ord("1"), ord("2"), ord("3")):
            selected_user = USERS[chr(key)]
            print(f"[SELECT] {selected_user}")

        # register selected user
        if key == ord("r"):
            if not selected_user:
                print("[WARN] Please select user first (press 1 or 2 or 3).")
                continue

            if num_faces != 1:
                print("[WARN] Please ensure ONLY ONE face is in view before registering.")
                continue

            # save image into opts.imageDir
            filename = f"{selected_user}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            abs_path = os.path.join(opts.imageDir, filename)
            cv2.imwrite(abs_path, frame)
            print(f"[SAVE] {abs_path}")

            # register to CodeProject.AI
            register_face(abs_path, selected_user)
            print("[OK] Registered one sample. (Do 5-10 samples per person)\n")

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()