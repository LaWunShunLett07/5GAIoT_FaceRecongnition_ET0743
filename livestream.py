import cv2
import imutils

rtsp_url = "rtsp://100.103.111.233:8554/feeder"

def main():
    # Open RTSP using FFmpeg backend (best chance for RTSP stability)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # Reduce internal buffering (works on many builds; harmless if ignored)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot open RTSP stream.")
        return

    while True:
        # Flush old frames so latency doesn't build up
        for _ in range(5):   # increase to 10 if still laggy
            cap.grab()

        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Resize (optional). If you want even lower latency, use width=640.
        frame = imutils.resize(frame, width=1280)

        cv2.imshow("Cam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()