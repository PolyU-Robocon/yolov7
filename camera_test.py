import cv2
import time

from webcam import Webcam


if __name__ == '__main__':
    webcam = Webcam(width=2560, height=1440)
    webcam.cam_init()
    webcam.start()
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    start = time.time()
    while True:
        if not webcam.used:
            frame = webcam.read()
            cv2.imshow('live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            w, h, fps = webcam.get_wh_fps()
            print(f"{w}x{h} fps: {round(1 / (time.time() - start), 5)}, {fps}")
            start = time.time()
        else:
            time.sleep(0.00001)
    print("END")
    webcam.stop()
    cap.release()
    cv2.destroyAllWindows()
