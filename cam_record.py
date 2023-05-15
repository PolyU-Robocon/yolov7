import cv2
import os
import time

from datetime import datetime


if __name__ == '__main__':
    cnt = 0
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not os.path.exists("./images"):
        os.mkdir("images")
    ret, frame = cap.read()
    print("STARTED CAM")
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        cv2.imshow('live', frame)
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break
        print(
            f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} fps: {round(1 / (time.time() - start), 5)}, {cap.get(cv2.CAP_PROP_FPS)}")
        cv2.imwrite(f"./images/{datetime.now().strftime('%m%d%Y_%H%M%S')}_{cnt}.jpg", frame)
    print("END")
    cap.release()
    cv2.destroyAllWindows()