import cv2
import time

from webcam import Webcam

if __name__ == '__main__':
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("STARTED CAM")

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        cv2.imshow('live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(
            f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} fps: {round(1 / (time.time() - start), 5)}, {cap.get(cv2.CAP_PROP_FPS)}")
    print("END")
    cap.release()
    cv2.destroyAllWindows()