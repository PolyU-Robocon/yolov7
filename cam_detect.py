import cv2
import detect_api
import time
import threading

from webcam import Webcam

webcam = Webcam(width=1920, height=1080)


def camera_start():
    webcam.cam_init()


if __name__ == '__main__':
    cam = 0
    cam_thread = threading.Thread(target=camera_start)
    cam_thread.start()
    now = time.time()

    weight = "runs/train/11k-1440-300-tiny/weights/best.pt"
    conf_thres = 0.25
    iou_thres = 0.45
    size = 1024
    classes = None
    detect = detect_api.Detect(weight, conf_thres, iou_thres, classes, view_img=True)
    detect.init_size(size)

    cam_thread.join()
    webcam.start()
    print(f"Initialized camera and Yolo v7 with {round((time.time() - now), 5)}s")
    start = time.time()
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    while True:
        if not webcam.used:
            frame = webcam.read()
            result = detect.detect_image(frame, size)
            w, h, fps = webcam.get_wh_fps()
            used_time = time.time() - start
            fps_cal = round(1 / used_time, 5)
            print(f"resolution: {w}x{h} detected {len(result)} object with {used_time}s, fps: cal {fps_cal} cam {fps}")
            start = time.time()
        else:
            time.sleep(0.00001)
    webcam.cam.release()
    cv2.destroyAllWindows()
