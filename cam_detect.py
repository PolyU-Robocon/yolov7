import cv2
import detect_api
import time
import threading

cap = None

def camera_start(cam):
    global cap
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print(f"CAMERA {cam} Initialized")


if __name__ == '__main__':
    cam = 0
    cam_thread = threading.Thread(target=camera_start, args=(cam, ))
    cam_thread.start()
    import os, shutil
    for file in os.scandir("./runs/detect"):
        if file.name.startswith("test"):
            shutil.rmtree(file.path)
            print(f"removed {file.path}")
    now = time.time()   
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)

    weight = "runs/train/11k-1440-300-tiny/weights/best.pt"
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    detect = detect_api.Detect(weight, conf_thres, iou_thres, classes, view_img=True)
    size = 736  
    detect.init_size(size)

    cam_thread.join()
    print(f"Initialized camrea and Yolov7 with {round((time.time() - now), 5)}s")

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #print(f"init time{time.time() - start}")
        result = detect.detect_image(frame, size)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        used_time = round((time.time() - start), 5)
        print(f"resolution: {width}x{height}detected {len(result)} object, used time: {used_time}s, fps: {round(1 / used_time, 5)}")
    cap.release()
    cv2.destroyAllWindows()
