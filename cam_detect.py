import cv2
import detect_api
import time

if __name__ == '__main__':
    import os, shutil
    for file in os.scandir("./runs/detect"):
        if file.name.startswith("test"):
            shutil.rmtree(file.path)
            print(f"removed {file.path}")
    weight = "runs/train/210-416-9-1000/weights/best.pt"
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    detect = detect_api.Detect(weight, conf_thres, iou_thres, classes, view_img=True)
    size = 640
    detect.init_size(size)

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #print(f"init time{time.time() - start}")
        result = detect.detect_image(frame, size)
        print(f"detected {len(result)} object, used time: {round((time.time() - start), 5)}s, fps: {round(1 / (time.time() - start), 5)}")
cap.release()
cv2.destroyAllWindows()