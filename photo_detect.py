import cv2
import detect_api
import time

if __name__ == '__main__':
    import os, shutil
    for file in os.scandir("./runs/detect"):
        if file.name.startswith("test"):
            shutil.rmtree(file.path)
            print(f"removed {file.path}")
    weight = "runs/train/210-416-48-1000-tiny/weights/best.pt"
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    detect = detect_api.Detect(weight, conf_thres, iou_thres, classes)
    size = 640
    detect.init_size(size)
    while True:
        a = input()
        path = "../test/images/" + a
        start = time.time()
        if os.path.exists(path):
            # Read image
            #print(path)
            im0s = cv2.imread(path)  # BGR
            #print(im0s)
            detect.detect_image(im0s, size)
        else:
            print("not find")
        print(f"time :{time.time() - start}, fps: {1 / (time.time() - start)}")