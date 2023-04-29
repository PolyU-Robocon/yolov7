import threading
import time

import cv2
import numpy as np
from scipy import stats

import detect_api
from colorize import colorize
from gui import GUI
from webcam import Webcam

DEBUG = False
webcam = Webcam(width=1920, height=1080, k4a=True)
fuck = 0


def camera_start():
    webcam.cam_init()


def xywh_to_xxyy(xywh, w, h, offset_w=0.5, offset_h=0.8):
    box_w = xywh[2] * w / 2 * offset_w
    box_h = xywh[3] * h / 2 * offset_h
    x = xywh[0] * w
    y = xywh[1] * h
    result = [x - box_w, x + box_w, y - box_h, y + box_h]
    for i in range(len(result)):
        result[i] = round(result[i])
    return result


def img_xxyy3(img, depth, color_depth, xxyy):
    min_dis = 9999
    for h in range(xxyy[2], xxyy[3]):
        w = depth[h][xxyy[0]:xxyy[1]]
        avg = np.average(w[w != 0])
        for w in range(xxyy[0], xxyy[1]):
            if abs(depth[h][w] - avg) < 150:
                img[h][w] = color_depth[h][w]
                min_dis = min(min_dis, depth[h][w])
    return img


def img_xxyy2(img, depth, color_depth, xxyy):
    cal = time.time()
    img_crop = img[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]]
    depth_crop = depth[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]]
    color_depth_crop = color_depth[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]]
    center = np.array([])
    for h in np.arange(img_crop.shape[0]):
        w = depth_crop[h][depth_crop[h] != 0]
        try:
            min_index = np.argmin(w)
        except Exception:
            pass
            #print(w)
        min_index = int((xxyy[1] - xxyy[0]) / 2)
        center = np.append(center, min_index)
        img_crop[h][min_index] = color_depth_crop[h][min_index]
    # avg = round(np.average(center))
    # print(center)
    avg = int(stats.mode(center[center != 0])[0][0])
    print(avg)
    img_crop[:, avg] = [0, 0, 255]
    if abs(avg - len(img_crop[0]) / 2) > 10:
        global fuck
        print(f"./error_pole/{fuck}.jpg")
        cv2.imwrite(f"./error_pole/{fuck}.jpg", img_crop)
        fuck += 1
    print((time.time() - cal) * 1000)
    return img


def img_xxyy(img, depth, color_depth, xxyy):
    min_dis = 9999
    cal = time.time()
    croped_region = depth[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]]
    non_zero_values = croped_region[croped_region != 0]
    non_zero_avg = np.average(non_zero_values)
    mask_of_where_abs_less_than_150 = np.abs(croped_region - non_zero_avg) < 0
    croped_img = img[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]]
    croped_color_depth = color_depth[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]]
    croped_img[mask_of_where_abs_less_than_150] = croped_color_depth[
        mask_of_where_abs_less_than_150]  # copy_depth_image
    img[xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]] = croped_img
    non_zero_min_depth = non_zero_values.min(initial=min_dis)
    print((time.time() - cal) * 1000)
    return img


def new_frame(img, depth, color_depth, result):
    width = len(img[0])
    height = len(img)
    for i in result:
        if i[0] == 0:
            cal = time.time()
            xxyy = xywh_to_xxyy(i[1:], width, height)
            img = img_xxyy2(img, depth, color_depth, xxyy)
            print((time.time() - cal) * 1000)
    return img


def main(gui):
    cam_thread = threading.Thread(target=camera_start, daemon=True)
    cam_thread.start()
    now = time.time()

    weight = "runs/train/2.3k-1440-300-tiny/weights/best.pt"
    conf_thres = 0.50
    iou_thres = 0.45
    size = 736
    classes = None
    detect = detect_api.Detect(weight, conf_thres, iou_thres, classes, view_img=True, trace=not DEBUG)
    detect.init_size(size)

    cam_thread.join()
    webcam.start()
    print(f"Initialized camera and Yolo v7 with {round((time.time() - now), 5)}s")
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    start = time.time()
    while True:
        if not webcam.used:
            frame, depth = webcam.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            result, img = detect.detect_image(frame, size)
            mid = int(img.shape[1] / 2)
            img[:, mid] = [0, 0, 255]
            cv2.imshow("live", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            w, h, fps = webcam.get_wh_fps()
            used_time = time.time() - start
            fps_cal = round(1 / used_time, 5)
            print(f"resolution: {w}x{h}, detected {len(result)} object with {used_time}s, fps: cal {fps_cal} cam {fps}")
            start = time.time()
        else:
            time.sleep(0.00001)
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gui = None#GUI()
    threading.Thread(target=main, name="main", daemon=False, args=(gui,)).start()
    #gui.start()
