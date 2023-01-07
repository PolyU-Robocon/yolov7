import threading
import time

import cv2

from detect_api import Detect
from servo_arduino import Servo
from webcam import Webcam

CAM_ID = 0
WEIGHT = "runs/train/11k-1440-300-tiny/weights/best.pt"
CONF_THRES = 0.25
IOU_THRES = 0.45
SIZE = 736  # mul of 16
ARDUINO_PIN = 9
SERVO_OFFSET = 9
SERVO_COM = "/dev/ttyUSB0"
TRACE = True


class PoleAim:
    def __init__(self, cam_id, weight, conf_thres, iou_thres, img_size, arduino_pin, servo_com, servo_offset, trace):
        self.cam_id = cam_id
        self.weight = weight
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.servo_offset = servo_offset
        self.servo_com = servo_com
        self.servo: Servo
        self.arduino_pin = arduino_pin
        self.servo_offset = servo_offset
        self.trace = trace
        self.result = []
        self.webcam = Webcam(width=1280, height=720)
        now = time.time()

        cam_thread = threading.Thread(target=self.camera_init)
        servo_thread = threading.Thread(target=self.servo_init)
        detect_thread = threading.Thread(target=self.detect_init)

        cam_thread.start()
        servo_thread.start()
        detect_thread.start()

        cam_thread.join()
        servo_thread.join()
        detect_thread.join()

        print(f"Initialized with {round((time.time() - now), 5)}s")

    def camera_init(self):
        self.webcam.cam_init()

    def servo_init(self):
        self.servo = Servo(self.arduino_pin, self.servo_offset, self.servo_com)

    def detect_init(self):
        self.detect = Detect(self.weight, self.conf_thres, self.iou_thres, view_img=True, trace=self.trace)
        self.detect.init_size(self.img_size)

    def aim(self, target):
        deg = (0.5 - target[1]) * 80
        self.servo.move(deg)
        print(deg)

    def detecting(self):
        self.webcam.start()
        cv2.namedWindow("live", cv2.WINDOW_NORMAL)
        # start = time.time()
        while True:
            if not self.webcam.used:
                frame = self.webcam.read()
                self.result = self.detect.detect_image(frame, self.img_size)
                # w, h, fps = self.webcam.get_wh_fps()
                # used_time = time.time() - start
                # fps_cal = round(1 / used_time, 5)
                # print(f"resolution: {w}x{h}, detected {len(self.result)} object with {used_time}s, fps: cal {fps_cal} cam {fps}")
                # start = time.time()
            else:
                time.sleep(0.00001)
        self.webcam.cam.release()
        cv2.destroyAllWindows()

    def process(self, targets):
        print(targets)
        for i in list(targets):
            if i[0] == 0:
                targets.remove(i)
        return targets

    def run(self):
        threading.Thread(target=self.detecting, daemon=True).start()
        while True:
            a = input()
            targets = self.process(self.result)
            try:
                a = int(a)
                for i in targets:
                    if i[6] == a:
                        self.aim(i)
            except ValueError:
                for i in targets:
                    if i[0] == 0:
                        self.aim(i)


def main():
    auto_aim = PoleAim(CAM_ID, WEIGHT, CONF_THRES, IOU_THRES, SIZE, ARDUINO_PIN, SERVO_COM, SERVO_OFFSET, TRACE)
    auto_aim.run()


if __name__ == '__main__':
    main()
