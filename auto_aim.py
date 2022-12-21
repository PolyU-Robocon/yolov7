import cv2
import time
import threading

from detect_api import Detect
from servo_arduino import Servo

camno = 0
weight = "runs/train/11k-1440-300-tiny/weights/best.pt"
conf_thres = 0.25
iou_thres = 0.45
size = 736 #mul of 16
arduino_pin = 9
servo_offset = 9
trace = False

class PoleAim:
    def __init__(self, camno, weight, conf_thres, iou_thres, img_size, arduino_pin, servo_offset, trace):
        self.camno = camno
        self.weight = weight
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.servo_offset = servo_offset
        self.servo: Servo
        self.arduino_pin = arduino_pin
        self.servo_offset = servo_offset
        self.trace = trace
        self.result = []
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
        print(f"CAMERA {self.camno} STARTING")
        self.cam = cv2.VideoCapture(self.camno)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if not self.cam.isOpened():
            print("Cannot open camera")
            exit()
        print(f"CAMERA {self.camno} STARTED")

    def servo_init(self):
        self.servo = Servo(self.arduino_pin, self.servo_offset)
    
    def detect_init(self):
        self.detect = Detect(self.weight, self.conf_thres, self.iou_thres, view_img=True, trace=self.trace)
        self.detect.init_size(self.img_size)

    def get_frame(self):
            start = time.time()
            ret, frame = self.cam.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            result = self.detect.detect_image(frame, size)
            #width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            #height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            #used_time = round((time.time() - start), 5)
            #print(f"resolution: {width}x{height}detected {len(result)} object, used time: {used_time}s, fps: {round(1 / used_time, 5)}")
            return result

    def aim(self, target):
        deg = (0.5 - target[1]) * 81
        self.servo.move(deg)
        print(deg)
    
    def detecting(self):
        cv2.namedWindow("live", cv2.WINDOW_NORMAL)
        while True:
            self.result = self.get_frame()
            #for i in self.result:
            #    if i[0] == 0:
            #        print(i[3])
            #        print(i[4])
        self.cam.release()
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
    auto_aim = PoleAim(camno, weight, conf_thres, iou_thres, size, arduino_pin, servo_offset, trace)
    auto_aim.run()

if __name__ == '__main__':
    main()