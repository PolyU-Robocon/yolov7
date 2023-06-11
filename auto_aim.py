import threading
import time

import cv2

from detect_api import Detect
from tensorrt_api import TRT_engine
from servo import ArduinoServo, ServoBase
from webcam import Webcam
from config import Config


class PoleAim:
    def __init__(self, config: Config):
        self.now = 1
        self.config = config
        self.camera_width_angle = self.config.camera_width_angle
        self.weight = self.config.weight
        self.cam_id = self.config.cam_id
        self.width = self.config.width
        self.height = self.config.height
        self.conf_thres = self.config.conf_thres
        self.iou_thres = self.config.iou_thres
        self.img_size = self.config.img_size
        self.servo_offset = self.config.servo_offset
        self.servo_com = self.config.servo_com
        self.servo: ServoBase
        self.arduino_pin = self.config.arduino_pin
        self.servo_offset = self.config.servo_offset
        self.trace = self.config.trace
        self.result = []
        self.webcam = Webcam(self.cam_id, self.width, self.height, k4a=self.config.k4a)
        self.end = False
        now = time.time()

        cam_thread = threading.Thread(target=self.camera_init, name="cam_init", daemon=True)
        # servo_thread = threading.Thread(target=self.servo_init, name="servo_init", daemon=True)
        detect_thread = threading.Thread(target=self.detect_init, name="detect_init", daemon=True)

        cam_thread.start()
        # servo_thread.start()
        detect_thread.start()
        self.servo_init()
        cam_thread.join()
        # servo_thread.join()
        detect_thread.join()
        if self.config.ros:
            self.init_listener()
        print(f"Initialized with {round((time.time() - now), 5)}s")

    def camera_init(self):
        self.webcam.cam_init()

    def servo_init(self):
        if self.config.ros:
            from servo import ROSServo
            self.servo = ROSServo(self.servo_offset, self.config.servo_ratio)
        else:
            self.servo = ArduinoServo(self.servo_offset, self.config.servo_ratio, self.arduino_pin, self.servo_com)

    def detect_init(self):
        if self.config.tensorrt:
            self.detect = TRT_engine(self.config.tensorrt_weight)
        else:
            self.detect = detect_api.Detect(self.weight, self.conf_thres, self.iou_thres, trace=not DEBUG)
            self.detect.init_size(self.img_size)

    def listen(self, data):
        pass

    def init_listener(self):
        return
        import rospy
        from std_msgs.msg import UInt8
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("chatter", UInt8, listen)

    def callback(data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def aim(self, target):
        self.now = 0
        deg = (target[1] - 0.5) * self.camera_width_angle
        self.servo.move(deg)  # Reverse
        print(deg)

    def detecting(self):
        self.webcam.start()
        cv2.namedWindow("live", cv2.WINDOW_NORMAL)
        # start = time.time()
        while True:
            if not self.webcam.used:
                frame, depth = self.webcam.read()
                # frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.result, img = self.detect.detect_image(frame, self.img_size, self.now)
                mid = int(img.shape[1] / 2)
                img[:, mid] = [0, 0, 255]
                mid = int(img.shape[1] / self.config.camera_width_angle * (self.config.camera_width_angle / 2 - 18))
                img[:, mid] = [0, 255, 0]
                mid = int(img.shape[1] / self.config.camera_width_angle * (self.config.camera_width_angle / 2 + 18))
                img[:, mid] = [0, 255, 0]
                cv2.imshow("live", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.end = True
                    break
                # w, h, fps = self.webcam.get_wh_fps()
                # used_time = time.time() - start
                # fps_cal = round(1 / used_time, 5)
                # print(f"resolution: {w}x{h}, detected {len(self.result)} object with {used_time}s, fps: cal {fps_cal} cam {fps}")
                # start = time.time()
            else:
                time.sleep(0.00001)
        self.webcam.cam.release()
        cv2.destroyAllWindows()
        exit(0)

    def process(self, targets):
        # print(targets)
        for i in list(targets):
            if i[0] == 1:
                targets.remove(i)
        return targets

    def left(self):
        self.now -= 1
        if self.now < 1:
            self.now = 1
        print(f"pole:  {self.now}")

    def right(self):
        self.now += 1
        if self.now > len(self.result):
            self.now = len(self.result)
        print(f"pole:  {self.now}")

    def run(self):
        threading.Thread(target=self.detecting, name="detecting", daemon=True).start()
        while not self.end:
            a = input()
            targets = self.process(self.result)
            try:
                a = int(a)
                for i in targets:
                    if i[6] == a:
                        self.aim(i)
            except ValueError:
                if a.startswith("rescale"):
                    args = a.split(" ")
                    try:
                        self.camera_width_angle = float(args[1])
                        print(self.camera_width_angle)
                    except Exception:
                        pass
                elif a.startswith("c"):
                    self.servo.deg_now = 90 + self.servo.offset
                    self.servo.move(0)
                elif a.startswith("r"):
                    self.left()
                elif a.startswith("l"):
                    self.right()


def main():
    config = Config(path="config.json")
    config.init_config()
    auto_aim = PoleAim(config)
    auto_aim.run()
    from joymessage import RosJoyMessage
    inputmsg = InputJoyMessage(auto_aim)
    try:
        inputmsg.run()
    except Exception as e:
        auto_aim.webcam.cam.release()
        cv2.destroyAllWindows()
        raise e
    exit()


if __name__ == '__main__':
    main()
