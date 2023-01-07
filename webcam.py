import cv2
import threading


class Webcam:
    def __init__(self, cam_id=0, width=1280, height=720, fourcc="MJPG"):
        self.id = cam_id
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.stopped = False
        self.used = True
        self.cam = None
        self.grabbed = None
        self.frame = None

    def cam_init(self):
        self.cam = cv2.VideoCapture(self.id)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        if not self.cam.isOpened():
            print("Cannot open camera")
            exit(0)
        print(f"CAMERA {self.id} Initialized")

        self.grabbed, self.frame = self.cam.read()
        if self.grabbed is False:
            print('No more frames to read')
            exit(0)

    def start(self):
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cam.read()
            if self.grabbed is False:
                print('No more frames to read')
                self.stopped = True
            else:
                self.used = False

    def read(self):
        self.used = True
        return self.frame

    def stop(self):
        self.stopped = True

    def get_wh_fps(self):
        width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        return width, height, fps
