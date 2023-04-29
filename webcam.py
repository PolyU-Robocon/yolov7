import cv2
import threading
import numpy as np


class Webcam:
    def __init__(self, cam_id=0, width=1280, height=720, fourcc="MJPG", k4a=False):
        self.id = cam_id
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.k4a = k4a
        self.stopped = False
        self.used = True
        self.cam = None
        self.grabbed = None
        self.frame = None
        self.depth = None

    def cam_init(self):
        if self.k4a:
            import pyk4a
            self.cam = pyk4a.PyK4A(
                pyk4a.Config(
                    color_resolution=pyk4a.ColorResolution.RES_720P,
                    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                )
            )
            self.cam.start()
        else:
            self.cam = cv2.VideoCapture(self.id)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

            if not self.cam.isOpened():
                print("Cannot open camera")
                exit(0)
        print(f"CAMERA {self.id} Initialized")
        if not self.k4a:
            self.grabbed, self.frame = self.cam.read()
            if self.grabbed is False:
                print('No more frames to read')
                exit(0)

    def start(self):
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if not self.k4a:
                self.grabbed, self.frame = self.cam.read()
                if self.grabbed is False:
                    print('No more frames to read')
                    self.stopped = True
                else:
                    self.used = False
            else:
                cap = self.cam.get_capture()
                self.frame = np.ascontiguousarray(cap.color[:,:,0:3])
                self.depth = cap.transformed_depth
                self.used = False

    def read(self):
        self.used = True
        return self.frame, self.depth

    def stop(self):
        self.stopped = True

    def get_wh_fps(self):
        if self.k4a:
            return self.width, self.height, 30
        width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        return width, height, fps

    def release(self):
        if not self.k4a:
            self.cam.release()
