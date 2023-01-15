# Inference for ONNX model
import cv2
import numpy as np
import cupy as cu
import onnxruntime as ort
import random
import time
import threading

cuda = True
w = "./runs/train/11k-1440-300-tiny/weights/best.onnx"
img_file = './image/test.jpg'
names = ['pole', 'pole_disk']
size = (1088, 1088)  # 1400, 1088
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)

cnt = 0
start1 = time.time()


def refuck():
    global start1
    start1 = time.time()


def fuck(start):
    global cnt
    cnt += 1
    current = time.time() - start
    print(f"end{cnt}:{current}")
    return current


def letterbox(im, new_shape=size, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = cu.mod(dw, stride), cu.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def init():
    global colors, outname, inname
    colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]


def run_img(img):
    image = img
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(cu.float32)
    im /= 255

    inp = {inname[0]: im}
    fuck(start1)
    # ONNX inference
    outputs = session.run(outname, inp)[0]
    fuck(start1)

    ori_images = [img]
    fuck(start1)
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = cu.array([x0, y0, x1, y1])
        box -= cu.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        name += ' ' + str(score)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
    fuck(start1)
    cv2.imshow("live", ori_images[0])
    fuck(start1)


class FakeCam:
    def __init__(self, width, height):
        self.img = None
        self.raw_img = None
        self.used = True

    def cam_init(self):
        pass

    def start(self):
        self.used = False

    def set_img(self, img):
        self.img = img
        self.raw_img = img

    def __new_img(self):
        self.img = self.raw_img.copy()

    def read(self):
        threading.Thread(target=self.__new_img, daemon=True).start()
        return self.img


def run():
    global cnt
    from webcam import Webcam
    # simg = cv2.imread(img_file)
    webcam = FakeCam(width=1920, height=1080)
    webcam.cam_init()
    webcam.start()
    webcam.used = False
    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    frame = cv2.imread("./image/test.jpg")
    webcam.set_img(frame)
    while True:
        cnt = 0
        refuck()
        if cv2.waitKey(1) == ord('q'):
            break
        fuck(start1)
        if not webcam.used:
            # frame = frame.copy()
            frame = webcam.read()
            # cv2.imwrite("test.jpg", frame)
            fuck(start1)
            run_img(frame)
            current = fuck(start1)
            print(f"fps: {round(1 / (current), 5)}, used_time = {current}")
        else:
            time.sleep(0.00001)
    cv2.destroyAllWindows()


init()
run()
