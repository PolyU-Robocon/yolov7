import cv2
import tensorrt as trt
import threading
import time
import torch
import numpy as np
from collections import OrderedDict, namedtuple
from webcam import Webcam

classes = ["pole", "disk"]


def plot_one_box(x, img, label, now=False):
    # Plots one bounding box on image img
    color = [0, 0, 255]
    if now:
        color = [0, 255, 0]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=1)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)


def visualize(img, bbox_array, now):
    cnt = 0
    results = []
    for temp in bbox_array:
        xmin = int(temp[1])
        ymin = int(temp[2])
        xmax = int(temp[3])
        ymax = int(temp[4])
        clas = int(temp[0])
        label = ""
        score = temp[5]
        if clas == 0:
            cnt += 1
            label += f"{cnt}-"
        label += f"{classes[clas]} {str(round(score, 2))}"
        if clas == 0 and cnt == now:
            plot_one_box([xmin, ymin, xmax, ymax], img, label, True)
        else:
            plot_one_box([xmin, ymin, xmax, ymax], img, label)
        temp[1] = (xmin + xmax) / 2 / img.shape[1]
        temp[2] = (ymin + ymax) / 2 / img.shape[0]
        temp[3] = xmax - xmin
        temp[4] = ymax - ymin
        results.append(temp)  # xywh
    return results, img


class TRT_engine:
    def __init__(self, weight) -> None:
        self.imgsz = [640, 640]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data,
                                                    int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self, im, color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img, self.r, self.dw, self.dh

    def preprocess(self, image):
        self.img, self.r, self.dw, self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img, 0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict(self, img, threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores = self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        num = int(nums[0])
        new_bboxes = []
        cnt = 0
        for i in range(num):
            if scores[i] < threshold:
                continue
            xmin = (boxes[i][0] - self.dw) / self.r
            ymin = (boxes[i][1] - self.dh) / self.r
            xmax = (boxes[i][2] - self.dw) / self.r
            ymax = (boxes[i][3] - self.dh) / self.r
            if classes[i] == 0:
                cnt += 1
                new_bboxes.append([classes[i], xmin, ymin, xmax, ymax, scores[i], cnt])
            else:
                new_bboxes.append([classes[i], xmin, ymin, xmax, ymax, scores[i]])
        new_bboxes = sorted(new_bboxes, key=lambda x: x[1])
        return new_bboxes

    def detect_image(self, img, size=0, now=1, threshold=0.5):
        results = self.predict(img, threshold)
        #print(now)
        used = []
        for i in results:
            if len(i) == 7:
                used += i
        if now == 0:
            now = sorted(used, key=lambda i: abs((i[1] + i[3]) / 2 - 0.5))[0][6]
        results, img = visualize(img, results, now)
        return results, img


def camera_start(webcam):
    webcam.cam_init()


if __name__ == "__main__":
    webcam = Webcam()
    cam_thread = threading.Thread(target=camera_start, args=(webcam,), daemon=True)
    cam_thread.start()
    trt_engine = TRT_engine("./2.3k-1440-400-tiny.engine")
    cam_thread.join()
    webcam.start()
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    start = time.time()
    while True:
        if not webcam.used:
            img, _ = webcam.read()
            time1 = time.time()
            results = trt_engine.predict(img, threshold=0.5)
            time2 = time.time()
            img = visualize(img, results)
            time3 = time.time()
            cv2.imshow("img", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(
                f"fps: {round(1 / (time.time() - start), 5)} time:{round((time.time() - start) * 1000, 5)}ms, objects: {len(results)}, pred={round((time2 - time1) * 1000, 5)}ms, visual={round((time3 - time2) * 1000, 5)}ms, show={round((time.time() - time3) * 1000, 5)}ms")
            start = time.time()
        else:
            time.sleep(0.00001)
