import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel


class Detect:
    def __init__(self, weights, conf_thres=0.25, iou_thres=0.45, classes=None, device="", view_img=True, save_dir=None,
                 trace=True):
        self.weights = weights
        self.save_dir = save_dir
        self.trace = trace
        self.save = False if self.save_dir is None else True
        self.device, self.model = self.init(weights, device)
        self.stride = int(self.model.stride.max())  # model stride
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms = conf_thres, iou_thres, classes, False

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Second-stage classifier
        # if classify:
        #    modelc = load_classifier(name='resnet101', n=2)  # initialize
        #    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        # else:
        self.classify = False
        self.modelc = None
        self.view_img = view_img

    def init(self, weights, device):
        # Initialize
        set_logging()
        device = select_device(device)
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model

        return device, model

    def init_size(self, size):
        # if trace:

        new_size = check_img_size(size, s=self.stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, size)
        if self.half:
            self.model.half()  # to FP16
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, new_size, new_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once

    def get_pt_cv_data(self, img0, img_size, img_return=False):
        # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img

    def detect_image(self, im0s, imgsz, path=None):
        save = path != None
        img = self.get_pt_cv_data(im0s, imgsz)
        # old_img_w = old_img_h = imgsz
        # old_img_b = 1
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # lag source
        '''# Warmup 
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]'''

        # Inference
        # t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)
        # t3 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)
        p, s, im0 = path, '', im0s
        # Process detections
        result = []
        cnt = 1
        for i, det in enumerate(pred):  # detections per image
            if save and self.save_dir != None:
                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                #print(det)
                # for *xyxy, conf, cls in reversed(det):
                #    print(*xyxy)
                det = sorted(det, key=lambda x: x[0])
                #for *xyxy, conf, cls in reversed(a):
                #    print(*xyxy)
                #print(a)
                # Write results
                for *xyxy, conf, cls in det:
                    cache_result = []
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    if conf <= 0.3:
                        continue
                    cache_result.append(int(line[0].item()))
                    for j in range(1, 5):
                        cache_result.append(line[j])
                    cache_result.append(line[5].item())
                    if line[0] != 1:  # injected
                        cache_result.append(cnt)
                    # print(cache_result)
                    result.append(cache_result)

                    # Write to file
                    if save:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Add bbox to image
                    if save or self.view_img:
                        add = ""
                        if line[0] != 1:  # injected
                            add = str(cnt)
                            cnt += 1
                        label = f'{add}{self.names[int(cls)]} {conf:.2f}'
                        if conf > 0.3:  # todo fuck out this out of api
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                # print(result)
            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # Stream results

            # Save results (image with detections)
            if save:
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
        return result, im0

    def detect(self, source, imgsz, save_img=False):
        # print(imgsz)
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #    ('rtsp://', 'rtmp://', 'http://', 'https://'))

        self.init_size(imgsz)

        # Set Dataloader
        # dataset = LoadImages(source, img_size=imgsz, stride=self.stride)

        t0 = time.time()
        import os
        # for path, img, im0s, _ in dataset:
        for file in os.scandir(source):
            # Read image
            im0s = cv2.imread(file.path)  # BGR
            self.detect_image(im0s, imgsz, file.path)

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    import os, shutil

    for file in os.scandir("./runs/detect"):
        if file.name.startswith("test"):
            shutil.rmtree(file.path)
            print(f"removed {file.path}")
    # check_requirements(exclude=('pycocotools', 'thop'))
    # Directories
    # print(opt.classes)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
    detect = Detect(opt.weights, opt.conf_thres, opt.iou_thres, opt.classes, device=opt.device, save_dir=save_dir)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect.detect(opt.source, opt.img_size)
                strip_optimizer(opt.weights)
        else:
            detect.detect(opt.source, opt.img_size)
