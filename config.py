import json
import os


DEFAULT_CONFIG = {
    "system": "linux",
    "cam_id": 0,
    "weight": "runs/train/exp/weights/best.pt",
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "size": 736,
    "arduino_pin": 9,
    "servo_offset": -13,
    "servo_ratio": 5,
    "servo_com": "/dev/ttyUSB0/",
    "trace": True
}


class Config:
    def __init__(self, path="config_linux.json"):
        self.path=path
        self.system = "windows"
        self.cam_id = 0
        self.camera_width_angle = 113
        self.width = 1280
        self.height = 720
        self.weight = ""
        self.tensorrt = False
        self.tensorrt_weight = "2.3k-1440-400-tiny.engine"
        self.k4a = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.img_size = 736
        self.arduino_pin = 9
        self.servo_offset = -13
        self.servo_ratio = 5
        self.servo_com = "COM18"
        self.trace = True

    def load_config(self, path):
        sync = False
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("Config not find")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump(DEFAULT_CONFIG, config_file, indent=4)
            return DEFAULT_CONFIG
        with open(path, "r", encoding="utf-8") as config_file:
            data = dict(json.load(config_file))
        for keys in DEFAULT_CONFIG.keys():
            if keys not in data.keys():
                print(f"Config {keys} not found, use default value {DEFAULT_CONFIG[keys]}")
                data.update({keys: DEFAULT_CONFIG[keys]})
                sync = True
        if sync:
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump(data, config_file, indent=4)
        return data

    def init_config(self):
        config_data = self.load_config(self.path)
        self.system = config_data["system"]
        self.cam_id = config_data["cam_id"]
        self.camera_width_angle = config_data["camera_width_angle"]
        self.width = config_data["width"]
        self.height = config_data["height"]
        self.weight = config_data["weight"]
        self.tensorrt = config_data["tensorrt"]
        self.tensorrt_weight = config_data["tensorrt_weight"]
        self.k4a = config_data["k4a"]
        self.conf_thres = config_data["conf_thres"]
        self.iou_thres = config_data["iou_thres"]
        self.img_size = config_data["size"]
        self.arduino_pin = config_data["arduino_pin"]
        self.servo_offset = config_data["servo_offset"]
        self.servo_ratio = config_data["servo_ratio"]
        self.servo_com = config_data["servo_com"]
        self.trace = config_data["trace"]
