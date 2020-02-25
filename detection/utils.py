import numpy as np
from PIL import Image
from tensorflow.python.client import device_lib
import tensorflow as tf
import os, re
import itertools

GPU_NAME = 'tower'
D = 256
FR_D = 512
MARGIN = 25

DATA_I = 0
LABEL_POINT_I = 1
LABEL_SEGM_I = 2
LABEL_ANGLE_I = 3
WEIGHTS_I = 4
WEIGHTS_CLASS_I = 5

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "frames")
POS_DIR = os.path.join(DATA_DIR, "detections")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
DET_DATA_DIR = os.path.join(DATA_DIR, "detection_data")


def find_last_checkpoint(path):
    files = [f for f in os.listdir(path) if re.search('index$', f)]
    nbs = map(lambda s: int(re.match(r'model_([0-9]+)\.ckpt\.index', s).group(1)), files)
    return max(nbs)


def get_offsets_in_frame(inference):
    offs_ax = range(0, FR_D, D)
    if inference:
        offs = list(itertools.product(offs_ax, offs_ax))
    else:
        offs = list(itertools.product(offs_ax, offs_ax, [0, 1], [0,1]))
    return offs


def find_devices():
    devices = device_lib.list_local_devices()
    cpu, gpu = "", ""
    i = 0
    while (i < len(devices)) and (cpu == "") and (gpu == ""):
        if devices[i].device_type == "CPU":
            cpu = devices[i].name
        elif devices[i].device_type == "GPU":
            gpu = devices[i].name
        i += 1
    if not tf.test.is_built_with_cuda():
        gpu = ""
    return (cpu, gpu)


def read_img(fr, path):
    img = Image.open(os.path.join(path, "%06d.png" % fr)).convert('L')
    img = np.asarray(img, dtype=np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img * 2 - 1
    return img

