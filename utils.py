import os, sys
import numpy as np
from PIL import Image
from multiprocessing import Pool
from functools import partial
from tensorflow.python.client import device_lib
import tensorflow as tf

GPU_NAME = 'tower'
D = 80
SQ = D // 2
EMB_SIZE = 64
FR1, FR2 = 0, 100
FR_D = 512

DATA_DIR = "sample_data"
IMG_DIR = os.path.join(DATA_DIR, "frames")
POS_DIR = os.path.join(DATA_DIR, "detections")
FTS_DIR = os.path.join(DATA_DIR, "detections_embeddings")
TRACK_DIR = os.path.join(DATA_DIR, "trajectories")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")


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


def crop(img, x, y):
    d = np.zeros((D, D), dtype=np.float32)
    x1 = 0 if x >= SQ else SQ - x
    x2 = 2*SQ if (img.shape[0] - x) >= SQ else SQ + (img.shape[0] - x)
    y1 = 0 if y >= SQ else SQ - y
    y2 = 2*SQ if (img.shape[1] - y) >= SQ else SQ + (img.shape[1] - y)
    d[x1:x2,y1:y2] = img[max(0, x - SQ):min(x + SQ, img.shape[0]), max(0, y - SQ):min(y + SQ, img.shape[1])]
    return d


def t2n(i, fls, txt_path, npy_path):
    fl = fls[i]
    out_fl = fl.split(".")[0] + ".npy"
    m = np.loadtxt(os.path.join(txt_path, fl), delimiter=",")
    np.save(os.path.join(npy_path, out_fl), m)


def txt2npy(txt_path, npy_path, nproc):
    fls = os.listdir(txt_path)
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    pool = Pool(processes=nproc)
    pool.map(partial(t2n, fls = fls, txt_path=txt_path, npy_path=npy_path), range(len(fls)))
    pool.close()
    pool.join()


class DownloadProgress:
    def __init__(self):
        self.old_percent = 0
        sys.stdout.write("downloading.. ")

    def progress_hook(self, count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if (percent > self.old_percent) and (percent % 3 == 0):
            self.old_percent = percent
            sys.stdout.write("%i%%.. " % percent)
        if percent == 100:
            sys.stdout.write('done!\n')