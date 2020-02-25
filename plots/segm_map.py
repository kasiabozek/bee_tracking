import numpy as np
import os
from detection import utils
import colorsys
from PIL import Image


def plot_segm_map(frame_nb, fl_nb=0, data_path=utils.DET_DATA_DIR):
    npz = np.load(os.path.join(data_path, "%06d.npz" % fl_nb))
    data = npz['data']
    img = data[frame_nb, 0, :, :]
    map = data[frame_nb, 1, :, :]
    map = map / np.max(map)

    map_img = Image.new("RGBA", map.shape, color=(0,0,0,0))

    for i1 in range(map.shape[0]):
        for i2 in range(map.shape[1]):
            if map[i1,i2] > 0:
                c = [ int(j * 255) for j in colorsys.hsv_to_rgb(map[i1,i2], 1, 1) ]
                map_img.putpixel((i1,i2), (c[0],c[1],c[2], 128))

    im = Image.fromarray(np.uint8((img+1)/2 * 255))
    im = im.convert("RGBA")
    Image.alpha_composite(im, map_img).save(os.path.join(utils.PLOTS_DIR, "%06d_segm_map.png" % frame_nb))


def plot_angle_map(frame_nb, fl_nb=0, data_path=utils.DET_DATA_DIR):
    npz = np.load(os.path.join(data_path, "%06d.npz" % fl_nb))
    data = npz['data']
    img = data[frame_nb, 0, :, :]
    map = data[frame_nb, 2, :, :]

    im = Image.fromarray(np.uint8((img+1)/2 * 255))
    im = im.convert("RGB")
    for i1 in range(map.shape[0]):
        for i2 in range(map.shape[1]):
            h = map[i1,i2]
            if h >= 0:
                rgb = colorsys.hsv_to_rgb(h, 1, 1)
                rgb = tuple(int(255 * x) for x in rgb)
                im.putpixel((i1,i2), rgb)
    im.save(os.path.join(utils.PLOTS_DIR, "%06d_angle_map.png" % frame_nb))


def plot_weight_map(frame_nb, fl_nb=0, data_path=utils.DET_DATA_DIR):
    npz = np.load(os.path.join(data_path, "%06d.npz" % fl_nb))
    data = npz['data']
    img = data[frame_nb, 0, :, :]
    map = data[frame_nb, 3, :, :]

    map = map - 1
    map = 255 * map / np.max(map)
    map = map.astype(np.int)
    map_img = Image.new("RGBA", map.shape, color=(0,0,0,0))
    for i1 in range(map.shape[0]):
        for i2 in range(map.shape[1]):
            if (map[i1,i2] > 1):
                map_img.putpixel((i1,i2), (255, 0, 0, map[i1,i2]))

    im = Image.fromarray(np.uint8((img+1)/2 * 255))
    im = im.convert("RGBA")
    Image.alpha_composite(im, map_img).save(os.path.join(utils.PLOTS_DIR, "%06d_weight_map.png" % frame_nb))

