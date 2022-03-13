import numpy as np
import os, re
import math
from utils import func, paths
from scipy.stats import norm

'''
Params:
  xc: center x coordinate
  yc: center y coordinate
  a: angle in radians
  h: total height of image
  w: total width of image
  r1: height (?) of ellipse
  r2: width (?) of ellipse
'''
def ellipse_around_point(xc, yc, a, h, w, r1, r2):
    ind = np.zeros((2, h, w), dtype=np.int)
    m = np.zeros((h, w), dtype=np.float32)
    for i in range(w):
        ind[0,:,i] = range(-yc, h-yc)
    for i in range(h):
        ind[1,i,:] = range(-xc, w-xc)
    rs1 = np.arange(r1, 0, -float(r1)/r2)
    rs2 = np.arange(r2, 0, -1.0)
    s = math.sin(a)
    c = math.cos(a)

    pdf0 = norm.pdf(0)
    for i in range(len(rs1)):
        i1 = rs1[i]
        i2 = rs2[i]
        v = norm.pdf(float(len(rs1) - i) / len(rs1)) / pdf0
    # rotated ellipse
        m[((ind[0,:,:] * s + ind[1,:,:] * c)**2 / i1**2 + (ind[1,:,:] * s - ind[0,:,:] * c)**2 / i2**2) <= 1] = v

    return m


def generate_segm_labels(img, pos, w=10, r1=7, r2=12):
    FR_H, FR_W = img.shape
    res = np.zeros((4, FR_H, FR_W), dtype=np.float32) # data,labels_segm, labels_angle, weight
    res[0] = img
    res[2] = -1

    for i in range(pos.shape[0]):
        x, y, obj_class, a = tuple(pos[i,:])
        obj_class += 1

        if obj_class == 2:
            a = 2 * math.pi
        else:
            a = math.radians(float(a))

        if obj_class == 1:
            m = ellipse_around_point(x, y, a, FR_H, FR_W, r1, r2)
        else:
            m = ellipse_around_point(x, y, a, FR_H, FR_W, r1, r1)

        mask = (m != 0)
        res[1][mask] = obj_class
        res[2][mask] = a / (2*math.pi)
        res[3][mask] = m[mask]

    res[3] = res[3]*(w - 1) + 1
    return res

'''
Takes in images and positions, generates labels, and stores compressed np file.
Assumes all images are of the same dimension (and multiples of 256).

Saved result is of shape [# frames, 4, frame height, frame width].
The 4 "channels" for each frame are: normalized image, class labels, angle labels, loss weights.

Params
  frame_nbs: list of frame numbers, corresponding to names of the image files in the format '00000<n>.png'.
  img_dir: directory holding .png files. 
     Images should be all of the same dimension, where dimension height and width are divisible by 256.
  pos_dir: directory holding .txt files with the same frame number format.
     Each row represents a bee object with 4 position args: x center, y center, bee class (0 for full/1 for cell), angle in degrees clockwise from the vertical.
     x and y are measured from the top left corner, which represents (0, 0).
  out_dir: directory to store the .npz file with labels for all frames.
'''
def create_from_frames(frame_nbs, img_dir, pos_dir, out_dir=paths.DET_DATA_DIR):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    files = [f for f in os.listdir(out_dir) if re.search('npz', f)]
    fl_nb = len(files)
    # Check shape of first frame.
    img_shape = func.read_img(0, img_dir).shape
    func.check_img_shape(img_shape)
    
    res = np.zeros((len(frame_nbs), 4, img_shape[0], img_shape[1]), dtype=np.float32)
    for i, frame_nb in enumerate(frame_nbs):
        print("frame %i.." % frame_nb)
        img = func.read_img(frame_nb, img_dir)
        pos = np.loadtxt(os.path.join(pos_dir, "%06d.txt" % frame_nb), delimiter=",", dtype=np.int)
        res[i] = generate_segm_labels(img, pos)
    np.savez(os.path.join(out_dir, "%06d.npz" % fl_nb), data=res, det=pos)

