import os, shutil
import random, subprocess, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
from functools import partial
import colorsys
import re
from utils import FR1, FR2, SQ, N_PROC, TRACK_DIR, IMG_DIR, POS_DIR, PLOTS_DIR, FFMPEG

WIDTH = 2
BEE_COL = (255, 0, 0, 200)
OTHER_BEE_COL = (255, 255, 0, 200)

###########################################

def add_center(draw, y, x, col, d=WIDTH):
    draw.rectangle([x-d, y-d, x+d, y+d], outline=col, fill=col)

def draw_head(x, y, a, draw, col, d, w):
    da = math.pi / 7
    dx = round(math.sin(a-da) * d)
    dy = round(math.cos(a-da) * d)
    (x1, y1) = (x - dx, y + dy)
    draw.line([(x1,y1),(x,y)], fill=col, width=w)
    dx = round(math.sin(a+da) * d)
    dy = round(math.cos(a+da) * d)
    (x2, y2) = (x - dx, y + dy)
    draw.line([(x2,y2),(x,y)], fill=col, width=w)

def add_arrow(draw, y, x, a, col, w=1, d=25.0):
    (x1,y1,x2,y2) = (0,0,0,0)
    dn = 4
    if (a == math.pi/2):
        (x1,y1,x2,y2) = (x+d/dn, y, x-d, y)
    elif (a == math.pi*3/4):
        (x1,y1,x2,y2) = (x-d/dn, y, x+d, y)
    elif (a == 0):
        (x1,y1,x2,y2) = (x, y+d/dn, x, y-d)
    elif (a == math.pi):
        (x1,y1,x2,y2) = (x, y-d/dn, x, y+d)
    else:
        dx = math.sin(a) * d
        dy = math.cos(a) * d
        (x1, y1, x2, y2) =(x-dx/dn, y+dy/dn, x+dx, y-dy)
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    draw.line([(x,y),(x2,y2)], fill=col, width=w)
    draw_head(x2, y2, a, draw, col, d/4, w)


def add_circle(draw, y, x, col, w=1, r=10):
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=col)
    for i in range(1, w):
        r = r - i
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=col)

###########################################


def plot_frame(fr_i, frs, tra, out_path):
    fr = frs[fr_i]
    img = Image.open(os.path.join(IMG_DIR, "%06d.png" % fr)).convert('RGBA')
    draw = ImageDraw.Draw(img)

    other_bees = np.loadtxt(os.path.join(POS_DIR, "%06d.txt" % fr), delimiter=',')
    for bee_i in range(other_bees.shape[0]):
        x = other_bees[bee_i, 0]
        y = other_bees[bee_i, 1]
        if other_bees[bee_i, 2] == 1:
            add_circle(draw, x, y, OTHER_BEE_COL)
        else:
            add_arrow(draw, x, y, math.radians(other_bees[bee_i, 3]), OTHER_BEE_COL)
        add_center(draw, x, y, OTHER_BEE_COL, d=WIDTH//2)

    wh = np.where(tra[:,0] == fr)[0]
    if wh.size > 0:
        y = tra[wh[0], 1]
        x = tra[wh[0], 2]

        draw.line([(x-SQ, y-SQ), (x+SQ, y-SQ), (x+SQ, y+SQ), (x-SQ, y+SQ), (x-SQ, y-SQ)], fill=BEE_COL, width=WIDTH)
        add_center(draw, y, x, BEE_COL)

    img.save(os.path.join(out_path, "%06d.png" % fr_i))


def plot_trajectory(id):
    print("plotting trajectory %i.." % id, flush=True)
    out_path = PLOTS_DIR + ("/%i" % id)

    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    else:
        fls = os.listdir(out_path)
        for fl in fls:
            os.remove(os.path.join(out_path, fl))

    tra = np.loadtxt(os.path.join(TRACK_DIR, "%06d.txt" % id), delimiter=',')
    fr1, fr2 = int(tra[0,0]), int(tra[-1,0])
    frs = list(range(fr1, fr2))

    pool = Pool(processes=N_PROC)
    pool.map(partial(plot_frame, frs=frs, tra=tra, out_path=out_path), list(range(len(frs))))
    pool.close()
    pool.join()

    movie_file = os.path.join(PLOTS_DIR, "%06d.mp4" % (id))
    if os.path.isfile(movie_file):
        os.remove(movie_file)


    subprocess.call([FFMPEG, "-r", "10", "-i", out_path + "/%06d.png", "-vf", "format=yuv420p", "-codec:v", "libx264", movie_file])
    shutil.rmtree(out_path)


##################################################################


def plot_frame_bees(fr, bees_in_frames, ind_cols, out_path):
    bif = bees_in_frames[fr]

    img = Image.open(os.path.join(IMG_DIR, "%06d.png" % fr)).convert('RGBA')
    draw = ImageDraw.Draw(img)
    all_bees = np.loadtxt(os.path.join(POS_DIR, "%06d.txt" % fr), delimiter=',').astype(np.int)

    fnt = ImageFont.truetype('arial.ttf', 20)

    for i in range(all_bees.shape[0]):
        x, y = all_bees[i, 0], all_bees[i, 1]

        wh = np.where((bif[:, 1] == all_bees[i,0]) & (bif[:, 2] == all_bees[i,1]))[0]
        if wh.size > 0:
            ids = bif[wh, 0]
            for id in ids:
                col = ind_cols[id]
                draw.line([(y-SQ, x-SQ), (y+SQ, x-SQ), (y+SQ, x+SQ), (y-SQ, x+SQ), (y-SQ, x-SQ)], fill=col, width=WIDTH)
                add_center(draw, x, y, col)
                (r,g,b,_) = ind_cols[id]
                draw.text((y, x), str(id), font=fnt, fill=(r,g,b,100))

        if all_bees[i, 2] == 1:
            add_circle(draw, x, y, OTHER_BEE_COL)
        else:
            add_arrow(draw, x, y, math.radians(all_bees[i, 3]), OTHER_BEE_COL)
        add_center(draw, x, y, OTHER_BEE_COL, d=WIDTH//2)

    img.save(os.path.join(out_path, "%06d.png" % fr))


def organize_by_frame():
    res = {}
    for fr in range(FR1, FR2):
        res[fr] = np.zeros((0,3), dtype=np.int)

    files = [f for f in os.listdir(TRACK_DIR) if re.search('([0-9]+)\.txt$', f)]
    tra_nbs = list(map(lambda s: int(re.match(r'([0-9]+)\.txt', s).group(1)), files))
    tra_nbs.sort()

    for id in tra_nbs:
        tra = np.loadtxt(os.path.join(TRACK_DIR, "%06d.txt" % id), delimiter=',').astype(np.int32)
        for i in range(tra.shape[0]):
            res[tra[i,0]] = np.append(res[tra[i,0]], [[id, tra[i,1], tra[i,2]]], axis=0)
    return res, tra_nbs


def plot_all_trajectories():
    out_path = os.path.join(PLOTS_DIR,"all")

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    bif, tra_nbs = organize_by_frame()

    ind_cols = {}
    n = len(tra_nbs)
    xs = list(map(lambda x: float(x)/n, tra_nbs))
    random.shuffle(xs)
    for i, ind in enumerate(tra_nbs):
        c = [int(j * 255) for j in colorsys.hsv_to_rgb(xs[i], 1, 1)]
        ind_cols[ind] = (c[0], c[1], c[2], 200)

    pool = Pool(processes=N_PROC)
    pool.map(partial(plot_frame_bees, ind_cols=ind_cols, bees_in_frames=bif, out_path=out_path), range(FR1, FR2))
    pool.close()
    pool.join()

    movie_file = os.path.join(PLOTS_DIR, "all_trajectories.mp4")
    if os.path.exists(movie_file):
        os.remove(movie_file)

    subprocess.call([FFMPEG, "-r", "10", "-i", out_path + "/%06d.png", "-vf", "format=yuv420p", "-codec:v", "libx264", movie_file])

    shutil.rmtree(out_path)

##################################################################


def plot_detections(fr, out_path):
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    img = Image.open(os.path.join(IMG_DIR, "%06d.png" % fr)).convert('RGBA')
    draw = ImageDraw.Draw(img)
    all_bees = np.loadtxt(os.path.join(POS_DIR, "%06d.txt" % fr), delimiter=',').astype(np.int)

    for i in range(all_bees.shape[0]):
        x, y = all_bees[i, 0], all_bees[i, 1]
        if all_bees[i, 2] == 1:
            add_circle(draw, x, y, OTHER_BEE_COL)
        else:
            add_arrow(draw, x, y, math.radians(all_bees[i, 3]), OTHER_BEE_COL)
        add_center(draw, x, y, OTHER_BEE_COL, d=WIDTH//2)

    img.save(os.path.join(out_path, "%06d.png" % fr))


def plot_detection_video():
    out_path = os.path.join(PLOTS_DIR, "detections")

    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    pool = Pool(processes=N_PROC)
    pool.map(partial(plot_detections, out_path=out_path), range(FR1, FR2))
    pool.close()
    pool.join()

    movie_file = os.path.join(PLOTS_DIR, "detections.mp4")
    if os.path.exists(movie_file):
        os.remove(movie_file)

    subprocess.call([FFMPEG, "-r", "10", "-i", out_path + "/%06d.png", "-vf", "format=yuv420p", "-codec:v", "libx264", movie_file])
