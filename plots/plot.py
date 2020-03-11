import colorsys
import math
import os
import random
import re
from datetime import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.func import FR1, FR2, SQ
from utils.paths import TRACK_DIR, IMG_DIR, POS_DIR, PLOTS_DIR

WIDTH = 2
BEE_COL = (255, 0, 0, 200)
OTHER_BEE_COL = (255, 255, 0, 200)

###########################################

def add_center(draw, x, y, col, d=WIDTH):
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

def add_arrow(draw, x, y, a, col, w=1, d=25.0):
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


def add_circle(draw, x, y, col, w=1, r=10):
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=col)
    for i in range(1, w):
        r = r - i
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=col)

###########################################


def time_text(img, tra, fr, fps, fsize):
    xmax, ymax = img.size
    m = 5
    corners = [ (xmax-fsize*2.5-m, ymax-fsize-m), (xmax-fsize*2.5-m, m), (m, ymax-fsize-m), (m, m) ]
    corner_i = 0
    if tra.size > 0:
        d = np.min(((tra[:, 1] - corners[corner_i][0]) ** 2 + (tra[:, 2] - corners[corner_i][1]) ** 2) ** 0.5)
        in_corner = d < SQ * 2
        while (corner_i < len(corners)) and in_corner:
            corner_i += 1
            d = np.min(((tra[:, 1] - corners[corner_i][0]) ** 2 + (tra[:, 2] - corners[corner_i][1]) ** 2) ** 0.5)
            in_corner = d < SQ * 2
        if corner_i == len(corners):
            corner_i = 0
    x, y = corners[corner_i]

    h = fr // (fps*60*60)
    fr = fr % (fps*60*60)
    min = fr // (fps * 60)
    fr = fr % (fps * 60)
    sec = fr // fps
    s = time(h, min, sec).strftime("%M:%S")

    return (x, y), s


def plot_frame(fr, tra, fps):
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
        x = tra[wh[0], 1]
        y = tra[wh[0], 2]

        draw.line([(x-SQ, y-SQ), (x+SQ, y-SQ), (x+SQ, y+SQ), (x-SQ, y+SQ), (x-SQ, y-SQ)], fill=BEE_COL, width=WIDTH)
        add_center(draw, x, y, BEE_COL)

    fsize = 20
    (x, y), s = time_text(img, tra, fr, fps, fsize)
    fnt = ImageFont.truetype('plots/arial.ttf', fsize)
    draw.text((x, y), s, font=fnt, fill=(255, 0, 0, 255))
    return img


def plot_trajectory(id, fps=10):
    print("plotting trajectory %i.." % id, flush=True)
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    tra = np.loadtxt(os.path.join(TRACK_DIR, "%06d.txt" % id), delimiter=',')
    fr1, fr2 = int(tra[0,0]), int(tra[-1,0])
    frs = list(range(fr1, fr2))

    imgs = [ plot_frame(fr, tra, fps) for fr in frs ]

    movie_file = os.path.join(PLOTS_DIR, "%06d.gif" % (id))

    imgs[0].save(movie_file, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)

    return movie_file

##################################################################

def plot_frame_bees(fr, bees_in_frames, hues):
    bif = bees_in_frames[fr]

    img = Image.open(os.path.join(IMG_DIR, "%06d.png" % fr)).convert('RGBA')
    markers = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(markers)
    all_bees = np.loadtxt(os.path.join(POS_DIR, "%06d.txt" % fr), delimiter=',').astype(np.int)

    fnt = ImageFont.truetype('plots/arial.ttf', 20)

    for i in range(all_bees.shape[0]):
        x, y = all_bees[i, 0], all_bees[i, 1]

        wh = np.where((bif[:, 1] == all_bees[i,0]) & (bif[:, 2] == all_bees[i,1]))[0]
        if wh.size > 0:
            ids = bif[wh, 0]
            for id in ids:
                r, g, b = [int(j * 255) for j in colorsys.hsv_to_rgb(hues[id], 1, 1)]
                add_center(draw, x, y, (r,g,b, 255))
                draw.text((x, y), str(id), font=fnt, fill=(r,g,b,255))
        else:
            if all_bees[i, 2] == 1:
                add_circle(draw, x, y, OTHER_BEE_COL)
            else:
                add_arrow(draw, x, y, math.radians(all_bees[i, 3]), OTHER_BEE_COL)
            add_center(draw, x, y, OTHER_BEE_COL, d=WIDTH//2)

    img = Image.alpha_composite(img, markers)

    return img


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


def plot_all_trajectories(fps=10):
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    bif, tra_nbs = organize_by_frame()

    hues = list(map(lambda x: float(x)/len(tra_nbs), tra_nbs))
    random.shuffle(hues)

    imgs = [plot_frame_bees(fr, bif, hues) for fr in range(FR1, FR2)]
    movie_file = os.path.join(PLOTS_DIR, "all_trajectories.gif")
    imgs[0].save(movie_file, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)

    return movie_file

##################################################################


def plot_detections(fr, save, fps=10):
    if save:
        if not os.path.exists(PLOTS_DIR):
            os.mkdir(PLOTS_DIR)

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

    fsize = 20
    (x, y), s = time_text(img, np.array([]), fr, fps, fsize)
    fnt = ImageFont.truetype('plots/arial.ttf', fsize)
    draw.text((x, y), s, font=fnt, fill=(255, 0, 0, 255))

    if save:
        img.save(os.path.join(PLOTS_DIR, "%06d.png" % fr))
    return img


def plot_detection_video(fps=10):
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    imgs = [ plot_detections(fr, False, fps) for fr in range(FR1, FR2) ]

    movie_file = os.path.join(PLOTS_DIR, "detections.gif")
    imgs[0].save(movie_file, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)
