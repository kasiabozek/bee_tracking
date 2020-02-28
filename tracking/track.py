import numpy as np
import os
import re

from utils.paths import TRACK_DIR, FTS_DIR
from utils.func import EMB_SIZE, SQ, FR_D

LAST_N = 10
VIS_CUT = 1.75
DDIV = 30
FR1, FR2, LOOK_BACK_BASE, LEN_MIN = 0, 100, 10, 10

###############################################

def get_look_back(x, y, is_butt):
    res = LOOK_BACK_BASE
    if (x < 0.1*FR_D) or (y < 0.1*FR_D) or (x > 0.9*FR_D) or (y > 0.9*FR_D):
        res = LOOK_BACK_BASE * 0.5
    if is_butt:
        res = LOOK_BACK_BASE * 10
    return res

def get_look_back_tra(t, i, last_seen):
    x = t[last_seen[i, LAST_N - 1], i, 0]
    y = t[last_seen[i, LAST_N - 1], i, 1]
    is_butt = np.all(t[last_seen[i, (LAST_N // 2):], i, 2] == 1)
    return get_look_back(x, y, is_butt)


def dist_m(t, tra_cols, last_seen, lens, fr_i, fr_preds):
    res = np.zeros([0,5])
    vs = np.zeros((LAST_N, EMB_SIZE))
    for i1 in range(t.shape[1]):
        if tra_cols[i1]:
            x1, y1 = tuple(t[last_seen[i1,LAST_N-1],i1,:2])
            is_butt = np.all(t[last_seen[i1, (LAST_N//2):], i1, 2] == 1)

            l1 = lens[i1]
            for vi in range(LAST_N):
                vs[vi,:] = t[last_seen[i1,vi],i1,4:]
            d = ((x1 - fr_preds[:, 0]) ** 2 + (y1 - fr_preds[:, 1]) ** 2) ** 0.5
            fr_d = fr_i - last_seen[i1,LAST_N-1]

            if is_butt == 0:  ## if last detection is a butt don't search further
                d = d / np.floor(fr_d ** 0.5)
            else:
                d = d * 3  ## more stringent on butts not moving
            d[fr_d > get_look_back(x1, y1, is_butt)] = 2 * SQ
            det_is = np.where(d <= SQ)[0]

            res_tra = np.zeros((det_is.shape[0], 5))
            res_tra[:, 0] = i1
            res_tra[:, 1] = det_is
            res_tra[:, 4] = 1 - l1/np.max(lens)
            for i2, det_i in enumerate(det_is):
                res_tra[i2, 2] = d[det_i]
                vd = [np.sum((fr_preds[det_i, 4:] - vs[vi,:]) ** 2) ** 0.5 for vi in range(LAST_N)]
                res_tra[i2, 3] = np.min(vd)
            res_tra = res_tra[res_tra[:, 3] <= VIS_CUT,:]
            res = np.append(res, res_tra, axis=0)
    return res


def get_matches(d):
    res = np.zeros((0, 2), dtype=np.int)
    d5 = d[:,3] + d[:,2]/DDIV + d[:,4]
    d = np.append(d, np.reshape(d5,(-1,1)), axis=1)
    min_vis = np.min(d[:,5])
    while d.shape[0] > 0:
        i = np.where(d[:,5] == min_vis)[0][0]
        i1, i2 = int(d[i, 0]), int(d[i, 1])
        res = np.append(res, np.array([i1, i2], ndmin=2), axis=0)
        d = np.delete(d, np.where(d[:, 0] == i1)[0], axis=0)
        d = np.delete(d, np.where(d[:, 1] == i2)[0], axis=0)
        if d.shape[0] > 0:
            min_vis = np.min(d[:, 5])
    return res


def save_trajectory(t, i, tr_nb, total_saved, total_discarded):
    res = t[:, i, :]
    res = res[:,:4]
    fr_nbs = np.reshape(np.array(range(res.shape[0])), (res.shape[0], 1))
    res = np.concatenate((fr_nbs, res), axis=1)
    res = res[res[:,1] != 0,:]
    if res.shape[0] >= LEN_MIN:
        print("  saving trajectory %i, length: %i" % (tr_nb, res.shape[0]), flush=True)
        np.savetxt(os.path.join(TRACK_DIR, "%06d.txt" % tr_nb), res, fmt="%i", delimiter=",")
        total_saved += res.shape[0]
        tr_nb += 1
    else:
        total_discarded += res.shape[0]
    return (total_saved, total_discarded, tr_nb)


def build_trajectories():
    n_dim = 4
    total_discarded = 0
    total_saved = 0

    if os.path.exists(TRACK_DIR):
        for fl in os.listdir(TRACK_DIR):
            os.remove(os.path.join(TRACK_DIR, fl))
    else:
        os.mkdir(TRACK_DIR)

    tr_nb = 0
    fr_preds = np.load(os.path.join(FTS_DIR, "%06d.npy" % FR1))
    t1 = np.zeros((FR2-FR1+1, fr_preds.shape[0], 4+EMB_SIZE), dtype=np.float32)
    last_seen = np.zeros([t1.shape[1], LAST_N], dtype=np.int)
    lens = [1]*fr_preds.shape[0]
    for i in range(fr_preds.shape[0]):
        t1[0,i,:] = fr_preds[i,:]

    t = np.zeros((t1.shape[0], t1.shape[1] * n_dim, t1.shape[2]), dtype=np.float32)
    t[:, :t1.shape[1], :] = t1
    tra_cols = np.zeros((t1.shape[1] * n_dim), dtype=np.bool)
    tra_cols[:t1.shape[1]] = True
    lens.extend([-1] * (t1.shape[1]*n_dim))
    last_seen = np.append(last_seen, np.zeros((last_seen.shape[0]*(n_dim-1), LAST_N), dtype=np.int) - 1, axis=0)

    for fr in range(FR1+1, FR2):
        fr_preds = np.load(os.path.join(FTS_DIR, "%06d.npy" % fr))
        d = dist_m(t, tra_cols, last_seen, lens, np.array([fr]*fr_preds.shape[0]), fr_preds)
        ms = get_matches(d)
        unm_past = list(filter(lambda i: tra_cols[i] and (not (i in ms[:, 0])), range(t.shape[1])))
        unm_curr = list(filter(lambda i: not (i in ms[:, 1]), range(fr_preds.shape[0])))

        if fr % 10 == 0:
            print("frame: %i finished: %i continued: %i saved: %i discarded: %i" % (fr, tr_nb, np.sum(tra_cols), total_saved, total_discarded), flush=True)
            print("  matched: %i  unmatched: %i   new: %i" % (ms.shape[0], len(unm_past), len(unm_curr)), flush=True)

        for i in range(ms.shape[0]):
            i1, i2 = tuple(ms[i,:])
            i1, i2 = (int(i1), int(i2))
            t[fr,i1,:] = fr_preds[i2,:]
            last_seen[i1,:(LAST_N-1)] = last_seen[i1,1:LAST_N]
            last_seen[i1,LAST_N-1] = fr
            lens[i1] += 1

        for i1 in unm_past:
            look_back = get_look_back_tra(t, i1, last_seen)
            if (fr - last_seen[i1,LAST_N-1]) > look_back:
                (total_saved, total_discarded, tr_nb) = save_trajectory(t, i1, tr_nb, total_saved, total_discarded)
                tra_cols[i1] = False
                t[:,i1,:] = 0

        for i, i2 in enumerate(unm_curr):
            new_i = np.where(np.logical_not(tra_cols))[0][0]
            t[fr,new_i,:] = fr_preds[i2,:]
            tra_cols[new_i] = True
            last_seen[new_i,:] = fr
            lens[new_i] = 1

    t = t[:,tra_cols,:]
    for i in range(t.shape[1]):
        (total_saved, total_discarded, tr_nb) = save_trajectory(t, i, tr_nb, total_saved, total_discarded)

######################################################

def sort_trajectories():
    files = [f for f in os.listdir(TRACK_DIR) if re.search('([0-9]+)\.txt$', f)]
    print("Sorting trajectories, %i files" % len(files), flush=True)
    files.sort()
    all_tras = []
    tra_lens = np.zeros((len(files), 2), dtype=np.int)
    for i, file in enumerate(files):
        tr = np.loadtxt(os.path.join(TRACK_DIR, file), delimiter=",", ndmin=2)
        all_tras.append(tr)
        tra_lens[i,:] = [tr[0,0], (tr[-1,0] - tr[0,0]) + 1]

    ord = np.lexsort((-tra_lens[:, 1], tra_lens[:, 0]))
    tra_lens = tra_lens[ord,:]
    np.savetxt(os.path.join(TRACK_DIR, "tra_lens.txt"), tra_lens, delimiter=",", fmt="%i")

    for i in range(tra_lens.shape[0]):
        np.savetxt(os.path.join(TRACK_DIR, "%06d.txt" % i), all_tras[ord[i]], fmt="%i", delimiter=",")




