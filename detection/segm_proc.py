import numpy as np
from sklearn.decomposition import PCA
import math

SZ_MIN = 50
SZ_MAX = 1000

#def points_dst(prs, lbs):
#    res = np.zeros((len(prs), len(lbs)))
#    for ip in range(len(prs)):
#        for il in range(len(lbs)):
#            res[ip,il] = np.sqrt((prs[ip][0]-lbs[il][0])**2+(prs[ip][1]-lbs[il][1])**2)
#    return res
#
#def remove_el(v, i):
#    v1 = v[:i]
#    v1.extend(v[(i+1):])
#    return v1
#
#def find_labels_from_arr(labels_point, labels_segm):
#    lxs, lys = np.where(labels_point != -1)
#    lbs = []
#    for j in range(len(lxs)):
#        lbs.append(lxs[j], lys[j], labels_segm[lxs[j], lys[j]], labels_point[lxs[j],lys[j]]))
#    return lbs


def mark_region(i1, i2, m, regions, nb):
    q = [(i1,i2)]
    while (len(q) > 0):
        (i1,i2) = q.pop(0)
        regions[i1,i2] = nb
        if (i1 > 0) and (regions[i1-1,i2] == 0) and m[i1-1,i2]:
            regions[i1-1,i2] = -1
            q.append((i1-1, i2))
        if (i2 > 0) and (regions[i1,i2-1] == 0) and m[i1,i2-1]:
            regions[i1,i2-1] = -1
            q.append((i1, i2-1))
        if (i1 < m.shape[0]-1) and (regions[i1+1,i2] == 0) and m[i1+1,i2]:
            regions[i1+1,i2] = -1
            q.append((i1+1, i2))
        if (i2 < m.shape[1]-1) and (regions[i1,i2+1] == 0) and m[i1,i2+1]:
            regions[i1,i2+1] = -1
            q.append((i1, i2+1))
    return regions

def connected_components(m): # due to problems with opencv2
    rg_nb = 0
    regions = np.zeros(m.shape)
    for i1 in range(m.shape[0]):
        for i2 in range(m.shape[1]):
            if (regions[i1,i2] == 0) and m[i1,i2]:
                rg_nb = rg_nb + 1
                regions = mark_region(i1, i2, m, regions, rg_nb)
    return rg_nb+1, regions

def find_center(regions, rg):
    ys, xs = np.where(regions == rg)
    sz = len(xs)
    res = (-1, -1, -1)
    if (sz > SZ_MIN) and (sz <= SZ_MAX):
        m = np.zeros((sz, 2))
        m[:,0] = -xs
        m[:,1] = ys
        pca = PCA(n_components=2)
        pca.fit(m)
        p = pca.components_
        a = math.pi / 2
        if p[0,1] != 0:
            a = (math.atan(p[0,0]/p[0,1]) + math.pi) % math.pi
        res = (np.mean(xs), np.mean(ys), a)
    return res


def find_type(pred, regions, rg):
    v = pred[regions == rg]
    res = float(np.sum(v == 2)) / np.prod(np.shape(v))
    return res


def find_angle(pred, regions, rg):
    v = pred[regions == rg]
    v = v[v >= 0]
    if v.size == 0:
        return 0
    v[v > 1] = 1
    return np.percentile(v, 99) * 2 * math.pi


def extract_positions(pred_class, pred_angle):
    region_nbs, regions = connected_components(pred_class > 0)
    res = []
    for rg in range(1, region_nbs):
        (x, y, ax) = find_center(regions, rg)
        if ax != -1:
            a = find_angle(pred_angle, regions, rg)
            cl = find_type(pred_class, regions, rg)
            res.append((x, y, cl, a, ax))
    return res


def angle_diff(a1, a2):
    if a2 > a1:
        a1, a2 = a2, a1
    d = a1 - a2
    if d > 180:
        a1 = -(360 - a1)
        d = a2 - a1
    return d


