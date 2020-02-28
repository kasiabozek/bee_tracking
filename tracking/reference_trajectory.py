import os, shutil, re
import numpy as np
from utils.paths import REF_DIR, TRACK_DIR



def read_all_refs(ref_dir=REF_DIR):
    files = [f for f in os.listdir(ref_dir) if re.search('([0-9]+).txt$', f)]
    files.sort()
    nbs = list(map(lambda f: int(re.match(r'([0-9]+).txt$', f).group(1)), files))
    res = []
    for i, file in enumerate(files):
        res.append(np.loadtxt(os.path.join(ref_dir, file), dtype=np.int, delimiter=","))
    return res, nbs


def overlap_tra(tra1, tra2):
    i1 = 0
    i2 = 0
    ov = 0.0
    while (i1 < tra1.shape[0]) and (i2 < tra2.shape[0]):
        if tra1[i1,0] < tra2[i2,0]:
            i1 += 1
        elif tra2[i2,0] < tra1[i1,0]:
            i2 += 1
        else:
            ov += (tra1[i1,1] == tra2[i2,1]) and (tra1[i1,2] == tra2[i2,2])
            i1 += 1
            i2 += 1
    ov = ov / (tra1.shape[0] + tra2.shape[0] - ov)
    return ov


def check_overlap(tra, refs):
    res = (0, -1)
    j = 0
    while (j < len(refs)) and (res[0] < 0.1):
        ov = overlap_tra(tra, refs[j])
        res = (ov, j)
        j += 1
    return res


def add_ref_trajectories(nbs, ref_dir=REF_DIR, tra_dir=TRACK_DIR):
    if not os.path.exists(REF_DIR):
        os.mkdir(REF_DIR)
    refs, ref_nbs = read_all_refs()

    for nb in nbs:
        tra = np.loadtxt(os.path.join(tra_dir, "%06d.txt" % nb), dtype=np.int, delimiter=",")
        ov, ref_i = check_overlap(tra, refs)
        if ov > 0.1:
            print("trajectory %i overlap with reference %s by %0.2f" % (nb, ref_nbs[ref_i], ov))
        else:
            ref_nb = ref_nbs[-1]+1 if len(ref_nbs) > 0 else 0
            print("adding trajectory %i as reference %i" % (nb, ref_nb))
            shutil.copyfile(os.path.join(tra_dir, "%06d.txt" % nb), os.path.join(ref_dir, "%06d.txt" % ref_nb))
            refs.append(tra)
            ref_nbs.append(ref_nb)

