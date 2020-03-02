import os
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from . import inception
import multiprocessing as mp
from utils.func import EMB_SIZE, DT, SQ, FR1, FR2
from utils.paths import IMG_DIR, TMP_DIR, POS_DIR, CHECKPOINT_DIR
from utils import func
import random, math
import queue
import ctypes
import time
from .reference_trajectory import read_all_refs

SH = [DT, DT, 1]
SH3 = [DT, DT, 3]
BATCH_SIZE = 12
BATCHES_PER_ITER = 10
LABEL_SIZE = 9
N_PROC = 3
BASE_LR = 0.0001
MAX_IMGS = 100
MARGIN = 0.5  # margin for the triplet loss function

tf.logging.set_verbosity(tf.logging.ERROR)

to_save = mp.Queue()

if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

CLASS_WEIGHTS = [1, 1]

hard_label_q = mp.Queue()
all_q = mp.Queue()
hard_q = mp.Queue()
all_totake = mp.Queue()
hard_totake = mp.Queue()
all_imgs, all_labels, hard_imgs, hard_labels = [], [], [], []


######## DATA GENERATOR #####################################################################


def tras_in_frames(ref_tras, fr1, fr2):
    res = []
    for i in range(len(ref_tras)):
        i1 = np.where(ref_tras[i][:,0] == fr1)[0]
        if i1.size > 0:
            i2 = np.where(ref_tras[i][:, 0] == fr2)[0]
            if i2.size > 0:
                res.append((i,i1[0],i2[0]))
    return res

def generate_all_pairs(ref_tras, fr1, fr2):
    track_nbs = tras_in_frames(ref_tras, fr1, fr2)
    fr_d = fr2 - fr1
    frame2_preds = np.loadtxt(os.path.join(POS_DIR, "%06d.txt" % fr2), dtype=np.int, delimiter=",")
    all_pairs = []
    tot_size = 0
    for tr, i1, i2 in track_nbs:
        x1, y1 = tuple(ref_tras[tr][i1,1:3])
        x2, y2 = tuple(ref_tras[tr][i2,1:3])
        pairs = [(x1,y1), (x2,y2)]
        ds = ((x1 - frame2_preds[:, 0]) ** 2 + (y1 - frame2_preds[:, 1]) ** 2) ** 0.5
        det_is = np.where(ds <= SQ * math.floor(fr_d ** 0.5))[0]
        for det_i in det_is:
            x, y = tuple(frame2_preds[det_i, :2])
            if not (x == x2 and y == y2):
                pairs.append((x, y))
                tot_size += 1
        if len(pairs) > 2:
            all_pairs.append(pairs)
    return all_pairs, tot_size

def save_to_arr(img, label, all):
    q_in, q_out, arr_img, arr_label = (all_totake, all_q, all_imgs, all_labels) if all else (hard_totake, hard_q, hard_imgs, hard_labels)
    i = q_in.get()
    arr_img[i][0].acquire()
    arr_label[i][0].acquire()
    np.copyto(arr_img[i][1][:, :, :], img)
    np.copyto(arr_label[i][1][:], label)
    arr_img[i][0].release()
    arr_label[i][0].release()
    q_out.put(i)


def all_producer():
    #random.seed()
    ref_tras, _ = read_all_refs()
    fr_d_max = 10
    frs = list(range(FR1, FR2-fr_d_max-1))
    fr_ds = list(range(1,fr_d_max))
    while True:
        fr1 = random.sample(frs, 1)[0]
        fr_d = random.sample(fr_ds, 1)[0]
        fr2 = fr1 + fr_d
        all_pairs, sz = generate_all_pairs(ref_tras, fr1, fr2)
        if sz > 0:
            frame1_img = func.read_img(fr1, IMG_DIR)
            frame2_img = func.read_img(fr2, IMG_DIR)
            for pairs in all_pairs:
                x1, y1 = pairs[0]
                anchor_img = np.reshape(func.crop(frame1_img, x1, y1), SH)
                x2, y2 = pairs[1]
                pos_img = np.reshape(func.crop(frame2_img, x2, y2), SH)
                for (x,y) in pairs[2:]:
                    neg_img = np.reshape(func.crop(frame2_img, x, y), SH)
                    img = np.concatenate((anchor_img, pos_img, neg_img), axis=2)
                    label = [fr1, x1, y1, fr2, x2, y2, x, y, 0]
                    save_to_arr(img, label, True)


def hard_producer():
    n = 3
    pairs = np.zeros((n, LABEL_SIZE), dtype=np.int)
    i = 0
    while True:
        (fr1, x1, y1, fr2, x2,y2, x3, y3) = hard_label_q.get()
        pairs[i, :] = [fr1, x1, y1, fr2, x2, y2, x3, y3, 1]
        i = (i + 1) % n
        if i == 0:
            fr_pairs = np.unique(pairs[:, [0, 3]], axis=0)
            for j in range(fr_pairs.shape[0]):
                (fr1, fr2) = tuple(fr_pairs[j, :])
                jhs = np.where((pairs[:, 0] == fr1) & (pairs[:, 3] == fr2))[0]
                frame1_img = func.read_img(fr1, IMG_DIR)
                frame2_img = func.read_img(fr2, IMG_DIR)
                for jh in jhs:
                    anchor_img = np.reshape(func.crop(frame1_img, pairs[jh, 1], pairs[jh, 2]), SH)
                    pos_img = np.reshape(func.crop(frame2_img, pairs[jh, 4], pairs[jh, 5]), SH)
                    neg_img = np.reshape(func.crop(frame2_img, pairs[jh, 6], pairs[jh, 7]), SH)
                    img = np.concatenate((anchor_img, pos_img, neg_img), axis=2)
                    label = pairs[jh, :]
                    save_to_arr(img, label, False)


def data_gen():
    while True:
        i = all_q.get()
        all_imgs[i][0].acquire()
        all_labels[i][0].acquire()
        img = np.copy(all_imgs[i][1])
        label = np.copy(all_labels[i][1])
        all_imgs[i][0].release()
        all_labels[i][0].release()
        yield img, label
        all_totake.put(i)

        try:
            i = hard_q.get(False)
            hard_imgs[i][0].acquire()
            hard_labels[i][0].acquire()
            img = np.copy(hard_imgs[i][1])
            label = np.copy(hard_labels[i][1])
            hard_imgs[i][0].release()
            hard_labels[i][0].release()
            yield img, label
            hard_totake.put(i)
        except queue.Empty:
            pass

##### MODEL ########################################################################


def triplet_loss(scope, imgs, is_train, n, margin, batch_all=True):
    anchor_img, pos_img, neg_img = tf.split(imgs, [1, 1, 1], 3, name="split")

    anchor, _ = inception.inception_v3(anchor_img, is_training=is_train, scope=scope, num_classes=n)
    tf.get_variable_scope().reuse_variables()
    positive, _ = inception.inception_v3(pos_img, is_training=is_train, scope=scope, num_classes=n)
    tf.get_variable_scope().reuse_variables()
    negative, _ = inception.inception_v3(neg_img, is_training=is_train, scope=scope, num_classes=n)

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    if batch_all:
        # batch all
        t_loss = pos_dist - neg_dist + margin
        t_loss = tf.maximum(t_loss, 0.0)
        valid_triplets = tf.to_float(tf.greater(t_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        # Get final mean triplet loss over the positive valid triplets
        t_loss = tf.reduce_sum(t_loss) / (num_positive_triplets + 1e-16)
    else:
        # batch hard
        hardest_positive_dist = tf.reduce_max(pos_dist, axis=0, keepdims=True)
        # For each anchor, get the hardest negative
        hardest_negative_dist = tf.reduce_min(neg_dist, axis=0, keepdims=True)
        t_loss = hardest_positive_dist - hardest_negative_dist + margin
        t_loss = tf.maximum(t_loss, 0.0)

    return pos_dist, neg_dist, t_loss


def build_ds():
    types = (tf.float32, tf.int32)
    shapes = (tf.TensorShape(SH3), tf.TensorShape([LABEL_SIZE]))

    train_ds = tf.data.Dataset.from_generator(data_gen, types, shapes)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_iter = train_ds.make_one_shot_iterator()

    return train_iter



def train(checkpoint_dir, n_iters):
    cpu, gpu = func.find_devices()
    tf_dev = gpu if gpu != "" else cpu
    with tf.Graph().as_default(), tf.device(cpu):

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        update_ops = []
        opt = tf.train.AdamOptimizer(learning_rate=BASE_LR)
        train_iter = build_ds()
        is_train = tf.constant(True, dtype=tf.bool, shape=[])

        outputs = []
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(tf_dev), tf.name_scope('%s_%d' % (func.GPU_NAME, 0)) as scope:
                imgs, label = train_iter.get_next()
                pos_distance, neg_distance, loss = triplet_loss(scope, imgs, is_train, EMB_SIZE, MARGIN)
                outputs = [loss, pos_distance, neg_distance, label]
                update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(func.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        batchnorm_updates_op = tf.group(*update_ops)
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        checkpoint = func.find_last_checkpoint(checkpoint_dir)
        acc_file = os.path.join(checkpoint_dir, "accuracy.csv")
        if checkpoint > 0:
            print("Restoring checkpoint %i.." % checkpoint, flush=True)
            saver.restore(sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % checkpoint))
            checkpoint += 1
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            checkpoint = 0

        init = tf.local_variables_initializer()
        sess.run(init)

        # TRAIN/TEST

        for itr in range(checkpoint, checkpoint+n_iters):
            ttime = time.time()
            outs = np.zeros((BATCHES_PER_ITER * BATCH_SIZE, 3))
            labels = np.zeros([BATCHES_PER_ITER * BATCH_SIZE, LABEL_SIZE])
            for i in range(BATCHES_PER_ITER):
                _, batch_outs = sess.run([train_op, outputs])
                inds = list(range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE))
                outs[inds, 0] = batch_outs[0]
                outs[inds, 1] = batch_outs[1]
                outs[inds, 2] = batch_outs[2]
                labels[inds, :] = batch_outs[3]

            hard_triplets = outs[:, 2] < (outs[:, 1] + MARGIN)
            is_all = labels[:, -1] == 0
            hard_of_all = np.sum(outs[is_all, 2] < (outs[is_all, 1] + MARGIN))
            is_hard = labels[:, -1] == 1
            hard_of_hard = np.sum(outs[is_hard, 2] < (outs[is_hard, 1] + MARGIN))
            print("-- %i / %i wrong triplets  %i / %i in all  %i / %i in previously wrong" %
                  (np.sum(hard_triplets), outs.shape[0], hard_of_all, np.sum(is_all), hard_of_hard, np.sum(is_hard)), flush=True)

            hard_labels = np.reshape(labels[hard_triplets, :], (-1, LABEL_SIZE))
            for i in range(hard_labels.shape[0]):
                hard_label_q.put(tuple(hard_labels[i, :-1]))

            with open(acc_file, 'ab') as f:
                np.savetxt(f, np.reshape(np.mean(outs, axis=0), (1, 3)), fmt='%.5f', delimiter=',', newline='\n')

            print("TRAIN/TEST EPOCH %i : %.2f min - loss: %.6f; mean pos dist: %.6f  mean neg dist: %.6f " %
                     (itr, (time.time() - ttime) / 60, np.mean(outs[:, 0]), np.mean(outs[:, 1]), np.mean(outs[:, 2])))

            saver.save(sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % itr))


def create_arrays():
    global all_imgs, all_labels, hard_imgs, hard_labels

    sz = SH3[0]*SH3[1]*SH3[2]
    all_imgs = [ mp.Array(ctypes.c_float, sz) for _ in range(MAX_IMGS) ]
    all_imgs = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.float32, count=sz).reshape(SH3)) for sharr in all_imgs]

    hard_imgs = [ mp.Array(ctypes.c_float, sz) for _ in range(MAX_IMGS) ]
    hard_imgs = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.float32, count=sz).reshape(SH3)) for sharr in hard_imgs]

    all_labels = [ mp.Array(ctypes.c_float, LABEL_SIZE) for _ in range(MAX_IMGS) ]
    all_labels = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.int32, count=LABEL_SIZE).reshape(LABEL_SIZE)) for sharr in all_labels]

    hard_labels = [ mp.Array(ctypes.c_float, LABEL_SIZE) for _ in range(MAX_IMGS) ]
    hard_labels = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.int32, count=LABEL_SIZE).reshape(LABEL_SIZE)) for sharr in hard_labels]


def start_workers():
    workers = [mp.Process(target=all_producer) for _ in range(N_PROC)]
    workers.extend([mp.Process(target=hard_producer) for _ in range(N_PROC)])
    for p in workers:
        p.start()
    for i in range(MAX_IMGS):
        all_totake.put(i)
        hard_totake.put(i)
    return workers



def stop_workers(workers):
    print("stopping workers..", flush=True)
    for p in workers:
        p.terminate()


def run_train(checkpoint_dir=os.path.join(CHECKPOINT_DIR,"inception"), n_iters=10):
    try:
        create_arrays()
        workers = start_workers()
        train(checkpoint_dir, n_iters)
    finally:
        stop_workers(workers)
