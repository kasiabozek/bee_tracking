import os
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from . import inception
from utils import func
import multiprocessing as mp
from utils.func import FR1, FR2, EMB_SIZE, DT
from utils.paths import IMG_DIR, FTS_DIR, TMP_DIR, POS_DIR, CHECKPOINT_DIR

SH = (DT, DT, 1)
BATCH_SIZE = 12
LABEL_SIZE = 5
N_PROC = 3

tf.logging.set_verbosity(tf.logging.ERROR)

to_save = mp.Queue()

if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

###############################################

def save_output_worker():
    while True:
        output_i = to_save.get()
        if output_i < 0:
            break
        fl = os.path.join(TMP_DIR, "outputs_%i.npz" % output_i)
        arrs = np.load(fl)
        (labels, vecs) = arrs['lab'], arrs['v']
        os.remove(fl)
        for i in range(BATCH_SIZE):
            vec = vecs[i, :]
            label = labels[i, :]
            if label[0] >= 0:
                fr, x, y, cl, a = tuple(label)
                vec_fr = np.reshape(np.append(np.array([x, y, cl, a]), vec), (1, -1))
                with open(os.path.join(FTS_DIR, "%06d.txt" % fr), 'a') as f:
                    np.savetxt(f, vec_fr, delimiter=',', newline='\n')


###############################################

def img_gen():
    for fr in range(FR1, FR2):
        frame_preds = np.loadtxt(os.path.join(POS_DIR, "%06d.txt" % fr), dtype=np.int, delimiter=",")
        frame_img = func.read_img(fr, IMG_DIR)
        for i in range(frame_preds.shape[0]):
            x, y, cl, a = tuple(frame_preds[i,:])
            img = np.reshape(func.crop(frame_img, x, y), SH)
            label = np.array([fr,x,y,cl,a],dtype=np.int)
            yield img, label
    img = np.zeros(SH)
    label = np.zeros((LABEL_SIZE), dtype=np.int)
    label[0] = -1
    while True:
        yield img, label


def build_ds():
    types = (tf.float32, tf.int32)
    shapes = (tf.TensorShape(SH), tf.TensorShape([LABEL_SIZE]))

    test_ds = tf.data.Dataset.from_generator(img_gen, types, shapes)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_iter = test_ds.make_one_shot_iterator()

    return test_iter


def build_model(checkpoint_dir):
    cpu, gpu = func.find_devices()
    tf_dev = gpu if gpu != "" else cpu
    with tf.Graph().as_default(), tf.device(cpu):
        test_iter = build_ds()
        is_train = tf.constant(False, dtype=tf.bool, shape=[])
        outputs = ()
        with tf.device(tf_dev), tf.name_scope('%s_%d' % (func.GPU_NAME, 0)) as scope:
            img, label = test_iter.get_next()
            v, _ = inception.inception_v3(img, is_training=is_train, scope=scope, num_classes=EMB_SIZE)
            outputs = (label, v)
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        checkpoint = func.find_last_checkpoint(checkpoint_dir)
        print("Restoring checkpoint %i.." % checkpoint, flush=True)
        saver.restore(sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % checkpoint))

        init = tf.local_variables_initializer()
        sess.run(init)

    return sess, outputs


def start_workers():
    workers = [mp.Process(target=save_output_worker) for _ in range(N_PROC)]
    for p in workers:
        p.start()
    return workers

def stop_workers(workers):
    for i in range(len(workers)):
        to_save.put(-1)
    for p in workers:
        p.join()

def test_bees(sess, outputs):
    finished = False
    fc = 0
    while not finished:
        outs = sess.run(outputs)
        if fc % 50 == 0:
            print("...frame: %i" % outs[0][0,0], flush=True)
        np.savez(os.path.join(TMP_DIR, "outputs_%i.npz" % fc), lab=outs[0], v=outs[1])
        to_save.put(fc)
        fc += 1
        finished = (outs[0][0,0] < 0)

###############################################

def build_embeddings(checkpoint_dir=os.path.join(CHECKPOINT_DIR, "inception")):
    if os.path.exists(FTS_DIR):
        for fl in os.listdir(FTS_DIR):
            os.remove(os.path.join(FTS_DIR, fl))
    else:
        os.mkdir(FTS_DIR)

    print("Building the model..", flush =True)
    sess, outputs = build_model(checkpoint_dir)
    try:
        workers = start_workers()
        print("Starting infrerence..", flush=True)
        test_bees(sess, outputs)
    finally:
        stop_workers(workers)
        print("Done..", flush=True)
    func.txt2npy(FTS_DIR, FTS_DIR, N_PROC)


