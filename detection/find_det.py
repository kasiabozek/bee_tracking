import os
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from . import unet
from . import segm_proc
import time
import numpy as np
import math
import shutil
import multiprocessing
from . import utils
from .utils import D, GPU_NAME, DATA_DIR, IMG_DIR, POS_DIR, TMP_DIR

N_PROC = 3
BATCH_SIZE = 4
NUM_LAYERS, NUM_FILTERS, CLASSES = 3, 32, 3

tf.logging.set_verbosity(tf.logging.ERROR)

to_save = multiprocessing.Queue()

def read_all_files():
    drs = [""]
    fls = []
    for dr in drs:
        dr_fls = os.listdir(IMG_DIR)
        dr_fls.sort()
        fls.extend(map(lambda fl: os.path.join(dr, fl), dr_fls))
    print("%i files" % len(fls), flush=True)
    return fls

######## POSTPROCESSING AND SAVING SEGMENTATION RESULTS ############

def save_output_worker():
    output = np.zeros((BATCH_SIZE, 2, D, D))
    while True:
        output_i, offs, cur_fr = to_save.get()
        if output_i < 0:
            break
        fl = os.path.join(TMP_DIR, "segm_outputs_%i.npy" % output_i)
        output[:,:,:,:] = np.load(fl)
        os.remove(fl)

        res = np.zeros((0, 4))
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = offs[batch_i]
            if (off_x >= 0) and (off_y >= 0):
                prs = segm_proc.extract_positions(output[batch_i, 0, :, :], output[batch_i, 1, :, :])
                res_batch = np.zeros((len(prs), 4))
                for i in range(len(prs)):
                    (x, y, cl, a, ax) = prs[i]
                    ax_d = math.degrees(ax)
                    a_d = math.degrees(a)
                    ax_d = ax_d + 180 if (segm_proc.angle_diff(a_d, ax_d) > 90) else ax_d
                    res_batch[i, :] = [x, y, cl, ax_d]
                res_batch[:, 0] += off_x
                res_batch[:, 1] += off_y
                res = np.append(res, res_batch, axis=0)
        print("processed frame %i, %i bees" % (cur_fr, res.shape[0]), flush=True)
        with open(os.path.join(POS_DIR, "%06d.txt" % cur_fr), 'a') as f:
            np.savetxt(f, res, fmt='%i', delimiter=',', newline='\n')

############# INFERENCE MODEL #####################

class DetectionInference:

    def __init__(self):
        self.batch_data = np.zeros((BATCH_SIZE, D, D, 1), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        None

    def build_model(self, checkpoint_nb, checkpoint_file):
        cpu, gpu = utils.find_devices()
        tf_dev = gpu if gpu != "" else cpu
        with tf.Graph().as_default(), tf.device(cpu):

            # equals the number of batches processed * num_gpus.
            #update_ops = []
            self.is_train = False

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                self.placeholder_img = tf.placeholder(tf.float32, shape=(BATCH_SIZE, D, D, 1), name="images")
                self.placeholder_prior = tf.placeholder(tf.float32, shape=(BATCH_SIZE, D, D, NUM_FILTERS), name="prior")

                logits, last_relu, angle_pred = unet.create_unet2(NUM_LAYERS, NUM_FILTERS, self.placeholder_img, self.is_train, prev=self.placeholder_prior, classes=CLASSES)
                self.outputs= (logits, angle_pred)
                self.priors = last_relu
                tf.get_variable_scope().reuse_variables()
                #update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

            #self.batchnorm_updates_op = tf.group(*update_ops)
            self.saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            print("Restoring checkpoint %i.." % checkpoint_nb, flush=True)
            self.saver.restore(self.sess, checkpoint_file)
            init = tf.local_variables_initializer()
            self.sess.run(init)


    def _feed_dict(self, offs, cur_fr, priors):
        img = utils.read_img(cur_fr, IMG_DIR)
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = offs[batch_i]
            if (off_x >= 0) and (off_y >= 0):
                self.batch_data[batch_i,:,:,0] = img[off_y:(off_y + D), off_x:(off_x + D)]
            else:
                self.batch_data[batch_i, :, :, :] = 0
        res = [(self.placeholder_prior, priors), (self.placeholder_img, self.batch_data)]
        return dict(res)


    def _load_offs_for_run(self, offsets, start_i):
        print("offsets %i (%i)" % (start_i, len(offsets)))
        res = []
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = (-1, -1) if start_i >= len(offsets) else offsets[start_i]
            res.append((off_x, off_y))
            start_i = start_i + 1
        return res, start_i


    def start_workers(self):
        self.workers = [multiprocessing.Process(target=save_output_worker) for _ in range(N_PROC)]
        for p in self.workers:
            p.start()

    def stop_workers(self):
        for i in range(N_PROC):
            to_save.put((-1, [], -1))
        for p in self.workers:
            p.join()


    def _save_output(self, outs, output_i):
        log_res = np.argmax(outs[0], axis=3)
        angle_res = outs[1][:, :, :, 0]
        res = np.append(np.expand_dims(log_res, axis=1), np.expand_dims(angle_res, axis=1), axis=1)
        np.save(os.path.join(TMP_DIR, "segm_outputs_%i.npy" % output_i), res)


    def run_inference(self, fls, offsets, start_off_i=0):
        global to_save
        t1 = time.time()
        output_i = 0
        n_runs = math.ceil(len(offsets) / BATCH_SIZE)
        print(n_runs)
        for i in range(n_runs):
            run_start = time.time()
            run_offs, start_off_i = self._load_offs_for_run(offsets, start_off_i)
            cur_fr = 0

            feed_dict = self._feed_dict(run_offs, cur_fr, np.zeros((BATCH_SIZE, D, D, NUM_FILTERS), dtype=np.float32))
            #outs, last_priors, _ = self.sess.run([self.outputs, self.priors, self.batchnorm_updates_op], feed_dict=feed_dict)
            outs, last_priors = self.sess.run([self.outputs, self.priors], feed_dict=feed_dict)
            self._save_output(outs, output_i)
            to_save.put((output_i, run_offs, cur_fr))
            output_i += 1

            for cur_fr in range(1, len(fls)):
                feed_dict = self._feed_dict(run_offs, cur_fr, last_priors)
                #outs, last_priors, _ = self.sess.run([self.outputs, self.priors, self.batchnorm_updates_op], feed_dict=feed_dict)
                outs, last_priors = self.sess.run([self.outputs, self.priors], feed_dict=feed_dict)
                self._save_output(outs, output_i)
                to_save.put((output_i, run_offs, cur_fr))
                output_i += 1

            print("OFFSETS FINISHED %i - time: %.3f min" % (i, (time.time() - run_start)/60), flush=True)
        print("ALL FINISHED - time: %.3f min" % ((time.time() - t1)/60))


######## MAIN FUNCTION ##############

def find_detections():
    print(DATA_DIR)
    if os.path.exists(POS_DIR):
        shutil.rmtree(POS_DIR)
    os.mkdir(POS_DIR)
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    checkpoint_dir = os.path.join(DATA_DIR, "checkpoints", "unet2")
    checkpoint_nb = utils.find_last_checkpoint(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, "model_%06d.ckpt" % checkpoint_nb)

    fls = read_all_files()
    offsets = utils.get_offsets_in_frame(True)

    with DetectionInference() as model_obj:
        model_obj.build_model(checkpoint_nb, checkpoint_file)
        model_obj.start_workers()
        try:
            model_obj.run_inference(fls, offsets)
        finally:
            model_obj.stop_workers()
