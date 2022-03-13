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
import itertools
import multiprocessing
from utils.paths import DATA_DIR, IMG_DIR, POS_DIR, TMP_DIR, CHECKPOINT_DIR
from utils.func import DS, GPU_NAME, NUM_LAYERS, NUM_FILTERS, CLASSES
from utils import func

N_PROC = 3
BATCH_SIZE = 4

tf.logging.set_verbosity(tf.logging.ERROR)

to_save = multiprocessing.Queue()

def read_all_files(img_dir):
    drs = [img_dir]
    fls = []
    for dr in drs:
        dr_fls = os.listdir(dr)
        dr_fls.sort()
        fls.extend(map(lambda fl: os.path.join(dr, fl), dr_fls))
    print("%i files" % len(fls), flush=True)
    return fls

def get_img_shape(img_dir):
    return func.read_img(0, img_dir).shape

# Generate offset (x,y) coordinates, to create 256x256 patches on each frame.
def generate_offsets_for_frame(img_shape):
    func.check_img_shape(img_shape) 
    h, w = img_shape
    xs = range(0, w, DS)
    ys = range(0, h, DS)
    return list(itertools.product(xs, ys))

######## POSTPROCESSING AND SAVING SEGMENTATION RESULTS ############

def save_output_worker(total_frames, output_dir):
    output = np.zeros((BATCH_SIZE, 2, DS, DS))
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
        print("processed frame %i patch %i, %i bees" % (cur_fr, (output_i-cur_fr)/total_frames + 1, res.shape[0]), flush=True)
        with open(os.path.join(output_dir, "%06d.txt" % cur_fr), 'a') as f:
            np.savetxt(f, res, fmt='%i', delimiter=',', newline='\n')

############# INFERENCE MODEL #####################

class DetectionInference:

    def __init__(self):
        self.batch_data = np.zeros((BATCH_SIZE, DS, DS, 1), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        None

    def build_model(self, checkpoint_dir):
        cpu, gpu = func.find_devices()
        tf_dev = gpu if gpu != "" else cpu
        with tf.Graph().as_default(), tf.device(cpu):

            # equals the number of batches processed * num_gpus.
            #update_ops = []
            self.is_train = False

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                self.placeholder_img = tf.placeholder(tf.float32, shape=(BATCH_SIZE, DS, DS, 1), name="images")
                self.placeholder_prior = tf.placeholder(tf.float32, shape=(BATCH_SIZE, DS, DS, NUM_FILTERS), name="prior")

                logits, last_relu, angle_pred = unet.create_unet2(NUM_LAYERS, NUM_FILTERS, self.placeholder_img, self.is_train, prev=self.placeholder_prior, classes=CLASSES)
                self.outputs= (logits, angle_pred)
                self.priors = last_relu
                tf.get_variable_scope().reuse_variables()
                #update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

            #self.batchnorm_updates_op = tf.group(*update_ops)
            self.saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            checkpoint_nb = func.find_last_checkpoint(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir, "model_%06d.ckpt" % checkpoint_nb)
            print("Restoring checkpoint %i.." % checkpoint_nb, flush=True)
            self.saver.restore(self.sess, checkpoint_file)
            init = tf.local_variables_initializer()
            self.sess.run(init)


    def _feed_dict(self, offs, fl, priors):
        img = func.read_img(img_file=fl)
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = offs[batch_i]
            if (off_x >= 0) and (off_y >= 0):
                self.batch_data[batch_i,:,:,0] = img[off_y:(off_y + DS), off_x:(off_x + DS)]
            else:
                self.batch_data[batch_i, :, :, :] = 0
        res = [(self.placeholder_prior, priors), (self.placeholder_img, self.batch_data)]
        return dict(res)


    def _load_offs_for_run(self, offsets, start_i):
        res = []
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = (-1, -1) if start_i >= len(offsets) else offsets[start_i]
            res.append((off_x, off_y))
            start_i = start_i + 1
        return res, start_i


    '''
    Start threads to process segmentation results and save .txt files in pos_dir, per frame.

    Params:
      total_frames: total number of frames to be process, used only for printing status.
      pos_dir: Dir to store .txt files (1 per frame) holding positions and angles of each bee detection.
    '''
    def start_workers(self, total_frames, pos_dir=POS_DIR):
        self.workers = [multiprocessing.Process(target=save_output_worker, args=(total_frames,pos_dir,)) for _ in range(N_PROC)]
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
        print("STARTING INFERENCE")
        for i in range(n_runs):
            run_offs, start_off_i = self._load_offs_for_run(offsets, start_off_i)
            last_priors = np.zeros((BATCH_SIZE, DS, DS, NUM_FILTERS), dtype=np.float32)
            for cur_fr,fl in enumerate(fls):
                feed_dict = self._feed_dict(run_offs, fl, last_priors)
                outs, last_priors = self.sess.run([self.outputs, self.priors], feed_dict=feed_dict)
                self._save_output(outs, output_i)
                to_save.put((output_i, run_offs, cur_fr))
                output_i += 1

        print("ALL FINISHED - time: %.3f min" % ((time.time() - t1)/60))


######## MAIN FUNCTION ##############

def find_detections(checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2"), img_dir=IMG_DIR, pos_dir=POS_DIR):
    print(DATA_DIR)
    if os.path.exists(pos_dir):
        shutil.rmtree(pos_dir)
    os.mkdir(pos_dir)
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    fls = read_all_files(img_dir)
    num_fls = len(fls)

    img_shape = get_img_shape(img_dir)
    offsets = generate_offsets_for_frame(img_shape)
    with DetectionInference() as model_obj:
        model_obj.build_model(checkpoint_dir)
        model_obj.start_workers(num_fls, pos_dir)
        try:
            model_obj.run_inference(fls, offsets)
        finally:
            model_obj.stop_workers()
