import os
import re
import time, math
from random import shuffle, randint
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from . import unet
from . import utils
from .utils import D, GPU_NAME, DET_DATA_DIR, NUM_LAYERS, NUM_FILTERS, CLASSES
from plots import segm_map

CHECKPOINT_DIR = os.path.join(utils.DATA_DIR, "checkpoints", "unet2")
BATCH_SIZE = 4
BASE_LR = 0.0001
MOVING_AVERAGE_DECAY = 0.9999

tf.logging.set_verbosity(tf.logging.ERROR)

acc_file = os.path.join(CHECKPOINT_DIR, "accuracy.csv")


def flip_h(data):
    data = np.flip(data, axis=2)

    for fr in range(data.shape[0]):
        ang = data[fr,2,:,:]
        is_1 = data[fr,1,:,:] == 1
        ang[is_1] = ((math.pi - ang[is_1]*2*math.pi) % (2 * math.pi)) / (2*math.pi)
        data[fr,2,:,:] = ang
    return data


def flip_v(data):
    data = np.flip(data, axis=3)

    for fr in range(data.shape[0]):
        ang = data[fr,2,:,:]
        is_1 = data[fr,1,:,:] == 1
        ang[is_1] = 1 - ang[is_1]
        data[fr,2,:,:] = ang
    return data


class TrainModel:

    def __init__(self, data_path, train_prop, dropout_ratio=0):
        self.data_path = data_path
        self.input_files = [f for f in os.listdir(data_path) if re.search('npz', f)]
        shuffle(self.input_files)
        self.train_prop = train_prop
        self.dropout_ratio = dropout_ratio

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        None

    #def _average_gradients(self, tower_grads):
    #    average_grads = []
    #    for grad_and_vars in zip(*tower_grads):
    #        grads = []
    #        for g, _ in grad_and_vars:
    #            expanded_g = tf.expand_dims(g, 0)
    #            grads.append(expanded_g)
    #        grad = tf.concat(axis=0, values=grads)
    #        grad = tf.reduce_mean(grad, 0)
    #        v = grad_and_vars[0][1]
    #        grad_and_var = (grad, v)
    #        average_grads.append(grad_and_var)
    #    return average_grads

    def _loss(self, img, label, weight, angle_label, prior):
        logits, last_relu, angle_pred = unet.create_unet2(NUM_LAYERS, NUM_FILTERS, img, self.is_train, prev=prior, classes=CLASSES)
        loss_softmax = unet.loss(logits, label, weight, CLASSES)
        loss_angle = unet.angle_loss(angle_pred, angle_label, weight)

        #tf.add_to_collection('losses', loss_softmax)
        #tf.add_to_collection('losses', loss_angle)
        #losses = tf.get_collection('losses')
        total_loss = loss_softmax + loss_angle #tf.add_n(losses, name='total_loss')
        return logits, total_loss, last_relu, angle_pred

    def build_model(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        cpu, gpu = utils.find_devices()
        tf_dev = gpu if gpu != "" else cpu

        with tf.Graph().as_default(), tf.device(cpu):
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            opt = tf.train.AdamOptimizer(learning_rate=BASE_LR)
            self.is_train = tf.placeholder(tf.bool, shape=[])
            self.placeholder_img = tf.placeholder(tf.float32, shape=(BATCH_SIZE, D, D, 1), name="images")
            self.placeholder_label = tf.placeholder(tf.uint8, shape=(BATCH_SIZE, D, D), name="labels")
            self.placeholder_weight = tf.placeholder(tf.float32, shape=(BATCH_SIZE, D, D), name="weight")
            self.placeholder_angle_label = tf.placeholder(tf.float32, shape=(BATCH_SIZE, D, D), name="angle_labels")
            self.placeholder_prior = tf.placeholder(tf.float32, shape=(BATCH_SIZE, D, D, NUM_FILTERS), name="prior")

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                    logits, loss, last_relu, angle_pred = self._loss(self.placeholder_img, self.placeholder_label,
                                                                     self.placeholder_weight, self.placeholder_angle_label,
                                                                     self.placeholder_prior)
                    self.outputs = (logits, loss, last_relu, angle_pred)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    grads = opt.compute_gradients(loss)

            #grads = self._average_gradients(grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            batchnorm_updates_op = tf.group(*update_ops)
            self.train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
            self.saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            checkpoint = utils.find_last_checkpoint(checkpoint_dir)
            if checkpoint > 0:
                print("Restoring checkpoint %i.." % checkpoint, flush=True)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % checkpoint))
                checkpoint += 1
            else:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                checkpoint = 0

            init = tf.local_variables_initializer()
            self.sess.run(init)
            return checkpoint

    def _accuracy(self, step, loss, logits, angle_preds, batch_data):
        batch_data = batch_data[:, step, :, :, :]

        pred_class = np.argmax(logits, axis=3)
        pred_angle = angle_preds[:, :, :, 0]

        lb = batch_data[:,1,:,:]
        angle = batch_data[:,2,:,:]
        is_bg = (lb == 0)
        is_fg = np.logical_not(is_bg)
        n_fg = np.sum(is_fg)
        bg = float(np.sum((pred_class[is_bg] == 0) & (pred_angle[is_bg] < 0)))/np.sum(is_bg)
        fg = 0
        fg_err = np.max(lb)
        angle_err = 0
        if n_fg > 0:
            fg = float(np.sum(pred_class[is_fg] != 0))/n_fg
            fg_err = np.mean(lb[is_fg] != pred_class[is_fg])
            angle_err = np.mean(np.abs(pred_angle[is_fg] - angle[is_fg]))
        return np.array([0, loss, bg, fg, fg_err, angle_err])


    def _sample_offsets(self, data):
        res = np.zeros((BATCH_SIZE, data.shape[0], data.shape[1], D, D))
        for i in range(BATCH_SIZE):
            off_x, off_y, fh, fv = randint(0, data.shape[2]-D), randint(0, data.shape[3]-D), randint(0, 1), randint(0, 1)
            cut_data = np.copy(data[:,:,off_x:(off_x+D),off_y:(off_y+D)])
            if fh:
                cut_data = flip_h(cut_data)
            if fv:
                cut_data = flip_v(cut_data)
            res[i] = cut_data
        return res, np.zeros((BATCH_SIZE, D, D, NUM_FILTERS), dtype=np.float32)


    def _input_batch(self, step, batch_data, last_relus, is_train):
        return {self.placeholder_img: np.resize(batch_data[:,step,0,:,:],(BATCH_SIZE, D, D, 1)),
                self.placeholder_label: batch_data[:,step,1,:,:], self.placeholder_angle_label: batch_data[:,step,2,:,:],
                self.placeholder_weight: batch_data[:,step,3,:,:],
                self.placeholder_prior: last_relus, self.is_train: is_train}


    def run_test(self, batch_data, last_step, last_relus, plot):
        t1 = time.time()

        accuracy_t = np.zeros((6))
        for step in range(last_step, batch_data.shape[1]):
            outs = self.sess.run(self.outputs, feed_dict=self._input_batch(step, batch_data, last_relus, False))
            last_relus = outs[2]
            accuracy_t += self._accuracy(step, outs[1], outs[0], outs[3], batch_data)
            if plot:
                for i in range(BATCH_SIZE):
                    im = segm_map.plot_segm_map_np(batch_data[i,step,0,:,:], batch_data[i,step,1,:,:])
                    im.save(os.path.join(utils.TMP_DIR, "%03d_%03d_segm_label.png" % (step, i)))

                    im = segm_map.plot_segm_map_np(batch_data[i, step, 0, :, :], np.argmax(outs[0][i], axis=2))
                    im.save(os.path.join(utils.TMP_DIR, "%03d_%03d_segm_out.png" % (step, i)))

                    im = segm_map.plot_angle_map_np(batch_data[i,step,0,:,:], batch_data[i,step,2,:,:])
                    im.save(os.path.join(utils.TMP_DIR, "%03d_%03d_angle_label.png" % (step, i)))

                    im = segm_map.plot_angle_map_np(batch_data[i,step,0,:,:], outs[3][i])
                    im.save(os.path.join(utils.TMP_DIR, "%03d_%03d_angle_out.png" % (step, i)))

        accuracy_t = accuracy_t / (batch_data.shape[1] - last_step)
        accuracy_t[0] = 1
        print("TEST - time: %.3f min, loss: %.3f, bg: %.3f, fg: %.3f, fg_err: %.3f, angle_err: %.3f" % ((time.time() - t1) / 60, accuracy_t[1], accuracy_t[2], accuracy_t[3], accuracy_t[4], accuracy_t[5]), flush=True)
        with open(acc_file, 'a') as f:
            np.savetxt(f, accuracy_t, fmt='%.5f', delimiter=',', newline='\n')


    def run_train_test_iter(self, itr, plot):
        file = self.input_files[itr % len(self.input_files)]
        npz = np.load(os.path.join(self.data_path, file))
        data = npz['data']
        t1 = time.time()
        train_steps = int(data.shape[0]*self.train_prop)
        batch_data, last_relus = self._sample_offsets(data)

        accuracy_t = np.zeros((6))
        for step in range(train_steps):
            _, outs = self.sess.run([self.train_op, self.outputs],
                                    feed_dict=self._input_batch(step, batch_data, last_relus, True))
            last_relus = outs[2]
            accuracy_t += self._accuracy(step, outs[1], outs[0], outs[3], batch_data)
        if plot:
            for i in range(BATCH_SIZE):
                im = segm_map.plot_segm_map_np(batch_data[i,step,0,:,:], np.argmax(outs[0][i], axis=2))
                im.save(os.path.join(utils.TMP_DIR, "%03d_%03d_segm_map.png" % (step, i)))
                im = segm_map.plot_angle_map_np(batch_data[i,step,0,:,:], outs[3][i])
                im.save(os.path.join(utils.TMP_DIR, "%03d_%03d_angle_map.png" % (step, i)))

        accuracy_t = accuracy_t / train_steps
        accuracy_t[0] = 0
        print("TRAIN - time: %.3f min, loss: %.3f, bg: %.3f, fg: %.3f, fg_err: %.3f, angle_err: %.3f" % ((time.time() - t1) / 60, accuracy_t[1], accuracy_t[2], accuracy_t[3], accuracy_t[4], accuracy_t[5]), flush=True)
        with open(acc_file, 'a') as f:
            np.savetxt(f, accuracy_t, fmt='%.5f', delimiter=',', newline='\n')

        if step < data.shape[0]:
            self.run_test(batch_data, train_steps, last_relus, plot)


def run_training(data_path=DET_DATA_DIR, train_prop=0.9, n_iters=10, checkpoint_dir=CHECKPOINT_DIR, plot=False):
    with TrainModel(data_path, train_prop) as model_obj:
        start_iter = model_obj.build_model(checkpoint_dir)
        for i in range(start_iter, start_iter + n_iters):
            print("ITERATION: %i" % i, flush=True)
            model_obj.run_train_test_iter(i, plot)
            model_obj.saver.save(model_obj.sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % i))

