import os
import json
import importlib

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics

from model.cvae import CVAE
from util.wrapper import save

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'tmp', 'log dir')
tf.app.flags.DEFINE_string('datadir', 'dataset', 'dir to dataset')
tf.app.flags.DEFINE_string('gpu_cfg', None, 'GPU config file')
tf.app.flags.DEFINE_string('config', 'configs/cvae_paper.json', 'Config file')


class MNISTLoader(object):
    '''
    I assume that 'download' and 'unzip' was done outside

    Training set: 60k
    Test set: 10k

    Subscript:
        u: unlabeled
        l: labeled
        t: test
    '''
    def __init__(self, path):
        self.x_l = self.load_images(
            filename=os.path.join(path, 'train-images-idx3-ubyte'),
            N=60000)
        self.y_l = self.load_labels(
            filename=os.path.join(path, 'train-labels-idx1-ubyte'),
            N=60000)
        self.x_t = self.load_images(
            filename=os.path.join(path, 't10k-images-idx3-ubyte'),
            N=10000)
        self.y_t = self.load_labels(
            filename=os.path.join(path, 't10k-labels-idx1-ubyte'),
            N=10000)
        self.x_u = None
        self.y_u = None  # [TODO] actually shouldn't exist

    def load_images(self, filename, N):
        '''
        Load MNIST from the downloaded and unzipped file
        into [N, 28, 28, 1] dimensions with [0, 1] values
        '''
        with open(filename) as f:
            x = np.fromfile(file=f, dtype=np.uint8)
        x = x[16:].reshape([N, 28, 28, 1]).astype(np.float32)
        x = x / 255.
        return x

    def load_labels(self, filename, N):
        ''' `int` '''
        with open(filename) as f:
            x = np.fromfile(file=f, dtype=np.uint8)
        x = x[8:].reshape([N]).astype(np.int32)
        return x

    # ==== Ad-hoc: the following are for SSL only ====
    def divide_semisupervised(self, N_u):
        '''
        Shuffle before splitting
        '''
        idx = np.arange(self.y_l.shape[0])
        self.x_l = self.x_l[idx]
        self.y_l = self.y_l[idx]

        self.x_u = self.x_l[:N_u]
        self.y_u = self.y_l[:N_u]

        self.x_l = self.x_l[N_u:]
        self.y_l = self.y_l[N_u:]


    def pick_supervised_samples(self, smp_per_class=10):
        ''' [TODO] improve it '''
        idx = np.arange(self.y_l.shape[0])
        rng_state = np.random.RandomState(seed=123)
        rng_state.shuffle(idx)
        x = self.x_l[idx]
        y = self.y_l[idx]
        index = list()
        for i in range(10):
            count = 0
            index_ = list()
            for j in range(y.shape[0]):
                if y[j] == i and count < smp_per_class:
                    index_.append(j)
                    count += 1
            index = index + index_
        # x_s = x[index]
        # y_s = y[index]
        return x[index], y[index]
    # =======================================


def get_optimization_ops(loss, arch):
    ''' Return a dict of optimizers '''
    optimizer_g = tf.train.RMSPropOptimizer(
        arch['training']['lr'])

    a = arch['training']['alpha']

    labeled_obj = loss['KL(z_l)'] + loss['log p(x_l)']
    if arch['training']['use_supervised_loss']:
        labeled_obj += a * loss['Labeled']  # + a * loss['H(y)']
    unlabeled_obj = loss['KL(z_u)'] + loss['log p(x_u)'] - loss['H(y)']
    obj_Ez = labeled_obj + unlabeled_obj

    optimize_list = []

    trainables = tf.trainable_variables()
    trainables = [v for v in trainables if 'Tau' not in v.name]
    y_embed_vars = tf.trainable_variables('y_embedding')
    classifier_vars = tf.trainable_variables('Classifier')
    generator_vars = tf.trainable_variables('Generator')
    encoder_vars = tf.trainable_variables('Encoder')

    # Labeled loss optimization
    optimize_list.append(
        optimizer_g.minimize(labeled_obj, var_list=trainables)
    )

    # Unlabeled loss optimization
    if arch['training']['fix_y_guider_for_unlabeled']:
        trainables_not_y_guider = [v for v in trainables if 'y_guider' not in v.name]
    else:
        trainables_not_y_guider = trainables
    optimize_list.append(
        optimizer_g.minimize(unlabeled_obj, var_list=trainables_not_y_guider)
    )

    # Meta loss optimization (logit level)
    if arch['training']['use_meta_gradient']:
        meta_obj = loss['GradReg'] * arch['training']['meta_outer_update_alpha']
        optimize_list.append(
            optimizer_g.minimize(meta_obj, var_list=generator_vars + encoder_vars)
        )

    # Meta weight loss optimization
    if arch['training']['use_meta_weight_gradient']:
        meta_weight_obj = loss['meta_weight_labeled'] * arch['training']['meta_outer_update_alpha']
        optimize_list.append(
            optimizer_g.minimize(meta_weight_obj, var_list=trainables)
        )
    return optimize_list


def halflife(t, N0=1., T_half=1., thresh=0.0):
    l = np.log(2.) / T_half
    Nt = (N0 - thresh) * np.exp(-l * t) + thresh
    return np.asarray(Nt).reshape([1,])


def reshape(b, h, w=None):
    if w is None: w = h
    b = np.reshape(b, [h, w, 28, 28])
    b = np.transpose(b, [0, 2, 1, 3])
    b = np.reshape(b, [h*28, w*28])
    return b


def make_thumbnail(y, z, arch, net):
    ''' Make a K-by-K thumbnail images '''
    with tf.name_scope('Thumbnail'):
        k = arch['y_dim']
        h, w, c = arch['hwc']

        y = tf.tile(y, [k, 1])

        z = tf.expand_dims(z, -1)
        z = tf.tile(z, [1, 1, k])
        z = tf.reshape(z, [-1, arch['z_dim']])

        xh = net.decode(z, y)  # 100, 28, 28, 1
        xh = tf.reshape(xh, [k, k, h, w, c])
        xh = tf.transpose(xh, [0, 2, 1, 3, 4])
        xh = tf.reshape(xh[:, :, :, :, 0], [k*h, k*w])
    return xh


def imshow(img_list, filename, titles=None):
    n = len(img_list)
    plt.figure(figsize=(10*n, 10))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(img_list[i], cmap='gray')
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])
    plt.savefig(filename)
    plt.close()


N_TEST = 10000


def main():

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    with open(args.config) as f:
        arch = json.load(f)
        print('\n{} is loaded\n'.format(args.config))
        json.dump(arch, open('{}/arch.json'.format(args.logdir), 'w'))

    dataset = MNISTLoader(args.datadir)
    dataset.divide_semisupervised(N_u=arch['training']['num_unlabeled'])
    x_s, y_s = dataset.pick_supervised_samples(
        smp_per_class=arch['training']['smp_per_class'])
    x_u = dataset.x_u
    x_t, y_t = dataset.x_t, dataset.y_t
    x_1, _ = dataset.pick_supervised_samples(smp_per_class=1)
    x_l_show = reshape(x_s, 10, arch['training']['smp_per_class'])
    imshow([x_l_show], os.path.join(args.logdir, 'x_labeled.png'))

    batch_size = arch['training']['batch_size']
    N_EPOCH = arch['training']['epoch']
    N_ITER = x_u.shape[0] // batch_size
    N_HALFLIFE = arch['training']['halflife']


    h, w, c = arch['hwc']
    X_u = tf.placeholder(shape=[None, h, w, c], dtype=tf.float32)
    X_l = tf.constant(x_s)
    Y_l = tf.one_hot(y_s, arch['y_dim'])

    print('\nmodule: {}, class: {}\n'.format(
        arch['model']['module'], arch['model']['class']))
    module = importlib.import_module(arch['model']['module'])
    net = eval('module.{}(arch)'.format(arch['model']['class']))
    loss = net.loss(X_u, X_l, Y_l)

    encodings = net.encode(X_u)
    Z_u = encodings['mu']
    Y_u = encodings['y']
    Xh = net.decode(Z_u, Y_u)

    label_pred = tf.argmax(Y_u, 1)
    Y_pred = tf.one_hot(label_pred, arch['y_dim'])
    Xh2 = net.decode(Z_u, Y_pred)

    ph_accuracy = tf.placeholder(shape=(), dtype=tf.float32)
    tf.summary.scalar('eval/accuracy', ph_accuracy)

    thumbnail = make_thumbnail(Y_u, Z_u, arch, net)

    optimize_list = get_optimization_ops(loss, arch=arch)


    if args.gpu_cfg:
        with open(args.gpu_cfg) as f:
            cfg = json.load(f)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=cfg['per_process_gpu_memory_fraction'])
        session_conf = tf.ConfigProto(
            allow_soft_placement=cfg['allow_soft_placement'],
            log_device_placement=cfg['log_device_placement'],
            inter_op_parallelism_threads=cfg['inter_op_parallelism_threads'],
            intra_op_parallelism_threads=cfg['intra_op_parallelism_threads'],
            gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
    else:
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=sess_config)

    # sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    train_summary_op = tf.summary.merge_all(scope='loss/Summary')
    eval_summary_op = tf.summary.merge_all(scope='eval')
    saver = tf.train.Saver()

    # ===============================
    # [TODO]
    #   1. batcher class
    #      1) for train and for test
    #      2) binarization
    #      3) shffule as arg
    #   5. TBoard (training tracker to monitor the convergence)
    # ===============================

    sqrt_bz = int(np.sqrt(batch_size))

    logfile = os.path.join(args.logdir, 'log.txt')

    try:
        step = 0
        for ep in range(N_EPOCH):
            np.random.shuffle(x_u)  # shuffle

            for it in range(N_ITER):
                step = ep * N_ITER + it

                idx = range(it * batch_size, (it + 1) * batch_size)
                tau = halflife(
                    step,
                    N0=arch['training']['largest_tau'],
                    T_half=N_ITER*N_HALFLIFE,
                    thresh=arch['training']['smallest_tau'])

                batch = np.random.binomial(1, x_u[idx])

                operations = [loss['Dis'], loss['KL(z)'], loss['H(y)'], loss['Labeled']]

                outputs = sess.run(
                    operations + optimize_list, {X_u: batch, net.tau: tau})
                l_x, l_z, l_y, l_l = outputs[:4]

                msg = 'Ep [{:03d}/{:d}]-It[{:03d}/{:d}]: Lx: {:6.2f}, KL(z): {:4.2f}, L:{:.2e}: H(u): {:.2e}'.format(
                    ep, N_EPOCH, it, N_ITER, l_x, l_z, l_l, l_y)
                print(msg)

                if it == (N_ITER -1):
                    # b, y, xh, xh2, summary = sess.run(    # TODO
                    #     [X_u, Y_u, Xh, Xh2, summary_op],  # TODO
                    b, y, xh, xh2, train_summary = sess.run(
                        [X_u, Y_u, Xh, Xh2, train_summary_op],
                        {X_u: batch,
                         net.tau: tau})
                    writer.add_summary(train_summary, step)

                    b = reshape(b, sqrt_bz)
                    xh = reshape(xh, sqrt_bz)
                    xh2 = reshape(xh2, sqrt_bz)

                    y = np.argmax(y, 1).astype(np.int32)
                    y = np.reshape(y, [sqrt_bz, sqrt_bz])

                    png = os.path.join(args.logdir, 'Ep-{:03d}-reconst.png'.format(ep))
                    with open(logfile, 'a') as f:
                        f.write(png + '  ')
                        f.write('Tau: {:.3f}\n'.format(tau[0]))
                        f.write(msg + '\n')
                        n, m = y.shape
                        for i in range(n):
                            for j in range(m):
                                f.write('{:d} '.format(y[i, j]))
                            f.write('\n')
                        f.write('\n\n')

                    imshow(
                        img_list=[b, xh, xh2],
                        filename=png,
                        titles=['Ground-truth',
                                'Reconstructed using dense label',
                                'Reconstructed using onehot label'])

                    # writer.add_summary(summary, step)  # TODO

                # Periodic evaluation
                if it == (N_ITER - N_ITER) and ep % arch['training']['summary_freq'] == 0:
                    # ==== Classification ====
                    y_p = list()
                    bz = 100
                    for i in range(N_TEST // bz):
                        b_t = x_t[i * bz: (i + 1) * bz]
                        b_t[b_t > 0.5] = 1.0  # [MAKESHIFT] Binarization
                        b_t[b_t <= 0.5] = 0.0
                        p = sess.run(
                            label_pred,
                            {X_u: b_t,
                             net.tau: tau})
                        y_p.append(p)
                    y_p = np.concatenate(y_p, 0)

                    # ==== Style Conversion ====
                    x_converted = sess.run(
                        thumbnail,
                        {X_u: x_1, Y_u: np.eye(arch['y_dim'])})

                    imshow(
                        img_list=[x_converted],
                        filename=os.path.join(
                            args.logdir,
                            'Ep-{:03d}-conv.png'.format(ep)))

                    # == Confusion Matrix ==
                    with open(logfile, 'a') as f:
                        cm = metrics.confusion_matrix(y_t, y_p)
                        n, m = cm.shape
                        for i in range(n):
                            for j in range(m):
                                f.write('{:4d} '.format(cm[i, j]))
                            f.write('\n')
                        acc = metrics.accuracy_score(y_t, y_p)
                        eval_summary = sess.run(eval_summary_op, {ph_accuracy: acc})
                        writer.add_summary(eval_summary, step)
                        f.write('Accuracy: {:.4f}\n'.format(acc))
                        f.write('\n\n')
    except KeyboardInterrupt:
        print('Aborted')

    finally:
        save(saver, sess, args.logdir, step)



if __name__ == '__main__':
    main()
