import os
import json
import importlib

import tensorflow as tf
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics

from model.cvae import CVAE
from util.wrapper import save
from util.datasets import MNISTLoader, CIFAR10Loader

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'tmp', 'log dir')
tf.app.flags.DEFINE_string('datadir', 'dataset', 'dir to dataset')
tf.app.flags.DEFINE_string('gpu_cfg', None, 'GPU config file')
tf.app.flags.DEFINE_string('config', 'configs/cvae_paper.json', 'Config file')


def get_optimization_ops(loss, arch):
    ''' Return a dict of optimizers '''
    optimizer_g = tf.train.RMSPropOptimizer(
        arch['training']['lr'])

    a = arch['training']['alpha']

    labeled_obj = 0
    if arch['training'].get('use_supervised_loss'):
        labeled_obj += a * loss['Labeled']  # + a * loss['H(y)']
    if arch['training'].get('use_unsupervised_loss', True):
        labeled_obj += loss['KL(z_l)'] + loss['log p(x_l)']

    unlabeled_obj = loss['KL(z_u)'] + loss['log p(x_u)'] - loss['H(y)']

    optimize_list = []

    trainables = tf.trainable_variables()
    trainables = [v for v in trainables if 'Tau' not in v.name]
    trainables = [v for v in trainables if 'classifier_batch_norm/gamma' not in v.name]
    y_embed_vars = tf.trainable_variables('y_embedding')
    classifier_vars = tf.trainable_variables('Classifier')
    classifier_vars = [v for v in classifier_vars if 'classifier_batch_norm/gamma' not in v.name]
    generator_vars = tf.trainable_variables('Generator')
    encoder_vars = tf.trainable_variables('Encoder')

    cls_bn_gamma = tf.trainable_variables('Classifier_1/classifier_batch_norm/gamma')
    cls_bn_beta = tf.trainable_variables('Classifier_1/classifier_batch_norm/beta')
    tf.summary.histogram('loss/Summary/classifier_batch_norm/gamma', cls_bn_gamma[0])
    tf.summary.histogram('loss/Summary/classifier_batch_norm/beta', cls_bn_beta[0])

    # Labeled loss optimization
    optimize_list.append(
        optimizer_g.minimize(labeled_obj, var_list=trainables)
    )

    # Unlabeled loss optimization
    if arch['training'].get('fix_y_guider_for_unlabeled', False):
        trainables_not_y_guider = [v for v in trainables if 'y_guider' not in v.name]
    else:
        trainables_not_y_guider = trainables
    if arch['training'].get('use_unsupervised_loss', True):
        optimize_list.append(
            optimizer_g.minimize(unlabeled_obj, var_list=trainables_not_y_guider)
        )

    # Meta loss optimization (logit level)
    if arch['training'].get('use_meta_gradient', False):
        meta_obj = loss['GradReg'] * arch['training']['meta_outer_update_alpha']
        optimize_list.append(
            optimizer_g.minimize(meta_obj, var_list=generator_vars + encoder_vars)
        )

    # Meta weight loss optimization
    if arch['training'].get('use_meta_weight_gradient', False):
        meta_weight_obj = loss['meta_weight_labeled'] * arch['training']['meta_outer_update_alpha']
        optimize_list.append(
            optimizer_g.minimize(meta_weight_obj, var_list=trainables)
        )
    return optimize_list


def halflife(t, N0=1., T_half=1., thresh=0.0):
    l = np.log(2.) / T_half
    Nt = (N0 - thresh) * np.exp(-l * t) + thresh
    return np.asarray(Nt).reshape([1,])


def reshape(b, nrow, ncol=None, arch=None):
    h, w, c = arch['hwc']
    if ncol is None: ncol = nrow
    b = np.reshape(b, [nrow, ncol, h, w, c])
    b = np.transpose(b, [0, 2, 1, 3, 4])
    b = np.reshape(b, [nrow*h, ncol*w, c])
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
        xh = tf.reshape(xh[:, :, :, :, :], [k*h, k*w, c])
    return xh


def imshow(img_list, filename, titles=None, image_pixel_type='binary'):
    n = len(img_list)
    plt.figure(figsize=(10*n, 10))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if image_pixel_type == 'binary':
            vis_img = img_list[i]
        elif image_pixel_type == 'continuous[-1,1]':
            vis_img = (img_list[i] + 1.0) / 2.0
        else:
            raise ValueError('Unknown image_pixel_type')

        if img_list[i].shape[2] == 1:
            plt.imshow(np.squeeze(vis_img, 2), cmap='gray')
        else:
            plt.imshow(vis_img)
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

    if arch['dataset']['name'] == 'mnist':
        dataset = MNISTLoader(args.datadir, config=arch['dataset'])
    elif arch['dataset']['name'] == 'cifar10':
        dataset = CIFAR10Loader(args.datadir, config=arch['dataset'])
    else:
        raise ValueError('Unknown dataset')

    dataset.divide_semisupervised(N_u=arch['training']['num_unlabeled'])
    x_s, y_s = dataset.pick_supervised_samples(
        smp_per_class=arch['training']['smp_per_class'])
    x_u = dataset.x_u
    x_t, y_t = dataset.x_t, dataset.y_t
    x_1, _ = dataset.pick_supervised_samples(smp_per_class=1)
    x_l_show = reshape(x_s, 10, arch['training']['smp_per_class'], arch=arch)
    imshow([x_l_show], os.path.join(args.logdir, 'x_labeled.png'),
           image_pixel_type=arch['image_pixel_type'])

    batch_size = arch['training']['batch_size']
    N_EPOCH = arch['training']['epoch']
    N_ITER = x_u.shape[0] // batch_size
    N_HALFLIFE = arch['training']['halflife']

    num_tile = math.ceil(float(x_u.shape[0]) / x_s.shape[0])
    x_s = np.tile(x_s, [num_tile, 1, 1, 1])
    y_s = np.tile(y_s, [num_tile])
    l_index = np.arange(x_s.shape[0])
    rng_state = np.random.RandomState(seed=234)

    h, w, c = arch['hwc']
    X_u = tf.placeholder(shape=[None, h, w, c], dtype=tf.float32)
    X_l = tf.placeholder(shape=[None, h, w, c], dtype=tf.float32)
    Y_l_ph = tf.placeholder(shape=[None], dtype=tf.int32)
    Y_l = tf.one_hot(Y_l_ph, arch['y_dim'])

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
            rng_state.shuffle(x_u)  # shuffle
            rng_state.shuffle(l_index)
            x_s = x_s[l_index]
            y_s = y_s[l_index].astype(np.int32)

            for it in range(N_ITER):
                step = ep * N_ITER + it

                idx = range(it * batch_size, (it + 1) * batch_size)
                tau = halflife(
                    step,
                    N0=arch['training']['largest_tau'],
                    T_half=N_ITER*N_HALFLIFE,
                    thresh=arch['training']['smallest_tau'])

                if arch['image_pixel_type'] == 'binary':
                    batch = np.random.binomial(1, x_u[idx])
                elif arch['image_pixel_type'] == 'continuous[-1,1]':
                    batch = x_u[idx]
                else:
                    raise ValueError('Unknown image_pixel_type')

                x_l_batch = x_s[idx]
                y_l_batch = y_s[idx]

                operations = [loss['Dis'], loss['KL(z)'], loss['H(y)'], loss['Labeled']]

                outputs = sess.run(
                    operations + optimize_list,
                    {X_u: batch, X_l: x_l_batch, Y_l_ph: y_l_batch, net.tau: tau})
                l_x, l_z, l_y, l_l = outputs[:4]

                msg = 'Ep [{:03d}/{:d}]-It[{:03d}/{:d}]: Lx: {:6.2f}, KL(z): {:4.2f}, L:{:.2e}: H(u): {:.2e}'.format(
                    ep, N_EPOCH, it, N_ITER, l_x, l_z, l_l, l_y)
                print(msg)

                if it == (N_ITER -1):
                    # b, y, xh, xh2, summary = sess.run(    # TODO
                    #     [X_u, Y_u, Xh, Xh2, summary_op],  # TODO
                    b, y, xh, xh2, train_summary = sess.run(
                        [X_u, Y_u, Xh, Xh2, train_summary_op],
                        {X_u: batch, X_l: x_l_batch, Y_l_ph: y_l_batch,
                         net.tau: tau})
                    writer.add_summary(train_summary, step)
                    b = reshape(b, sqrt_bz, arch=arch)
                    xh = reshape(xh, sqrt_bz, arch=arch)
                    xh2 = reshape(xh2, sqrt_bz, arch=arch)

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
                                'Reconstructed using onehot label'],
                        image_pixel_type=arch['image_pixel_type'])

                    # writer.add_summary(summary, step)  # TODO

                # Periodic evaluation
                #if it == (N_ITER - N_ITER) and ep % arch['training']['summary_freq'] == 0:
                if step % arch['training']['summary_freq'] == 0:
                    # ==== Classification ====
                    y_p = list()
                    bz = 100
                    for i in range(N_TEST // bz):
                        if arch['image_pixel_type'] == 'binary':
                            processed_x_t = x_t[i * bz: (i + 1) * bz]
                            processed_x_t[processed_x_t > 0.5] = 1.0  # [MAKESHIFT] Binarization
                            processed_x_t[processed_x_t <= 0.5] = 0.0
                        elif arch['image_pixel_type'] == 'continuous[-1,1]':
                            processed_x_t = x_t[i * bz: (i + 1) * bz]
                        else:
                            raise ValueError('Unknown image_pixel_type')
                        p = sess.run(label_pred, {X_u: processed_x_t, net.tau: tau})
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
                            'Ep-{:03d}-conv.png'.format(ep)),
                        image_pixel_type=arch['image_pixel_type'])

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
