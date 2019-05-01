import tensorflow as tf
import math

from tensorflow import keras
from tensorflow.contrib import slim

from util.layer import GaussianKLD, GaussianSampleLayer, lrelu, GumbelSampleLayer

EPS = tf.constant(1e-10)


class CVAE(object):
    '''
    Semi-supervised VAE with Gumbel softmax with architecture described in:
        https://arxiv.org/abs/1611.01144
    '''
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        self._encoder_layers = {}
        self._generator_layers = {}

        with tf.variable_scope('Tau'):
            self.tau = tf.nn.relu(
                10. * tf.Variable(
                    tf.ones([1]),
                    name='tau')) + 0.1

        #self._generate = tf.make_template(
        #    'Generator', self._generator)
        self._classification_weight = tf.make_template(
            'Classifier', self._classifier_weight)
        self._classify_with_weight = tf.make_template(
            'Classifier', self._classifier_with_weight)

    def _classify(self, x, is_training=False, weight=None):
        if weight is None:
            weight = self._classification_weight()
        return self._classify_with_weight(x, is_training, weight)

    def _sanity_check(self):
        for net in ['encoder', 'generator', 'classifier']:
            assert len(self.arch[net]['output']) > 2
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel'])
            assert len(self.arch[net]['output']) == len(self.arch[net]['stride'])

    def _classifier_weight(self):
        n_layer = len(self.arch['classifier']['output'])
        subnet = self.arch['classifier']

        weight = {}
        h, w, input_channel = self.arch['hwc']
        for i in range(n_layer):
            kh, kw = subnet['kernel'][i]
            ic, oc = input_channel, subnet['output'][i]
            weight['Conv_{}/weights'.format(i)] = tf.get_variable(
                name='Conv_{}/weights'.format(i),
                shape=[kh, kw, ic, oc], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            weight['Conv_{}/biases'.format(i)] = tf.get_variable(
                name='Conv_{}/biases'.format(i),
                shape=[1, 1, 1, oc], dtype=tf.float32,
                initializer=tf.initializers.zeros()
            )

            sh, sw = subnet['stride'][i]
            h = math.ceil(float(h) / sh)
            w = math.ceil(float(w) / sw)
            input_channel = oc

        weight['fully_connected/weights'] = tf.get_variable(
            name='fully_connected/weights',
            shape=[h * w * input_channel, self.arch['y_dim']], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
        )
        weight['fully_connected/biases'] = tf.get_variable(
            name='fully_connected/biases',
            shape=[self.arch['y_dim']], dtype=tf.float32,
            initializer=tf.initializers.zeros(),
        )
        return weight

    def _classifier_with_weight(self, x, is_training, weight):
        n_layer = len(self.arch['classifier']['output'])
        subnet = self.arch['classifier']
        for i in range(n_layer):
            x = tf.nn.conv2d(
                x, weight['Conv_{}/weights'.format(i)],
                strides=[1] + subnet['stride'][i] + [1], padding='SAME')
            x = x + weight['Conv_{}/biases'.format(i)]
            x = tf.nn.relu(x)

        x = slim.flatten(x)
        y_logit = tf.matmul(x, weight['fully_connected/weights'])
        y_logit = y_logit + weight['fully_connected/biases']
        return y_logit

    def _encode(self, x, y, is_training):
        with tf.variable_scope('Encoder'):
            return self._encoder_network(x, y, is_training)

    def _encoder_network(self, x, y, is_training):
        n_layer = len(self.arch['encoder']['output'])
        subnet = self.arch['encoder']
        h, w, c = self.arch['hwc']

        with tf.variable_scope('y_guider'):
            self._encoder_layers['Dense_y2x'] = self._encoder_layers.get(
                'Dense_y2x', keras.layers.Dense(
                    h * w * c, use_bias=False,
                    activation=keras.activations.sigmoid, name='Dense_y2x')
            )
            y2x = self._encoder_layers['Dense_y2x'](y)
            y2x = tf.reshape(y2x, [-1, h, w, c])

        x = tf.concat([x, y2x], 3)

        for i in range(n_layer):
            self._encoder_layers['Conv_{}'.format(i)] = self._encoder_layers.get(
                'Conv_{}'.format(i),
                keras.layers.Conv2D(
                    filters=subnet['output'][i], kernel_size=subnet['kernel'][i],
                    strides=subnet['stride'][i], padding='same',
                    activation=None, use_bias=True, name='Conv_{}'.format(i))
            )
            x = self._encoder_layers['Conv_{}'.format(i)](x)
            x = keras.activations.relu(x)

        x = keras.layers.Flatten()(x)

        self._encoder_layers['Dense_z_mu'] = self._encoder_layers.get(
            'Dense_z_mu',
            keras.layers.Dense(self.arch['z_dim'], name='Dense_z_mu')
        )
        z_mu = self._encoder_layers['Dense_z_mu'](x)
        self._encoder_layers['Dense_z_lv'] = self._encoder_layers.get(
            'Dense_z_lv',
            keras.layers.Dense(self.arch['z_dim'], name='Dense_z_lv')
        )
        z_lv = self._encoder_layers['Dense_z_lv'](x)

        return z_mu, z_lv

    def _generate(self, z, y, is_training):
        with tf.variable_scope('Generator'):
            return self._generator_network(z, y, is_training)

    def _generator_network(self, z, y, is_training):
        ''' In this version, we only generate the target, so `y` is useless '''
        subnet = self.arch['generator']
        n_layer = len(subnet['output'])

        with tf.variable_scope('y_guider'):
            self._generator_layers['Dense_y2h'] = self._generator_layers.get(
                'Dense_y2h', keras.layers.Dense(
                    subnet['hidden_dim'], use_bias=False, name='Dense_y2h')
            )
            y2h = self._generator_layers['Dense_y2h'](y)

        self._generator_layers['Dense_z2h'] = self._generator_layers.get(
            'Dense_z2h', keras.layers.Dense(
                subnet['hidden_dim'], use_bias=True, name='Dense_z2h')
        )
        z2h = self._generator_layers['Dense_z2h'](z)

        h = keras.activations.relu(y2h + z2h)
        h = tf.expand_dims(h, axis=1)
        h = tf.expand_dims(h, axis=1)  # [b, 1, 1, c]
        for i in range(n_layer):
            self._generator_layers['ConvT_{}'.format(i)] = self._generator_layers.get(
                'ConvT_{}'.format(i), keras.layers.Conv2DTranspose(
                    filters=subnet['output'][i], kernel_size=subnet['kernel'][i],
                    strides=subnet['stride'][i], padding=subnet['padding'][i],
                    activation=None, use_bias=True, name='ConvT_{}'.format(i))
            )
            h = self._generator_layers['ConvT_{}'.format(i)](h)
            if i < n_layer - 1:
                h = keras.activations.relu(h)

        logit = h
        return tf.nn.sigmoid(logit), logit

    def circuit_loop(self, x, y_L=None):
        '''
        x:
        y_L: true label
        '''
        y_logit_pred = self._classify(x, is_training=self.is_training)
        y_u_pred = tf.nn.softmax(y_logit_pred)
        y_logsoftmax_pred = tf.nn.log_softmax(y_logit_pred)

        y_logit_sample = GumbelSampleLayer(y_logsoftmax_pred)
        y_sample = tf.nn.softmax(y_logit_sample / self.tau)
        y_logsoftmax_sample = tf.nn.log_softmax(y_logit_sample)

        z_mu, z_lv = self._encode(x, y_sample, is_training=self.is_training)
        z = GaussianSampleLayer(z_mu, z_lv)

        xh, xh_sig_logit = self._generate(z, y_sample, is_training=self.is_training)

        # labeled reconstruction
        if y_L is not None:
            z_mu_L, z_lv_L = self._encode(x, y_L, is_training=self.is_training)
            z_L = GaussianSampleLayer(z_mu, z_lv)

            xh_L, xh_sig_logit_L = self._generate(z, y_L, is_training=self.is_training)
        else:
            z_mu_L, z_lv_L, z_L = None, None, None
            xh_L, xh_sig_logit_L = None, None

        return dict(
            z=z,
            z_mu=z_mu,
            z_lv=z_lv,
            z_L=z_L,
            z_mu_L=z_mu_L,
            z_lv_L=z_lv_L,
            y_pred=y_u_pred,
            y_logsoftmax_pred=y_logsoftmax_pred,
            y_logit_pred=y_logit_pred,
            y_sample=y_sample,
            y_logit_sample=y_logit_sample,
            y_logsoftmax_sample=y_logsoftmax_sample,
            xh=xh,
            xh_sig_logit=xh_sig_logit,
            xh_L=xh_L,
            xh_sig_logit_L=xh_sig_logit_L,
        )

    def loss(self, x_u, x_l, y_l):
        unlabel = self.circuit_loop(x_u)
        labeled = self.circuit_loop(x_l, y_l)

        with tf.name_scope('loss'):
            # def mean_sigmoid_cross_entropy_with_logits(logit, truth):
            #     '''
            #     truth: 0. or 1.
            #     '''
            #     return tf.reduce_mean(
            #         tf.nn.sigmoid_cross_entropy_with_logits(
            #             logit,
            #             truth * tf.ones_like(logit)))

            loss = dict()

            # Note:
            #   `log p(y)` should be a constant term if we assume that y
            #   is equally distributed.
            #   That's why I omitted it.
            #   However, since y is now an approximation, I'm not sure
            #   whether omitting it is correct.

            # [TODO] What PDF should I use to compute H(y|x)?
            #   1. Categorical? But now we have a Continuous y @_@
            #   2. Gumbel-Softmax? But the PDF is.. clumsy

            with tf.name_scope('Labeled'):
                z_mu = labeled['z_mu_L']
                z_lv = labeled['z_lv_L']
                loss['KL(z_l)'] = tf.reduce_mean(
                    GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)))

                loss['log p(x_l)'] = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=slim.flatten(labeled['xh_sig_logit_L']),
                            labels=slim.flatten(x_l)),
                        1))

                loss['Labeled'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=labeled['y_logit_pred'],
                        labels=y_l))

                unlabeled_recon_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=slim.flatten(labeled['xh_sig_logit']),
                            labels=slim.flatten(x_l)),
                        1))

                # logit grad regularize
                unlabeled_recon_grad = tf.gradients(
                    unlabeled_recon_loss, labeled['y_logit_pred'])
                meta_update_lr = self.arch['training']['meta_inner_update_lr']
                grad_logit = tf.stop_gradient(labeled['y_logit_pred'])\
                    - meta_update_lr * unlabeled_recon_grad[0]
                loss['GradReg'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=grad_logit, labels=y_l))

            with tf.name_scope('Unlabeled'):
                z_mu = unlabel['z_mu']
                z_lv = unlabel['z_lv']
                loss['KL(z_u)'] = tf.reduce_mean(
                    GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)))

                loss['log p(x_u)'] = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=slim.flatten(unlabel['xh_sig_logit']),
                            labels=slim.flatten(x_u)),
                        1))

                y_prior = tf.ones_like(unlabel['y_sample']) / self.arch['y_dim']

                '''Eric Jang's code
                # loss and train ops
                kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/K)),[-1,N,K])
                KL = tf.reduce_sum(kl_tmp,[1,2])
                elbo=tf.reduce_sum(p_x.log_prob(x),1) - KL
                '''
                # https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb

                # J: I chose not to use 'tf.nn.softmax_cross_entropy'
                #    because it takes logits as arguments but we need
                #    to subtract `log p` before `mul` p
                loss['H(y)'] = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(
                            unlabel['y_pred'],
                            tf.log(unlabel['y_pred'] + EPS) - tf.log(y_prior)),
                        -1))

                # Using Gumbel-Softmax Distribution:
                #   1. Incorrect because p(y..y) is a scalar-- unless we can get
                #      the parametic form of the H(Y).
                #   2. The numerical value can be VERY LARGE, causing trouble!
                #   3. You should regard 'Gumbel-Softmax' as a `sampling step`

                # log_qy = GumbelSoftmaxLogDensity(
                #     y=unlabel['y_sample'],
                #     p=unlabel['y_pred'],
                #     tau=self.tau)
                # # loss['H(y)'] = tf.reduce_mean(- tf.mul(tf.exp(log_qy), log_qy))
                # loss['H(y)'] = tf.reduce_mean(- log_qy)

                # # [TODO] How to define this term? log p(y)
                # loss['log p(y)'] = - tf.nn.softmax_cross_entropy_with_logits(
                loss['log p(y)'] = 0.0

            with tf.name_scope('Meta'):
                meta_update_lr = self.arch['training']['meta_inner_update_lr']
                unsup_obj = loss['KL(z_l)'] + loss['log p(x_l)'] + \
                    loss['KL(z_u)'] + loss['log p(x_u)'] - loss['H(y)']
                weights = self._classification_weight()
                unsup_grads = tf.gradients(unsup_obj, list(weights.values()))
                unsup_grads = {k: v for k, v in zip(weights.keys(), unsup_grads)}
                fast_weights = {}
                for k, v in weights.items():
                    unsup_grad = unsup_grads[k]
                    # It is almost impossible to use hessian in this impl.
                    if self.arch['training']['meta_weight_no_hessian']:
                        unsup_grad = tf.stop_gradient(unsup_grad)
                    fast_weights[k] = weights[k] - meta_update_lr * unsup_grad
                fast_y_logit_pred = self._classify_with_weight(
                    x_l, is_training=self.is_training, weight=fast_weights)
                loss['meta_weight_labeled'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=fast_y_logit_pred, labels=y_l))

            loss['KL(z)'] = loss['KL(z_l)'] + loss['KL(z_u)']
            loss['Dis'] = loss['log p(x_l)'] + loss['log p(x_u)']
            loss['H(y)'] = loss['H(y)'] + loss['log p(y)']

            # For summaries
            with tf.name_scope('Summary'):
                tf.summary.histogram('unlabel/z', unlabel['z'])
                tf.summary.histogram('unlabel/z_mu', unlabel['z_mu'])
                tf.summary.histogram('unlabel/z_lv', unlabel['z_lv'])

                for k, v in loss.items():
                    tf.summary.scalar('{}'.format(k), v)

        return loss

    def encode(self, x):

        y_logit = self._classify(x, is_training=False)
        y = tf.nn.softmax(y_logit / self.tau)

        z_mu, z_lv = self._encode(x, y, is_training=False)
        return dict(mu=z_mu, log_var=z_lv, y=y)

    def classify(self, x):
        y_logit = self._classify(x, is_training=False)
        y = tf.nn.softmax(y_logit / self.tau)
        return y

    def decode(self, z, y, tanh=False):
        xh, _ = self._generate(z, y, is_training=False)
        return xh
