import tensorflow as tf
import numpy as np
from tqdm import *
import image_utils
import math

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.3, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# ksize = 3,3 for overlapping pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def local_normalize(X):
    return tf.nn.lrn(X, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

#axes=[0,1,2]
def normalize_batch(X, gamma, beta, axes=[0,1,2]):
    mean, var = tf.nn.moments(X, axes=axes, keep_dims=False)
    return tf.nn.batch_normalization(X, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-5)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

CONV = 'CONV'
POOL = 'POOL'
FC = 'FC'
DROPOUT = 'DROPOUT'
RELU = 'RELU'
NORM = 'NORM'
RES_BEG = 'RES_BEG'
RES_END = 'RES_END'
classes = 10

def build_layers(layers):
    model_layers = []
    current_shape = [3,24,24]
    for layer in layers:
        layer_dict = {}
        layer_dict['name'] = layer['name']
        if layer['name'] == CONV:
            layer_dict['W'] = weight_variable(shape =[
                                                layer['shape'][0],
                                                layer['shape'][1],
                                                current_shape[0],
                                                layer['shape'][2]],
                                            stddev=math.sqrt(2.0/(9.0*current_shape[0])))
            layer_dict['b'] = bias_variable([layer['shape'][2]])
            current_shape[0] = layer['shape'][2]
        elif layer['name'] == FC:
            layer_dict['W'] = weight_variable([np.prod(current_shape), layer['shape'][0]])
            layer_dict['b'] = bias_variable([layer['shape'][0]])
            layer_dict['W_params'] = np.prod(current_shape)
            current_shape = layer['shape']
        elif layer['name'] == POOL:
            current_shape[1] = current_shape[1]//2
            current_shape[2] = current_shape[2]//2
        elif layer['name'] == NORM:
            layer_dict['mean'] = tf.Variable(tf.random_normal([current_shape[0]]))
            layer_dict['var'] = tf.Variable(tf.random_normal([current_shape[0]]))+1
            if 'axes' in layer:
                layer_dict['axes'] = layer['axes']
        # in case of dropout change nothing
        model_layers.append(layer_dict)
    return model_layers

def model(X, layers):
    temp = []
    for layer in layers:
        if layer['name'] == CONV:
            X = conv2d(X, layer['W']) + layer['b']
        elif layer['name'] == POOL:
            X = max_pool_2x2(X)
        elif layer['name'] == FC:
            X = tf.reshape(X, [-1, layer['W_params']])
            X = tf.matmul(X, layer['W']) + layer['b']
        elif layer['name'] == DROPOUT:
            X = tf.nn.dropout(X, keep_probs)
        elif layer['name'] == RELU:
            X = tf.nn.relu(X)
        elif layer['name'] == NORM:
            if 'axes' in layer:
                X = normalize_batch(X, layer['var'], layer['mean'], axes = layer['axes'])
            else:
                X = normalize_batch(X, layer['var'], layer['mean'])
        elif layer['name'] == RES_BEG:
            temp = X
        elif layer['name'] == RES_END:
            X = X + temp
    return X

layers = [
    {'name':NORM},

    {'name':CONV, 'shape':[3,3,128]},#32

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},#16
    {'name':RES_END},

    {'name':POOL},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':POOL},

    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':POOL},

    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},
    {'name':RES_END},

    {'name':NORM},
    {'name':RELU},
    {'name':DROPOUT},
    {'name':FC, 'shape':[512]},
    {'name':NORM, 'axes':[0]},
    {'name':RELU},
    {'name':FC, 'shape':[10]}
]

test_layers = [
    {'name':DROPOUT},
    {'name':NORM},
    {'name':POOL},

    {'name':CONV, 'shape':[3,3,2]},
    {'name':NORM},
    {'name':RELU},
    {'name':POOL},
    {'name':POOL},
    {'name':FC, 'shape':[10]}
]
saver_name = 'saver/saver2.ckpt'

x = tf.placeholder(tf.float32, [None, 24,24,3])
y = tf.placeholder(tf.float32, [None, classes])
keep_probs = tf.placeholder(tf.float32)

def _cost_op(model):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))

def loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# returns part of set (of batch size) and batch size
#(because may not be equal to batch_size in case of last iteration)
def batch_iterator(_set, batch_size):
    total_batch = int(len(_set)/batch_size)
    for i in range(total_batch):
        batch = _set[i*batch_size:(i+1)*batch_size]
        yield (batch, batch_size)
    if total_batch*batch_size < len(_set):
        yield (_set[total_batch*batch_size:], len(_set)-total_batch*batch_size)

class Model:
    def __init__(self, test=False):
        self.layer_model = None
        if test:
            self.layer_model = model(x, build_layers(test_layers))
        else:
            self.layer_model = model(x, build_layers(layers))

        self.cost_op = _cost_op(self.layer_model)
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost_op)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
        var_avg = self.ema.apply(tf.trainable_variables())
        self.ema_saver = tf.train.Saver(self.ema.variables_to_restore())
        self.saver = tf.train.Saver()
        self.training_op = tf.group(self.optimizer_op, var_avg)

    # predicted labels op
    def _predict_op(self):
        return tf.argmax(self.layer_model, 1)

    # return array [0,1,1,0,...] where 1 is equal
    def _count_equal_op(predicted):
        return tf.equal(predicted, tf.argmax(y, 1))

    def count_accuracy(self, images, labels, session):
        batch_size=100
        images_iter = batch_iterator(images, batch_size=batch_size)
        labels_iter = batch_iterator(labels, batch_size=batch_size)
        eq_sum = 0.
        self.ema_saver.restore(session, saver_name)
        while True:
            try:
                batch_images, _ = next(images_iter)
                batch_labels, _ = next(labels_iter)
                crops = [[0,0],[0,8],[8,0],[8,8],[4,4]]
                outputs = np.zeros((batch_size, 10))
                for crop in crops:
                    cropped_images = image_utils.crop(batch_images, crop[0], crop[1])
                    predictions = self.layer_model.eval({
                        x: cropped_images,
                        keep_probs: 1.0
                    })
                    outputs = outputs + predictions
                avg_outputs = outputs / len(crops)
                add_equal = tf.reduce_sum(tf.cast(
                    tf.equal(tf.argmax(avg_outputs, axis=1), tf.argmax(y, 1)), tf.float32)).eval({
                        y: batch_labels,
                })
                eq_sum += add_equal
            except StopIteration:
                self.saver.restore(session, saver_name)
                return eq_sum / len(labels)

    def train(self, images, labels, session):
        batch_size = 32
        total_batch = int(len(images)/batch_size)
        shuffle_in_unison(images, labels)
        self.saver.restore(session, saver_name)
        images_iter = batch_iterator(images, batch_size=batch_size)
        labels_iter = batch_iterator(labels, batch_size=batch_size)
        avg_cost = 0.
        for i in tqdm(range(total_batch)):
            try:
                batch_images, _ = next(images_iter)
                batch_images = image_utils.crop_images(batch_images)
                batch_labels, _ = next(labels_iter)
                _, c = session.run([self.training_op, self.cost_op], feed_dict={
                    x: batch_images,
                    y: batch_labels,
                    keep_probs: 0.7
                })
                avg_cost += c / total_batch
            except StopIteration:
                pass
        self.saver.save(session, saver_name)
        print("Cost:", avg_cost)

def get_model(test=False):
    return Model(test)
