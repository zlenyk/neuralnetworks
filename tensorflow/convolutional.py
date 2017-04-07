#gets to around 92% on mnist

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tqdm import *
import numpy as np
from skimage import transform
import sys
import _pickle as cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = np.transpose(images, (0,2,3,1))
    return images

# e.g 2 -> [0, 0, 1, ...]
def format_labels(labels):
    np.arange(len(labels))
    ret_labels = np.zeros((len(labels), 10))
    ret_labels[np.arange(len(labels)), labels] = 1
    return ret_labels

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_cifar_data(filename):
    dict = unpickle(filename)
    return (_convert_images(dict[b'data']), format_labels(dict[b'labels']))

def scale_and_rotate_image(image, image_shape, angle_range=15.0, scale_range=0.1):
    angle = 2 * angle_range * np.random.random() - angle_range
    scale = 1 + 2 * scale_range * np.random.random() - scale_range

    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
    tf_scale = transform.SimilarityTransform(scale=scale)
    tf_shift = transform.SimilarityTransform(translation=[-14, -14])
    tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])

    image = transform.warp(image.reshape(image_shape),
                           (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse).ravel()
    return image

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
(train_images, train_labels) = get_cifar_data('cifar/data_batch_1')
for i in range(2,6):
    train_pref = 'cifar/data_batch_'
    new_set = get_cifar_data(train_pref+str(i))
    (new_images,new_labels) = new_set
    train_images = np.concatenate((train_images, new_images))
    train_labels = np.concatenate((train_labels, new_labels))
test_set = get_cifar_data('cifar/test_batch')


learning_rate = 0.001
display_step = 1
batch_size = 128
num_epochs = 30
input_size = 3072
classes = 10

x = tf.placeholder(tf.float32, [None, 32,32,3])
y = tf.placeholder(tf.float32, [None, classes])
keep_probs = tf.placeholder(tf.float32)

CONV = 'CONV'
POOL = 'POOL'
FC = 'FC'
DROPOUT = 'DROPOUT'
RELU = 'RELU'
NORM = 'NORM'
RES_BEG = 'RES_BEG'
RES_END = 'RES_END'

def build_layers(layers):
    model_layers = []
    current_shape = [3,32,32]
    for layer in layers:
        layer_dict = {}
        layer_dict['name'] = layer['name']
        if layer['name'] == CONV:
            layer_dict['W'] = weight_variable([layer['shape'][0], layer['shape'][1], current_shape[0], layer['shape'][2]])
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
        # in case of dropout or relu change nothing
        model_layers.append(layer_dict)
    return model_layers


def normalize_batch(X, gamma, beta):
    mean, var = tf.nn.moments(X, axes=[0,1,2], keep_dims=False)
    return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-5)

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
            normalize_batch(X, layer['var'], layer['mean'])
        elif layer['name'] == RES_BEG:
            temp = X
        elif layer['name'] == RES_END:
            X = X + temp
    return X

layers = [
    {'name':DROPOUT},
    {'name':CONV, 'shape':[3,3,32]},#32
    {'name':NORM},
    {'name':RELU},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,32]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,32]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':POOL},

    {'name':DROPOUT},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':POOL},

    {'name':DROPOUT},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':CONV, 'shape':[3,3,64]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':RES_END},

    {'name':POOL},

    {'name':FC, 'shape':[10]}
]

layer_model = model(x, build_layers(layers))

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=layer_model, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for epoch in range(num_epochs):
    avg_cost = 0.
    total_batch = int(len(train_images)/batch_size)
    shuffle_in_unison(train_images, train_labels)
    for i in tqdm(range(total_batch)):
        batch_images = train_images[i*batch_size:(i+1)*batch_size]
        batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_images,
                                                      y: batch_labels,
                                                      keep_probs: 1.0})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        predict = tf.equal(tf.argmax(layer_model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(predict, "float"))
        test_batch_size = 100
        acc = []
        for i in range(len(test_set[0])//test_batch_size):
            acc.append(accuracy.eval({x: test_set[0][i*test_batch_size:(i+1)*test_batch_size], y: test_set[1][i*test_batch_size:(i+1)*test_batch_size], keep_probs: 1.0}))
        acc = np.average(acc)
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost), "Accuracy:", acc)

print("Optimization Finished!")
