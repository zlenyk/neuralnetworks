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

def get_labels():
    labels = unpickle('cifar/batches.meta')
    names = []
    for label in labels[b'label_names']:
        names.append(label.decode('UTF-8'))
    return names

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

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.3, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def normalize_batch(X, gamma, beta):
    #return tf.nn.lrn(X, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    mean, var = tf.nn.moments(X, axes=[0,1,2], keep_dims=False)
    return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-5)

def crop(images, c1, c2):
    images = np.delete(images, range(c1), axis=1)
    images = np.delete(images, range(images.shape[1]-8+c1,images.shape[1]), axis=1)
    images = np.delete(images, range(c2), axis=2)
    images = np.delete(images, range(images.shape[2]-8+c2,images.shape[2]), axis=2)
    return images

def crop_images(images):
    return crop(images, np.random.randint(0,8), np.random.randint(0,8))

def crop_centrally(images):
    return crop(images, 4, 4)

def count_accuracy(predicted_labels, true_labels):
    is_equal = tf.equal(predicted_labels, tf.argmax(y,1))
    return tf.reduce_mean(tf.cast(is_equal, "float")).eval({y: true_labels})

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
    current_shape = [3,24,24]
    for layer in layers:
        layer_dict = {}
        layer_dict['name'] = layer['name']
        if layer['name'] == CONV:
            layer_dict['W'] = weight_variable(  shape = [layer['shape'][0], layer['shape'][1], current_shape[0], layer['shape'][2]],
                                                stddev=5e-2,
                                                )
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
            X = normalize_batch(X, layer['var'], layer['mean'])
        elif layer['name'] == RES_BEG:
            temp = X
        elif layer['name'] == RES_END:
            X = X + temp
    return X

layers = [
    {'name':DROPOUT},
    {'name':NORM},

    {'name':CONV, 'shape':[3,3,32]},#32

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':RES_END},

    {'name':POOL},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,32]},#16
    {'name':RES_END},

    {'name':POOL},

    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':RES_END},

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,64]},#16
    {'name':RES_END},

    {'name':POOL},

    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},#16

    {'name':RES_BEG},
    {'name':NORM},
    {'name':RELU},
    {'name':CONV, 'shape':[3,3,128]},#16
    {'name':RES_END},

    {'name':NORM},
    {'name':RELU},
    {'name':FC, 'shape':[10]}
]

test_layers = [
    {'name':DROPOUT},
    {'name':NORM},
    {'name':POOL},

    {'name':CONV, 'shape':[3,3,2]},#32
    {'name':NORM},
    {'name':RELU},
    {'name':POOL},
    {'name':POOL},
    {'name':FC, 'shape':[10]}
]

(train_images, train_labels) = get_cifar_data('cifar/data_batch_1')
for i in range(2,6):
    train_pref = 'cifar/data_batch_'
    new_set = get_cifar_data(train_pref+str(i))
    (new_images,new_labels) = new_set
    train_images = np.concatenate((train_images, new_images))
    train_labels = np.concatenate((train_labels, new_labels))
test_set = get_cifar_data('cifar/test_batch')

#train_images = train_images[:1000]
#train_labels = train_labels[:1000]
display_step = 1
batch_size = 32
num_epochs = 20
input_size = 3072
classes = 10
log_file = open('results.txt', 'w')

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.98, staircase=True)

x = tf.placeholder(tf.float32, [None, 24,24,3])
y = tf.placeholder(tf.float32, [None, classes])
keep_probs = tf.placeholder(tf.float32)

models_probs = np.empty((0,len(test_set[0]),classes))
models_labels = []
sess = tf.InteractiveSession()

names = get_labels()
labels_classes = np.argwhere(test_set[1] == 1)
classes_indices = []
for i in range(classes):
    class_indices = np.argwhere(labels_classes[:,1] == i).ravel()
    classes_indices.append(class_indices)
classes_indices = np.asarray(classes_indices)

for k in range(5):
    layer_model = model(x, build_layers(layers))
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=layer_model, labels=y))

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    ema = tf.train.ExponentialMovingAverage(decay=0.995)
    var_avg = ema.apply(tf.trainable_variables())

    tf.global_variables_initializer().run()
    with tf.control_dependencies([optimizer]):
        training_op = tf.group(var_avg)

    for epoch in range(num_epochs):
        avg_cost = 0.
        total_batch = int(len(train_images)/batch_size)
        shuffle_in_unison(train_images, train_labels)
        for i in tqdm(range(total_batch)):
            batch_images = train_images[i*batch_size:(i+1)*batch_size]
            batch_images = crop_images(batch_images)
            batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
            _, c = sess.run([training_op, cost], feed_dict={x: batch_images,
                                                          y: batch_labels,
                                                          keep_probs: 1.0})
            # Compute average loss
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            (test_images, test_labels) = test_set
            test_batch_size = 100
            test_images = crop_centrally(test_images)

            evaluate_model = tf.argmax(layer_model, 1)
            predicted_labels = []
            for i in range(len(test_images)//test_batch_size):
                new_labels = evaluate_model.eval({
                    x: test_images[i*test_batch_size:(i+1)*test_batch_size],
                    y: test_labels[i*test_batch_size:(i+1)*test_batch_size],
                    keep_probs: 1.0})
                predicted_labels = np.append(predicted_labels, new_labels)
            predicted_labels = predicted_labels.astype(int)
            print("Model:", k, "Epoch:", '%04d' % (epoch+1),
                "cost=", "{:.9f}".format(avg_cost),
                "Accuracy:", count_accuracy(predicted_labels, test_labels))

            for j in range(classes):
                print(names[j], " "*(10-len(names[j])), "-",\
                    count_accuracy(
                        predicted_labels[classes_indices[j]],
                        test_labels[classes_indices[j]]))

    softmax = tf.nn.softmax(logits=layer_model)
    model_probs = np.empty((0,10))
    for i in range(len(test_images)//test_batch_size):
        probs = softmax.eval({
            x: test_images[i*test_batch_size:(i+1)*test_batch_size],
            keep_probs: 1.0
        })
        model_probs = np.concatenate((model_probs,probs))
    model_probs = np.asarray(model_probs)
    models_probs = np.append(models_probs, np.expand_dims(model_probs,axis=0), axis=0)

worst_probs = np.amin(models_probs, axis=0)
predicted_labels = tf.argmax(worst_probs, axis=1).eval()
print("Joined accuracy:", count_accuracy(predicted_labels, test_labels))
for j in range(classes):
    print(names[j], " "*(10-len(names[j])), "-",\
        count_accuracy(
            predicted_labels[classes_indices[j]],
            test_labels[classes_indices[j]]))

print("Optimization Finished!")
log_file.close()
