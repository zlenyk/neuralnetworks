#code written myself, exceptions:
# scale_and_rotate_image function (https://piotrmicek.staff.tcs.uj.edu.pl/machine-learning/)
# tensorflow tutorial

#gets to around 98.5% on mnist
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tqdm import *
import numpy as np
from skimage import transform
import sys

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

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
display_step = 1
batch_size = 32
num_epochs = 50
input_size = 784
classes = 10
hidden_layers = [500]
all_layers = [input_size] + hidden_layers + [classes]
training = True

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, classes])
keep_probs = tf.placeholder(tf.float32)
W = [tf.Variable(tf.random_normal([all_layers[i], all_layers[i+1]]))
    for i in range(len(all_layers)-1)]
b = [tf.Variable(tf.random_normal([all_layers[i+1]]))
    for i in range(len(all_layers)-1)]
# for batch normalization
scales = [tf.Variable(tf.random_normal([1])) + 1
    for i in range(len(all_layers)-1)]
biases = [tf.Variable(tf.random_normal([1]))
    for i in range(len(all_layers)-1)]

def model(X, W, b):
    for i in range(len(W)-1):
        layer_w = W[i]
        layer_b = b[i]
        X = tf.matmul(X, W[i]) + b[i]
        mean, var = tf.nn.moments(X,[0], keep_dims=True)
        X = tf.nn.batch_normalization(X, mean, var, biases[i], scales[i],1e-5)
        X = tf.nn.relu(X)
        X = tf.nn.dropout(X, keep_probs)
    return tf.matmul(X, W[-1]) + b[-1]

layer_model = model(x, W, b)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=layer_model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Training cycle
for epoch in range(num_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in tqdm(range(total_batch)):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = np.apply_along_axis(scale_and_rotate_image, 1, batch_x, [28,28])
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                      y: batch_y,
                                                      keep_probs: 0.8})
    if epoch % display_step == 0:
        predict = tf.equal(tf.argmax(layer_model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(predict, "float"))
        print("Epoch:", '%04d' % (epoch+1),\
                "Accuracy:", \
            accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_probs: 1.0}))

print("Optimization Finished!")
