from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage import transform

def get_labels(labels):
    return np.argmax(labels, axis=1)

def import_data():
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    train_data = mnist.train._images
    train_labels = mnist.train._labels
    validation_data = mnist.validation._images
    validation_labels = mnist.validation._labels
    test_data = mnist.test._images
    test_labels = mnist.test._labels
    return ((train_data, train_labels), (validation_data, validation_labels), (test_data, test_labels))

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
