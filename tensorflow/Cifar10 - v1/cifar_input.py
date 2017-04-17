import _pickle as cPickle
import numpy as np
import os

data_dir = '../cifar'

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def get_labels():
    labels = unpickle(os.path.join(data_dir,'batches.meta'))
    #labels = unpickle('cifar/batches.meta')
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

def get_cifar_data(filename):
    dict = unpickle(filename)
    return (_convert_images(dict[b'data']), format_labels(dict[b'labels']))

def import_cifar():
    (train_images, train_labels) = get_cifar_data(os.path.join(data_dir,'data_batch_1'))
    for i in range(2,6):
        train_pref = 'data_batch_'+str(i)
        new_set = get_cifar_data(os.path.join(data_dir,train_pref))
        (new_images,new_labels) = new_set
        train_images = np.concatenate((train_images, new_images))
        train_labels = np.concatenate((train_labels, new_labels))
    test_set = get_cifar_data(os.path.join(data_dir,'test_batch'))
    return (train_images, train_labels), test_set
