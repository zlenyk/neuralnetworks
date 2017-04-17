import cifar_input
import numpy as np

classes = 10

def get_class_indices(_set):
    names = cifar_input.get_labels()
    labels_classes = np.argwhere(_set[1] == 1)
    classes_indices = []
    for i in range(classes):
        class_indices = np.argwhere(labels_classes[:,1] == i).ravel()
        classes_indices.append(class_indices)
    return np.asarray(classes_indices)
