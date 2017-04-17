import numpy as np

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
