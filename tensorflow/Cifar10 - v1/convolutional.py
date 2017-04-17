#All the code written myself (ZYgmunt Lenyk)
import cifar_input
import cifar_model
import tensorflow as tf
from skimage import transform
import sys
import math
import numpy as np
import utils

def count_accuracy(predicted_labels, true_labels):
    is_equal = tf.equal(predicted_labels, tf.argmax(y,1))
    return tf.reduce_mean(tf.cast(is_equal, "float")).eval({y: true_labels})

(train_images, train_labels), test_set = cifar_input.import_cifar()

display_step = 3
batch_size = 32
num_epochs = 30
classes = 10
saver_name = 'saver/saver.ckpt'
ema_name = 'saver/ema.ckpt'

#global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.1
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                           1000, 0.98, staircase=True)
models_probs = np.empty((0,len(test_set[0]),classes))
models_labels = []

sess = tf.InteractiveSession()
#indices = utils.get_class_indices(_set = test_set)

for k in range(5):
    model = cifar_model.get_model(test=False)
    tf.global_variables_initializer().run()
    model.saver.save(sess, saver_name)
    model.ema_saver.save(sess, ema_name)
    for epoch in range(num_epochs):
        model.train(train_images, train_labels, sess)

        #save_name = saver.save(tf.get_default_session(), saver_name)
        if epoch % display_step == 0:
            #model.ema_saver.restore(tf.get_default_session(), ema_name)
            (test_images, test_labels) = test_set
            print("Model:", k, "Epoch:", '%04d' % (epoch+1),
                "Accuracy:", model.count_accuracy(test_images, test_labels, sess))
            """
            for j in range(classes):
                print(names[j], " "*(10-len(names[j])), "-",\
                    count_accuracy(
                        predicted_labels[classes_indices[j]],
                        test_labels[classes_indices[j]]))
            #saver.restore(tf.get_default_session(), save_name)
            """
"""
    #ema_saver.restore(tf.get_default_session(), ema_name)
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
"""
