"""
Accessory methods for using TensorFlow -- Mostly taken out from the TensorFlow tutorials!

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""

import tensorflow as tf
import numpy as np
import random

#Make weight and bias variables -- From the TensorFlow tutorial
def weight(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)

def bias(shape):
    intial = tf.constant(0.1, shape=shape)
    return tf.Variable(intial)

#Finds the product of a dimension tuple to find the total legth
def dim_prod(dim_arr):
    return np.prod([d for d in dim_arr if d != None])

#Start a TensorFlow session
def start_sess():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    return sess

#Train the model
def train(sess, y, x_hold, y_hold, keep_prob, X, Y, valX, valY, lrate=0.5, epsilon=1e-8, n_epoch=100, batch_size=10, print_epoch=100, save_path=None):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_hold*tf.log(y+1e-10), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=lrate,epsilon=epsilon).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_hold,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #Flatten the input images for the placeholder
    flat_len = dim_prod(x_hold._shape_as_list())
    X = X.reshape((X.shape[0],flat_len))

    print 'Starting training session...'

    sess.run(tf.initialize_all_variables())
    batch_num = 0
    batches = batchify(X,Y,batch_size)
    print 'Number of batches:',len(batches)
    for i in range(n_epoch):
        avg_acc = 0
        random.shuffle(batches)
        for batchX,batchY in batches:
            avg_acc = avg_acc + accuracy.eval(session=sess, feed_dict={x_hold:batchX, y_hold:batchY, keep_prob:1})
            train_step.run(session=sess,feed_dict={x_hold:batchX, y_hold:batchY, keep_prob:0.5})
        print 'Epoch '+str(i)+': '+str(avg_acc/len(batches))
    if (not valX is None) & (not valY is None):
        #Validation
        valX = valX.reshape((valX.shape[0],flat_len))
        val_accuracy = accuracy.eval(session=sess,feed_dict={x_hold:valX, y_hold:valY, keep_prob:1})
        print 'Val acc:',val_accuracy

    if not save_path is None:
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess,save_path)
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(save_path+'_graph',sess.graph)
        writer.flush()
        writer.close()
        print 'Model saved'
    return val_accuracy

#Test a model
def test(sess, X, Y, model_path):
    correct_prediction = tf.equal(tf.argmax(self.net,1), tf.argmax(self.y_hold,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,model_path)
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    test_accuracy = accuracy.eval(session=sess,feed_dict={x_hold:X,y_hold:Y,keep_prob:1})
    return test_accuracy

#Split to mini batches
def batchify(X, Y, batch_size):
    batches = [(X[i:i+batch_size],Y[i:i+batch_size]) for i in xrange(0,X.shape[0],batch_size)]
    random.shuffle(batches)
    return batches
