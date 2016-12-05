"""
Module for face detection.
Trains a small convolutional neural network for binary classification of an image as a face/non-face.
Uses a simple sliding window approach with variable sized windows to localize faces.
See demo.py for usage using the pre-trained model "face_model".

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""

import cv2
import tensorflow as tf
from tensorflow import nn
import tfac
import face_ds
import numpy as np

#Localization parameters
DET_SIZE = (300,300)    #Run all localization at a standard size
BLUR_DIM = (50,50)      #Dimension for blurring the face location mask
CONF_THRESH = 0.99      #Confidence threshold to mark a window as a face

X_STEP = 10     #Horizontal slide for the sliding window
Y_STEP = 10     #Vertical stride for the sliding window
WIN_MIN = 40    #Minimum sliding window size
WIN_MAX = 100   #Maximum sliding window size
WIN_STRIDE = 10   #Stride to increase the sliding window

#Build the net in the session
def build_net(sess):
    in_len = 32
    in_dep = 1

    x_hold = tf.placeholder(tf.float32,shape=[None,in_dep*in_len*in_len])
    y_hold = tf.placeholder(tf.float32,shape=[None,2])
    keep_prob = tf.placeholder(tf.float32)

    xt = tf.reshape(x_hold,[-1,in_len,in_len,in_dep])

    #Layer 1 - 5x5 convolution
    w1 = tfac.weight([5,5,in_dep,4])
    b1 = tfac.bias([4])
    c1 = nn.relu(nn.conv2d(xt,w1,strides=[1,2,2,1],padding='VALID')+b1)
    o1 = c1

    #Layer 2 - 3x3 convolution
    w2 = tfac.weight([3,3,4,16])
    b2 = tfac.bias([16])
    c2 = nn.relu(nn.conv2d(o1,w2,strides=[1,2,2,1],padding='VALID')+b2)
    o2 = c2

    #Layer 3 - 3x3 convolution
    w3 = tfac.weight([3,3,16,32])
    b3 = tfac.bias([32])
    c3 = nn.relu(nn.conv2d(o2,w3,strides=[1,1,1,1],padding='VALID')+b3)
    o3 = c3

    dim = 32 * 4*4
        
    #Fully connected layer - 600 units
    of = tf.reshape(o3,[-1,dim])
    w4 = tfac.weight([dim,600])
    b4 = tfac.bias([600])
    o4 = nn.relu(tf.matmul(of,w4)+b4)

    o4 = nn.dropout(o4, keep_prob)

    #Output softmax layer - 2 units
    w5 = tfac.weight([600,2])
    b5 = tfac.bias([2])
    y = nn.softmax(tf.matmul(o4,w5)+b5)

    sess.run(tf.initialize_all_variables())

    return y,x_hold,y_hold,keep_prob

#Method to run the training
def train_net():
    train,val,test = face_ds.load_find_ds()
    sess = tfac.start_sess()
    y,x_hold,y_hold,keep_prob = build_net(sess)
    acc = tfac.train(sess,
                    y,
                    x_hold,
                    y_hold,
                    keep_prob,
                    train[0],train[1],
                    test[0],test[1],
                    lrate=1e-4,
                    epsilon=1e-16,
                    n_epoch=8,
                    batch_size=100,
                    print_epoch=1,
                    save_path=model_path)
    print "Accuracy:",acc
    sess.close()

#Basic sliding window detector to find faces
#Returns an image showing only the faces along with the sliding window mask (before blurring)
def localize(img,model_path):
    sess = tfac.start_sess()
    y,x_hold,y_hold,keep_prob = build_net(sess)
    saver = tf.train.Saver()
    saver.restore(sess,model_path)

    #Run all detection at a fixed size
    img = cv2.resize(img,DET_SIZE)
    mask = np.zeros(img.shape)
    #Run sliding windows of different sizes
    for bx in range(WIN_MIN,WIN_MAX,WIN_STRIDE):
        by = bx
        for i in xrange(0, img.shape[1]-bx, X_STEP):
            for j in xrange(0, img.shape[0]-by, Y_STEP):
                sub_img = cv2.resize(img[i:i+bx,j:j+by],face_ds.IN_SIZE)
                X = sub_img.reshape((1,tfac.dim_prod(face_ds.IN_SIZE)))
                out = y.eval(session=sess,feed_dict={x_hold:X,keep_prob:1})[0]
                if out[0] >= CONF_THRESH:
                    mask[i:i+bx,j:j+by] = mask[i:i+bx,j:j+by]+1

    sess.close()
    mask = np.uint8(255*mask/np.max(mask))
    faces = img*(cv2.threshold(cv2.blur(mask,BLUR_DIM),0,255,cv2.THRESH_OTSU)[1]/255)
    return (faces,mask)
