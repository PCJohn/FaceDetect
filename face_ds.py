from __future__ import division
import os
import numpy as np
import cv2
import random
import requests

FACE_PATH = "/home/prithvi/dsets/Faces/positive/"
NON_FACE_PATH = "/home/prithvi/dsets/Faces/negative/"
IN_SIZE = (32,32)   #Input dimensions of image for the network
SNAP_COUNT = 5      #Number of random snapshots per non-face image
MIN_LEN = 10        #Minimum length for the ranom snaphsots of non-faces
GOOD = [1,0]        #Vector output for faces
BAD = [0,1]         #Vector output for non-faces

FACE_COUNT = 36000  #Number of images of each class (positive and negative) in the dataset
TRAIN_SPLIT = int(0.6*FACE_COUNT)
VAL_SPLIT = int(0.2*FACE_COUNT) + TRAIN_SPLIT

#Method to generate multiple snapshots from an image
def rand_snap(img):
    r = []
    x = img.shape[0]
    y = img.shape[1]
    #Generate 5 snapshots of different sizes
    for i in range(SNAP_COUNT):
        snap_size = max([MIN_LEN,int(random.random()*200)])
        fx = int(random.random()*(x-snap_size))
        fy = int(random.random()*(y-snap_size))
        snap = img[fx:fx+snap_size,fy:fy+snap_size]
        r.append(cv2.resize(snap,IN_SIZE))
    return r

#Load the dataset for face/non face classification
def load_find_ds():
    ds = []
    #Load faces (positive samples)
    for n in os.listdir(FACE_PATH):
        name = FACE_PATH+n
        for img_path in os.listdir(name):
            t_img = cv2.resize(cv2.imread(name+"/"+img_path,0),IN_SIZE)
            ds.append((t_img, GOOD))
            ds.append((cv2.flip(t_img,1),GOOD)) #Use the horizontal mirror image
    random.shuffle(ds)
    ds = ds[:FACE_COUNT] 
    #Load non-faces (negative samples) from dataset
    nface_ds = []
    for n in os.listdir(NON_FACE_PATH):
        name = NON_FACE_PATH+n
        for img_path in os.listdir(name):
            t_img = cv2.imread(name+"/"+img_path,0)
            nface_ds.extend([(r,BAD) for r in rand_snap(t_img)])
            nface_ds.append((cv2.resize(t_img, IN_SIZE),BAD))
    random.shuffle(nface_ds)
    nface_ds = nface_ds[:FACE_COUNT]

    #Make the train, val and test sets: Ensure 50% for each set
    train = ds[:TRAIN_SPLIT]
    train.extend(nface_ds[:TRAIN_SPLIT])
    random.shuffle(train)
    val = ds[TRAIN_SPLIT:VAL_SPLIT]
    val.extend(nface_ds[TRAIN_SPLIT:VAL_SPLIT])
    random.shuffle(val)
    test = ds[TRAIN_SPLIT:]
    test.extend(nface_ds[TRAIN_SPLIT:])
    random.shuffle(test)

    trainX,trainY = map(np.array,zip(*train))
    valX,valY = map(np.array,zip(*val))
    testX,testY = map(np.array,zip(*test))

    return ((trainX,trainY),(valX,valY),(testX,testY))
