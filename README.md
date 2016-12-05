# FaceDetect

This is a module for face detection with convolutional neural networks (CNNs). It uses a small CNN as a binary classifier to distinguish between faces and non-faces. A simple sliding window (with multiple windows of varying size) is used to locaize the faces in the image.

Requirements:
    
    1. TensorFlow
    2. OpenCV for Python

Dataset:

Positive samples (images of faces) for the classification were taken from 3 sources:

    1. Cropped labelled faces in the wild (http://conradsanderson.id.au/lfwcrop/)
    2. MIT CBCL face recognition database (http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html)
    3. 
    
Negative samples (non-faces) were random snapshots taken from 4 sources:

    1. Caltech 101 object categories (https://www.vision.caltech.edu/Image_Datasets/Caltech101/)
    2. MIT scenes ()
    3. Texture database (http://www-cvr.ai.uiuc.edu/ponce_grp/data/)
    4. Caltech cars (Rear) background dataset (http://www.robots.ox.ac.uk/~vgg/data3.html)
    5. Caltech houses dataset (http://www.robots.ox.ac.uk/~vgg/data3.html)
    
Random snapshots from these images were 

Demo:

The repo includes a pre-trained model: face_model. This can directly be used for localization. Sample usage of this model with FaceDetect.py can be seen in demo.py. Running this should output the resut of running the localizer on demo.jpg.
Demos on running the localizer on other images can be found here: <youtube link>

Output of demo.py:
<Image here>
