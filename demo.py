"""
Demo for the face detection. Runs the sliding window detector on demo.jpg
Displays the input image, the localized faces and the sliding window mask

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""

import FaceFinder
import cv2

model_path = 'face_model'
img = cv2.imread("demo.jpg",0)
faces,mask = FaceFinder.localize(img,model_path)
cv2.imshow("faces",faces)
cv2.imshow("sliding window mask",mask)
cv2.imshow("input image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
