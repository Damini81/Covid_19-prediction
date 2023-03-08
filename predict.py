import tensorflow as tf
#load the model
classifierLoad = tf.keras.models.load_model('covid_model.h5')
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

test_image1 = cv2.imread('1 (2).png',0)

test_image = image.load_img('1 (2).png', target_size = (200,200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)
if result[0][0] == 1:
    print("covid")
    test_image1 = cv2.resize(test_image1, (200,200))
    cv2.imshow('sampleimage',test_image1)
    cv2.waitKey(0)

elif result[0][1] == 1:
    print("normal")
    test_image1 = cv2.resize(test_image1, (200,200))
    cv2.imshow('sampleimage',test_image1)
    cv2.waitKey(0)

else:
    print("can't recognize")
