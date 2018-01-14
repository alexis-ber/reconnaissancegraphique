from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
import cv2
import numpy as np
from keras.applications import VGG16

model = load_model('model_graph2.h5')
model.load_weights('poids_graph.h5')

nb_test = 90

for i in range(1,nb_test):

    img = cv2.imread('testfin/grph/Img' + str(1400 + i) + '.jpg')
    img = cv2.resize(img, (150,150))
    img = np.reshape(img, [1,150,150, 3])
    pred = (model.predict(img))
    print(pred)
    print(sum(sum(pred)))
for i in range(1,nb_test):

    img = cv2.imread('testfin/txt/Text' + str(1400 + i) + '.jpg')
    img = cv2.resize(img, (150,150))
    img = np.reshape(img, [1,150,150, 3])
    pred = (model.predict(img))
    print(pred)
    print(sum(sum(pred)))