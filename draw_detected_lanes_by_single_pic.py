import numpy as np
import cv2
from scipy.misc import imresize,imsave,imread
from IPython.display import HTML
from keras.models import load_model
import time

# Load Keras model
model = load_model('full_CNN_model.h5')
#model.summary()

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255
   
    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(prediction).astype(np.uint8)
    lane_drawn = np.dstack((blanks, prediction, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (540, 960, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result, lane_image

lanes = Lanes()

for i in range(1,3):
    origin_image = imread("pic/" + str(i) + ".jpg")
    tStart = time.time()#計時開始
    result, lane_image = road_lines(origin_image)
    tEnd = time.time()#計時結束
    print(tEnd - tStart)
    imsave( "pic/result_"+ str(i) + ".jpg",result )   
    imsave( "pic/mask_" + str(i) + ".jpg",lane_image)