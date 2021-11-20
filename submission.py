
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
#from imutils import paths

import tensorflow as tf
tf.test.gpu_device_name()

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Flatten, Dense
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

#model location____CHANGE_PATH_LOCATION_ACCORDINGLY
model=load_model("/content/drive/My Drive/Sunny's team/mnist127.5_model.h5")
#model=load_model()

def predict(img):
    
    label_category=np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
  
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(500,300))
    img0=cv2.GaussianBlur(img,(5,5),0)
    _,thresh = cv2.threshold(img0,130, 255, cv2.THRESH_BINARY_INV)

    thresh=cv2.dilate(thresh,(5,5),iterations=3)
  
    connectivity = 4
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    captcha=[]
    for i in range(1, num_labels):
      a = stats[i, cv2.CC_STAT_AREA]
      if a >50:
          x = stats[i, cv2.CC_STAT_LEFT]
          y = stats[i, cv2.CC_STAT_TOP]
          w = stats[i, cv2.CC_STAT_WIDTH]
          h = stats[i, cv2.CC_STAT_HEIGHT]
      
          if w*h >2000:
            letter = thresh[y:y+h, x:x+w]
            letter=cv2.dilate(letter,(10,10),iterations=6)
            letter=cv2.copyMakeBorder(letter,12,12,12,12,cv2.BORDER_CONSTANT,value=0)
            
            letter=cv2.resize(letter,(28,28))
  
            letter=letter/127.5-1.0
            letter = np.expand_dims(letter, axis=2)
            letter= np.expand_dims(letter, axis=0)
            prediction=model.predict_classes(letter)

            captcha.append((x,label_category[prediction[0]]))

    captcha = sorted(captcha, key=lambda x: x[0])
    predicted_captcha=""
    for i in range(len(captcha)):
      predicted_captcha=predicted_captcha+captcha[i][1]
    
    '''plt.imshow(img,cmap='gray')
    plt.show()
    print("CAPTCHA_PREDICTED is: ",predicted_captcha)'''
    return predicted_captcha

def test():

    image_paths = ["/content/drive/My Drive/Sunny's team/test_images/test1.png","/content/drive/My Drive/Sunny's team/test_images/test2.png","/content/drive/My Drive/Sunny's team/test_images/test3.png","/content/drive/My Drive/Sunny's team/test_images/test4.png","/content/drive/My Drive/Sunny's team/test_images/test5.png"]
    correct_answers = ["23AC","0P4M","FCLDEJ","QP54","AgTLCFI"]
    score = 0

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) 
        answer = predict(image) 
        #print("answr is :",answer)
        if correct_answers[i] == answer:

            score += 10
    
    print('The final score of the participant is',score)

#......................................................FINAL MODEL PREDICTION......................................
image=cv2.imread("/content/drive/My Drive/sample/test101.png")
print("predicted captcha",predict(image))

if __name__ == "__main__":
    test()

