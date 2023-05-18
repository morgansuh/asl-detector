import cv2 
import numpy as np
import math
import os
from tensorflow import keras
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import numpy as np


new_model = keras.models.load_model(os.path.join('models','imageclassifier.h5'))
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#setting up camera + hand detector 
capture = cv2.VideoCapture(0)
det = HandDetector(maxHands=1) #only take one hand 
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

offset = 20

save_folder = "Data/S"

while True: 
    s, img = capture.read()
    hand, img = det.findHands(img)

    #bounding box
    if hand:
        hand = hand[0]
        x, y, w, h = hand['bbox'] 

        #keep images square
        background = np.ones((300,300,3),np.uint8)*255
        

        hand_cropped = img[y-offset : y+h+offset , x-offset : x+w+offset]

        height, width, _ = hand_cropped.shape

        aspect_ratio= h/w

        if aspect_ratio > 1:
            k = 300/h
            w_calculated = k * w
            w_cal= math.ceil(w_calculated)
            image_resized = cv2.resize(hand_cropped,(w_cal,300))
            height, width, _ = image_resized.shape
            #center
            width_gap = math.ceil((300 - w_cal)/2)

            background[:,width_gap:w_cal+width_gap] = image_resized
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            resize = tf.image.resize(background, (224,224))
            image_array = np.asarray(resize)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = new_model.predict(data)
            index = np.argmax(prediction)
            predicted_letter = alphabet[index]
            
        
        else:
            k = 300/w
            h_calculated = k * h
            h_cal= math.ceil(h_calculated)
            image_resized = cv2.resize(hand_cropped,(300,h_cal))
            height, width, _ = image_resized.shape
            #center
            height_gap = math.ceil((300 - h_cal)/2)

            background[height_gap:h_cal+height_gap,:] = image_resized
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            resize = tf.image.resize(background, (224,224))
            image_array = np.asarray(resize)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = new_model.predict(data)
            index = np.argmax(prediction)
            predicted_letter = alphabet[index]


        cv2.putText(img,predicted_letter,(x,y+offset),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)
        cv2.imshow("Hand_cropped" , hand_cropped)
        cv2.imshow("Image_background" , background)
        

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    counter= 0