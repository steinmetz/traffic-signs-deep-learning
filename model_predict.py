# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:07:10 2019

@author: pietr
"""

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
#import serial
#import pyfirmata

#board = pyfirmata.Arduino('/dev/tty...') # oder usbserial.---


# 1=Stop
# 0=Vorfahrt
CATEGORIES = ["Vorfahrt", "Stop"]

# Serielle Schnittstelle
"""
ser = serial.Serial(
        port=' /dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)

"""

def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray') 
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = tf.keras.models.load_model("64x3-CNN.model")

"""
for x in range(1, 10):
    prediction = model.predict([prepare('image_%d.jpg' %x)])
    print(prediction)  # will be a list in a list.
    print(CATEGORIES[int(prediction[0][0])])
    if (prediction == [[0.]]):
        pred = 0
    if (prediction == [[1.]]):
        pred = 1
    print(pred)
    ser.write(pred)
    board.write(pred)
"""
    
prediction = model.predict([prepare('vorfahrt1neu.jpg')])
#print(prediction)  # will be a list in a list.
result = CATEGORIES[int(prediction[0][0])]
#print(result)
if (prediction == [[0.]]):
        pred = 0
if (prediction == [[1.]]):
        pred = 1
print(pred, result)
