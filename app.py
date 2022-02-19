#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
from tensorflow import keras 
import urllib
import PIL
from PIL import Image
import cv2
import os
from resizeimage import resizeimage

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



app = FastAPI()




@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="100" name="text" value="URL of the aaa"/>  
    <input type="submit"/> 
    </form>'''


@app.post('/predict') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    #text = my_pipeline(text) #cleaning and preprocessing of the texts
    text=str(text)
    loaded_model = tf.keras.models.load_model('mnist.h5')
    
################
    urllib.request.urlretrieve(text, "testing.png")
    # img = Image.open("testing.png")

    # image = np.asarray(image.resize((224, 224)))[..., :3]
   # load_img_rz = img.resize((28,28))

    load_img_rz = np.asarray(Image.open("testing.png").resize((28,28)))




    numpyimgdata_reshaped_grey = cv2.cvtColor(load_img_rz, cv2.COLOR_BGR2GRAY)
    numpyimgdata_reshaped_grey = cv2.bitwise_not(numpyimgdata_reshaped_grey)
    your_new_array = np.expand_dims(numpyimgdata_reshaped_grey, axis=-1)
    numpyimgdata_reshaped = your_new_array.reshape(-1,28, 28, 1)
###############
    image_predicted_array = loaded_model.predict(numpyimgdata_reshaped)
    test_pred = np.argmax(image_predicted_array, axis=1)    ###Argmax is most commonly used in machine learning for finding the class with the largest predicted probability.

    x=test_pred[0]
    label={0: "T-/top",1: "Trouser",2: "Pullover",3: "Dress",4: "Coat/Jacket",5: "Sandal",6: "T-Shirt/top",7: "Sneaker",8: "Bag",9: "Ankle boot "}

    for i in label.keys():
        if i==x:
            y=label[i]

    
    img1 = cv2.imread("testing.png")

    hsv_frame = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    height, width, _ = img1.shape

    cx = int(width / 2)
    cy = int(height / 2)

    # Pick pixel value
    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]
    sat_value = pixel_center[1]
    l_value = pixel_center[2]

    #print(hue_value,sat_value,l_value)

    color = "Undefined"
    if hue_value == 0 and l_value < 30:
        color = "BLACK"
    elif hue_value < 5:
        color = "RED"
    elif hue_value < 22:
        color = "ORANGE"
    elif hue_value < 47 and l_value < 100:
        color = "YELLOW"
    elif hue_value < 47 and l_value > 200:
        color = "CREAMISH SHADE"
    elif hue_value < 78:
        color = "GREEN"
    elif hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "Cannot find"
        
    color

    
    return { #returning a dictionary as endpoint
        "TYPE": str(y),
        "COLOR": str(color)
    }
# uvicorn app:app --reload
# https://i.stack.imgur.com/E5dH1.png





