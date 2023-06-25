from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os
from PIL import Image



def model_loading():

    return load_model('model/')


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["tomato_model"] = model_loading()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: bytes = File(...)):
    with open('temp_tomato_leaf.jpg','wb') as imagef:
        imagef.write(file)
        image = Image.open("temp_tomato_leaf.jpg")
        array = tf.keras.preprocessing.image.img_to_array(image)
        array=array/255
        image = np.expand_dims(array, axis = 0)
        tomato_class = np.argmax(ml_models['tomato_model'].predict(image))
        
    return tomato_class