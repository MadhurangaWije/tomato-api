from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os
from PIL import Image



def model_loading():

    return load_model('model/')

def get_label_name_by_id(id_str):
    id_class_name_dict = {'Late_blight': 2, 'healthy': 9, 'Early_blight': 1, 'Septoria_leaf_spot': 4, 'Tomato_Yellow_Leaf_Curl_Virus': 7, 'Bacterial_spot': 0, 'Target_Spot': 6, 'Tomato_mosaic_virus': 8, 'Leaf_Mold': 3, 'Spider_mites Two-spotted_spider_mite': 5}
    dict_id = { v:k for (k, v) in id_class_name_dict.items()}
    return dict_id[id_str]


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["tomato_model"] = model_loading()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def root():
    return "OK"


@app.post("/predict")
async def predict(file: bytes = File(...)):
    with open('temp_tomato_leaf.jpg','wb') as imagef:
        imagef.write(file)
        image = Image.open("temp_tomato_leaf.jpg")
        array = tf.keras.preprocessing.image.img_to_array(image)
        array=array/255
        image = np.expand_dims(array, axis = 0)
        tomato_class = np.argmax(ml_models['tomato_model'].predict(image))
        print(tomato_class)
    return { "disease_name": get_label_name_by_id(tomato_class.item())}