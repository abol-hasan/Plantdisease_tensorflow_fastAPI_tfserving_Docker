from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("item.html", {"request": request})


class_name = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


def bytes_to_images(image):
    im_size = 224
    img = BytesIO(image)
    img = Image.open(img)
    img = np.array(img)
    img = tf.image.resize(img, [im_size, im_size])
    return img


@app.post("/predict", response_class=HTMLResponse)
async def prediction(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image_np = bytes_to_images(contents)
    img = tf.expand_dims(image_np, axis=0)

    img = img.numpy()
    data = {"instances": img.tolist()}
    response = requests.post(
        "http://server:8501/v1/models/tf_serving_plant:predict", json=data
    )
    predicted = response.json()["predictions"][0]

    predicted_label = class_name[np.argmax(predicted)]
    confidence = max(predicted)

    confidence = "{:.5f}".format(confidence)

    return templates.TemplateResponse(
        "item.html",
        context={
            "request": request,
            "predicted_label": predicted_label,
            "confidence": confidence,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    
