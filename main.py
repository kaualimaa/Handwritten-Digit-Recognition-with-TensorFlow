import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
from PIL import Image
from io import BytesIO

from preprocess_data import preprocess_data


class Request(BaseModel):
    data: str


PORT = 2940
HOST = '0.0.0.0'

app = FastAPI()
origins = ["*", "http://127.0.0.1:5500/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = tf.keras.models.load_model("model.h5")


def get_base64_data(data: str) -> str:
    return data.split(",")[1]


def base64_to_bytes(data: str) -> BytesIO:
    return BytesIO(base64.b64decode(data))


def bytes_to_image(data: BytesIO) -> Image:
    image = Image.open(data)
    image = image.resize((28, 28))

    return image


def image_to_array(data: Image) -> np.ndarray:
    return np.array(data)


@app.get("/")
def index():
    return {"message": "Welcome to the API! To make a prediction just make a request to '/predict' with the image..."}


@app.post("/predict")
async def predict(req: Request):
    data_base64 = get_base64_data(req.data)
    data_bytes = base64_to_bytes(data_base64)
    data_image = bytes_to_image(data_bytes)
    image_array = image_to_array(data_image)
    # Reshape image from (28, 28, 4) to (28, 28, 1)
    image_array = np.mean(image_array, axis=2, keepdims=True).reshape((1, 28, 28, 1))
    processed_img = preprocess_data(image_array)

    # Make predictions
    predictions_probs = model.predict(processed_img)
    prediction = np.argmax(predictions_probs)

    return {
        "data": str(prediction)
    }


if __name__ == "__main__":
    uvicorn.run("main:app", port=PORT, host=HOST, reload=True)
