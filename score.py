import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('mnist_outputs/new-model/mnist_cnn_model')
    model = tf.save_model_load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)["data"], dtype = np.float32)
    data = np.reshape(data, (-1, 28,28,1))
    out = model(data)
    y_hat = np.argmax(out, axis = 1)

    return y_hat.tolist()
