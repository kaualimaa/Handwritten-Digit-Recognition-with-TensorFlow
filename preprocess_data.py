import tensorflow as tf
import numpy as np


IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1


def reshape(data):
    return data.reshape(
        data.shape[0],
        IMG_WIDTH,
        IMG_HEIGHT,
        IMG_CHANNELS
    )


def normalize(data):
    return data / 255


def preprocess_data(data: np.ndarray | tf.Tensor):
    """
    Reshape and normalize the data
    :param data: Numpy array or Tensor
    :return: data: Data reshaped and normalized
    """
    assert (
        isinstance(data, np.ndarray) or isinstance(data, tf.Tensor)
    ), "Invalid datatype... NumPy array or TensorFlow tensor is required!"

    reshaped_data = reshape(data)
    normalized_data = normalize(reshaped_data)

    return normalized_data
