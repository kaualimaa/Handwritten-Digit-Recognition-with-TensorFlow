import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from typing import List


def create_callbacks(
        patience: int,
        hist_freq: int,
        log_dir: str
) -> List[tf.keras.callbacks.Callback]:
    """
    Create model's callbacks
    :param patience: EarlyStopping patience
    :param hist_freq: TensorBoard histogram frequency
    :param log_dir: TensorBoard log dir
    :return:
    """
    assert isinstance(patience, int), "`patience` needs to be of type int"
    assert isinstance(hist_freq, int), "`hist_freq` needs to be of type int"
    assert isinstance(log_dir, str), "`log_dir` needs to be of type str"

    early_stopping_loss = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=1,
        patience=patience
    )

    early_stopping_accuracy = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        patience=patience
    )

    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=hist_freq
    )

    return [early_stopping_loss, early_stopping_accuracy, tensorboard_callback]


if __name__ == "__main__":
    callbacks = create_callbacks(patience=10, log_dir='aa', hist_freq=1)
    print(callbacks)
