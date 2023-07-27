import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.activations import relu, softmax
from keras.models import Sequential
import numpy as np


available_cpus = tf.config.experimental.list_physical_devices("GPU")
if available_cpus:
    for gpu in available_cpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU...")


def create_model(input_shape: tuple) -> tf.keras.models.Sequential:
    sequential_model = Sequential()

    sequential_model.add(Conv2D(
        input_shape=input_shape,
        kernel_size=5,
        filters=8,
        strides=1,
        activation=relu
    ))

    sequential_model.add(MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(Conv2D(
        kernel_size=5,
        filters=16,
        strides=1,
        activation=relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    sequential_model.add(Flatten())

    sequential_model.add(Dropout(.2))

    sequential_model.add(Dense(
        units=10,
        activation=softmax,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    # Compile the model
    sequential_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
        metrics=["accuracy"]
    )

    return sequential_model


def evaluate_model(
        model_to_evaluate: Sequential,
        model_history: dict,
        test_data: tuple[np.ndarray, np.ndarray]
):
    loss, accuracy = model_to_evaluate.evaluate(*test_data)

    print(f"\nModel evaluate (on test data):\n"
          f"Loss: {loss}\n"
          f"Accuracy: {(accuracy*100):.2f}%")

    print(f"\nTraining evaluation (last epoch):\n"
          f"Loss: {model_history['loss'][-1]:.3f}\n"
          f"Accuracy: {model_history['accuracy'][-1]*100:.3f}%")

    print(f"\nValidation evaluation (last epoch):\n"
          f"Loss: {model_history['val_loss'][-1]:.3f}\n"
          f"Accuracy: {model_history['val_accuracy'][-1]*100:.3f}%")


if __name__ == "__main__":
    from load_data import load_data
    from callbacks import create_callbacks
    from preprocess_data import preprocess_data
    from datetime import datetime
    from keras.utils import plot_model

    log_dir = f"./.logs/fit/{datetime.now().strftime('%Y%m%d - %H%M%S')}"

    # Load data
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_data()

    # Preprocess data
    X_train = preprocess_data(X_train)
    X_validation = preprocess_data(X_validation)
    X_test = preprocess_data(X_test)

    # Create callbacks
    callbacks = create_callbacks(patience=10, hist_freq=1, log_dir=log_dir)

    model = create_model(input_shape=(28, 28, 1))

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_data=(X_validation, y_validation),
        callbacks=callbacks
    )

    evaluate_model(
        model_to_evaluate=model,
        test_data=(X_test, y_test),
        model_history=history.history
    )

    plot_model(model, show_shapes=True)

    res = input("Save model(Y/N): ").strip().upper()
    if res == "Y":
        model.save("model.h5")
    else:
        quit()

