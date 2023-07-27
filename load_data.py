import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load mnist dataset and split it into training, testing and validation datasets...
    :return:
        Split data (tuple): (X_train, y_train, X_validation, y_validation, X_test, y_test)
    """
    # Load data from mnist
    mnist_dataset = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist_dataset.load_data()

    # Split test data into test and validation datasets
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5, random_state=29
    )

    return X_train, y_train, X_validation, y_validation, X_test, y_test
