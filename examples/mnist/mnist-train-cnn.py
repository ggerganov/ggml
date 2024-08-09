#!/usr/bin/env python3
import sys
from time import time
import gguf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train(model_path):
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, dtype=tf.float32),
            layers.Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu", dtype=tf.float32),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", dtype=tf.float32),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax", dtype=tf.float32),
        ]
    )

    model.summary()
    batch_size = 500
    epochs = 20
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    t_start = time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    print(f"Training took {time()-t_start:.2f}s")

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.6f}")
    print(f"Test accuracy: {100*score[1]:.2f}%")

    gguf_writer = gguf.GGUFWriter(model_path, "mnist-cnn")

    kernel1 = model.layers[0].weights[0].numpy()
    kernel1 = np.moveaxis(kernel1, [2,3], [0,1])
    gguf_writer.add_tensor("kernel1", kernel1, raw_shape=(8, 1, 3, 3))

    bias1 = model.layers[0].weights[1].numpy()
    bias1 = np.repeat(bias1, 28*28)
    gguf_writer.add_tensor("bias1", bias1, raw_shape=(1, 8, 28, 28))

    kernel2 = model.layers[2].weights[0].numpy()
    kernel2 = np.moveaxis(kernel2, [0,1,2,3], [2,3,1,0])
    gguf_writer.add_tensor("kernel2", kernel2, raw_shape=(16, 8, 3, 3))

    bias2 = model.layers[2].weights[1].numpy()
    bias2 = np.repeat(bias2, 14*14)
    gguf_writer.add_tensor("bias2", bias2, raw_shape=(1, 16, 14, 14))

    dense_w = model.layers[-1].weights[0].numpy()
    dense_w = dense_w.transpose()
    gguf_writer.add_tensor("dense_w", dense_w, raw_shape=(10, 7*7*16))

    dense_b = model.layers[-1].weights[1].numpy()
    gguf_writer.add_tensor("dense_b", dense_b)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"GGUF model saved to '{model_path}'")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)
    train(sys.argv[1])
