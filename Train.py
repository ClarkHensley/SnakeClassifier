#!/usr/bin/env python3

from import_tensorflow import import_tensorflow
tf = import_tensorflow()
from Prep_Image import prepareImages

(x_train, y_train), (x_test, y_test) = prepareImages()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    # convolutional neural network

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(144, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
    ])

print(model.summary())

model.compile(optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_test, y_test), shuffle=True)

model.save("savedModels/testModel")

print("Done")

