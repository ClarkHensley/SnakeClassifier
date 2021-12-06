#!/usr/bin/env Python3

import numpy as np
import PIL
import os
from import_tensorflow import import_tensorflow
tf = import_tensorflow()
from Prep_Image import testImages

model = tf.keras.models.load_model('savedModels/newModel', compile = True)

#root_dir = input("Please copy and paste the absolute path to the directory you want this script to run on: ")
root_dir = "/home/clark/Documents/notes/AI/SnakeClassifier/data/Snake"
prediction_data, prediction_names = testImages(root_dir)

results = model.predict(prediction_data)

final_string = ""
for i in range(len(results)):
    if results[i][1] > results[i][0]:
        final_string += os.path.join(root_dir, prediction_names[i])
        final_string += " "

with open("results.txt", "w") as h:
    h.write(final_string)

