#!/usr/bin/env Python3

import numpy as np
import PIL
import os
from import_tensorflow import import_tensorflow
tf = import_tensorflow()
from Prep_Image import testImages

THRESHOLD = 0.30

model = tf.keras.models.load_model('savedModels/bestSoFar', compile = True)

snake_dir = "/home/clark/Documents/notes/AI/SnakeClassifier/data/Snake"
snake_data, snake_names = testImages(snake_dir)
snake_results = model.predict(snake_data)

no_dir = "/home/clark/Documents/notes/AI/SnakeClassifier/data/No-Snake"
no_data, no_names = testImages(no_dir)
no_results = model.predict(no_data)

for delta in range(100, 0, -1):

    snake_list = []
    for i in range(len(snake_results)):
        if snake_results[i][0] >= delta / 100:
            snake_list.append(os.path.join(snake_dir, snake_names[i]))
    
    no_list = []
    for i in range(len(no_results)):
        if no_results[i][0] >= delta / 100:
            no_list.append(os.path.join(no_dir, no_names[i]))
    
    print("Threshold is: " + str(delta) + "%")
    
    print((len(snake_list) / len(snake_results)) * 100, end = "")
    print("% True Positives")
    
    print((len(no_list) / len(no_results)) * 100, end = "")
    print("% False Positives")

    print("\n\n")

print("Done!")

