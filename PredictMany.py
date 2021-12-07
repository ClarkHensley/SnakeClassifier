#!/usr/bin/env Python3

from PIL import Image
import numpy as np
import os
from import_tensorflow import import_tensorflow
tf = import_tensorflow()

THRESHOLD = 0.50

model = tf.keras.models.load_model('savedModels/bestSoFar', compile = True)

file_list = []
root_dir = "/mnt/2021 Photos/Snake Shed 4.10.21 done/North Back 4.10.21/100STLTH"
for root, dirs, files in os.walk(root_dir):
    for f in files:
        file_list.append(os.path.join(root_dir, f))

def getInt(string):
    return(int(string.split("/")[-1][4:8]))

file_list.sort(key=getInt)

pictures = []
names = []
for i in range(len(file_list)):
    if len(pictures) >= 1000 or getInt(file_list[i]) < 4000:
        continue
    with Image.open(file_list[i]) as h:
        h_2 = h.resize((256, 144))
        h_2 = np.asarray(h_2)
        h_2 = h_2 / 255.0
        pictures.append(h_2)
        names.append(file_list[i].split("/")[-1])
        print(names[-1])

results = model.predict(np.asarray(pictures))

for i in range(len(results)):

    if results[i][0] >= THRESHOLD:
        print("SNAKE", end = ": ")
        print(names[i])

