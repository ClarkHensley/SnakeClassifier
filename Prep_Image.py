#!/usr/bin/env python3

from PIL import Image
import numpy as np
import os
import random
import copy

new_size = (256, 144)

def prepareImages(root_dir = None):

    training_images = []
    testing_images = []
    training_labels = []
    testing_labels = []
    if root_dir is None:
        root_dir = os.path.realpath(".")
    for root, dirs, files_t in os.walk(root_dir):
        for d in dirs:
            if d == "data":
                t_dir = os.path.join(root_dir, d)
                for subroot, subdirs, files_t in os.walk(t_dir):
                    for d_2 in subdirs:
                        l_dir = os.path.join(t_dir, d_2)
                        for subsubroot, dirs_2, files in os.walk(l_dir):
                            for f in files:
                                training_string = os.path.join(l_dir, f)
                                with Image.open(training_string) as h:
                                    h_2 = h.resize(new_size)
                                    training_images.append(np.array(h_2))
                                if d_2 == "Snake":
                                    training_labels.append(0)
                                else:
                                    training_labels.append(1)

    temp_indices = [x for x in range(len(training_images))]
    removal_indices = []
    while len(removal_indices) < ((5 * len(training_images)) / 6):
        removal_indices.append(temp_indices.pop(random.randint(0, len(temp_indices) - 1)))

    removal_indices.sort(reverse=True)
    for i in removal_indices:
       testing_images.append(copy.deepcopy(training_images[i]))
       testing_labels.append(copy.copy(training_labels[i]))
       del training_images[i]
       del training_labels[i]
    
    training_images = np.asarray(training_images)
    training_labels = np.asarray(training_labels)
    testing_images = np.asarray(testing_images)
    testing_labels = np.asarray(testing_labels)

    return (training_images, training_labels), (testing_images, testing_labels)

def testImages(root_dir, pictures=[], names=[]):
    for root, dirs, files in os.walk(root_dir):
        
        for f in files:
            if len(pictures) > 10:
                break
            temp = os.path.join(root_dir, f)
            with Image.open(temp) as h:
                h_2 = h.resize(new_size)
                h_2 = np.asarray(h_2)
                h_2 = h_2 / 255.0
                pictures.append(h_2)
                print(f)
                names.append(f)
        
    return np.asarray(pictures), names

