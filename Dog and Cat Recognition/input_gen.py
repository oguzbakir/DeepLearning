import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "./PetImages"
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 70  # Every image is normalized to IMG_SIZExIMG_SIZE

img_array = []
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):  # For every image in directory
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert images to grayscale array
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resized_array, class_num])
            except Exception as e:
                pass


create_training_data()  # OK, we have data but its half dog half cat sequentially.
#  We have to shuffle it.
random.shuffle(training_data)

X = []  # Feature set
y = []  # Labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Dump arrays with pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

"""To load them use
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
"""

