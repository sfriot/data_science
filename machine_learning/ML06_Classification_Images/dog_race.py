# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:13:05 2020

@author: Admin
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.image import resize
from tensorflow.keras import Input, Model, models
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.applications import EfficientNetB4
# from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image


def plot_image(predicted_race, predicted_percentage, img):
    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.title("Race prédite : {} ({:.1%})".format(predicted_race,
              predicted_percentage))
    plt.show()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--file',
                        required=True,
                        help='Path to the image to classify')
    opt = parser.parse_args()
    preprocess = joblib.load("model/dogs_preprocess.joblib")
    my_model = models.load_model("model/dogs_model")
    img = Image.open(opt.file)
    img_to_predict = np.array(img)
    dog_race = np.load("dog_race.npy")
    good_size_img = resize(tf.expand_dims(img_to_predict, axis=0),
                           size=[256, 256])
    ready_img = preprocess(good_size_img)
    prediction = my_model.predict(ready_img)
    predicted_race = dog_race[np.argmax(prediction[0])]
    predicted_percentage = np.max(prediction[0])
    print("Race prédite = {} ({:.1%})".format(predicted_race,
                                              predicted_percentage))
    plot_image(predicted_race, predicted_percentage, img_to_predict)


if __name__ == '__main__':
    main()
