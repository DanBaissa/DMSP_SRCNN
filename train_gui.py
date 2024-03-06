# train_gui.py

import tkinter as tk
from tkinter import filedialog
from srcnn import build_srcnn  # Import the SRCNN model definition
import numpy as np
import os
import tensorflow as tf


def select_folder(title="Select Folder"):
    return filedialog.askdirectory(title=title)


def load_image(img_path, color_mode='grayscale'):
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode=color_mode)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize based on the maximum value within the image
    max_val = np.max(img_array)
    if max_val > 0:  # Avoid division by zero
        img_array = img_array / max_val  # Normalize to [0, 1] based on max value

    return img_array



def train_model(lr_folder, hr_folder):
    lr_image_paths = [os.path.join(lr_folder, f) for f in os.listdir(lr_folder) if
                      os.path.isfile(os.path.join(lr_folder, f))]
    hr_image_paths = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder) if
                      os.path.isfile(os.path.join(hr_folder, f))]

    # Placeholder model; actual input shape will be set with the first batch
    model = build_srcnn(input_shape=(None, None, 1))

    for lr_path, hr_path in zip(lr_image_paths, hr_image_paths):
        lr_img = load_image(lr_path)
        hr_img = load_image(hr_path)

        # Train on the pair of images
        model.train_on_batch(lr_img, hr_img)

    # Save the model
    model.save('srcnn_model.h5')
    print("Model saved as srcnn_model.h5")


def on_train_click():
    lr_folder = select_folder("Select Low-Resolution Image Folder")
    hr_folder = select_folder("Select High-Resolution Image Folder")
    train_model(lr_folder, hr_folder)


root = tk.Tk()
root.title("SRCNN Trainer")

train_button = tk.Button(root, text="Train SRCNN", command=on_train_click)
train_button.pack()

root.mainloop()
