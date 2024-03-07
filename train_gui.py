import tkinter as tk
from tkinter import filedialog
from srcnn import build_srcnn  # Make sure the SRCNN model can handle variable input shapes
import numpy as np
import os
import tensorflow as tf
import cv2
import rasterio

def resize_and_pad(img):
    # Determine the next power of 2 for each dimension
    h, w = img.shape
    target_height = next_power_of_2(h)
    target_width = next_power_of_2(w)

    # Resize image to the next power of 2 dimensions (if needed)
    if h != target_height or w != target_width:
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Calculate padding (in case the next power of 2 is already the target size, padding will be 0)
    pad_vert = target_height - h
    pad_top, pad_bottom = pad_vert // 2, pad_vert - (pad_vert // 2)
    pad_horiz = target_width - w
    pad_left, pad_right = pad_horiz // 2, pad_horiz - (pad_horiz // 2)

    # Pad the image to match target dimensions
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return padded_img, (target_width, target_height)

def process_images(lr_img, hr_img):
    lr_h, lr_w = lr_img.shape[:2]
    new_lr_w = next_power_of_2(lr_w)
    new_lr_h = next_power_of_2(lr_h)

    padded_lr_img = resize_and_pad(lr_img, new_lr_w, new_lr_h)
    resized_hr_img = cv2.resize(hr_img, (new_lr_w, new_lr_h))

    return padded_lr_img, resized_hr_img

def select_folder(title="Select Folder"):
    return filedialog.askdirectory(title=title)

def load_image_rasterio(img_path, target_size=None):
    with rasterio.open(img_path) as src:
        img = src.read(1)  # Read the first band

    # Convert to float16 to reduce memory usage
    img = img.astype(np.float16)

    # Resize and pad if a target size is specified
    if target_size:
        img = resize_and_pad(img, target_size[0], target_size[1])

    # Normalize the image
    max_val = img.max()
    if max_val > 0:  # Avoid division by zero
        img /= max_val  # Normalize to [0, 1]

    # Add channel and batch dimensions
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img


def train_model(lr_folder, hr_folder, batch_size=1):
    lr_image_paths = [os.path.join(lr_folder, f) for f in os.listdir(lr_folder) if os.path.isfile(os.path.join(lr_folder, f))]
    hr_image_paths = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder) if os.path.isfile(os.path.join(hr_folder, f))]

    # Ensure that the number of LR and HR images is the same
    assert len(lr_image_paths) == len(hr_image_paths), "The number of low-resolution and high-resolution images must be the same."

    model = build_srcnn()  # Initialize the SRCNN model

    for lr_path, hr_path in zip(lr_image_paths, hr_image_paths):
        lr_img = load_image_rasterio(lr_path)  # Load and process the low-resolution image
        hr_img = load_image_rasterio(hr_path)  # Load and process the high-resolution image

        # Train the model on the current batch of images
        model.train_on_batch(lr_img, hr_img)

    # Save the trained model
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
