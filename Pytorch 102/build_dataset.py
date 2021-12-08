from Preprocessing import config
from imutils import paths
import numpy as np
import shutil
import os

def copy_images(image_paths, folder):
    # check to see if destination folder exists or not if not then create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # loop over the image paths
    for path in image_paths:
        # grab the image name and its label from path and create a placeholder corresponding to separate label folder
        image_name = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[-2]
        label_folder = os.path.join(folder, label)

        # check to see if label folder exists if not then create it
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        # construct the destination image path and copy current image to it
        destination = os.path.join(label_folder, image_name)
        shutil.copy(path, destination)

# load all the image paths and randomly shuffle them
print('[INFO] loading image paths...')
image_paths = list(paths.list_images(config.flowers_dataset_path))
np.random.shuffle(image_paths)

# generate training and validation paths
val_path_len = int(len(image_paths) * config.val_split)
train_path_len = len(image_paths) - val_path_len
train_paths = image_paths[:train_path_len]
val_paths = image_paths[train_path_len:]

# copy training and validation images to their respective directories
print('[INFO] copying training and validation images...')
copy_images(train_paths, config.train)
copy_images(val_paths, config.val)