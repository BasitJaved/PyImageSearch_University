from Preprocessing import config
from imutils import paths
import random
import shutil
import os

# grab the path to all input images in original input directory and shuffle them
imagePaths = list(paths.list_images(config.INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# compute training and testing split
i = int(len(imagePaths)*config.Train_split)
trainPaths = imagePaths[:i]
testpaths = imagePaths[i:]

# we will be using part of training data for Validation
i = int(len(trainPaths)*config.Val_split)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define the dataset that we will be building
datasets = [
    ('training', trainPaths, config.Train_path),
    ('validation', valPaths, config.Val_path),
    ('Testing', testpaths, config.Test_path)
]

# loop over the datasets
for(dType, imagePaths, baseOutput) in datasets:
    # Show which data split we are creating
    print(f'[INFO] building {dType} split')

    # if the output directory does not exist create it
    if not os.path.exists(baseOutput):
        print(f'[INFO] Creating {baseOutput} directory')
        os.makedirs(baseOutput)

    #loop over all the image paths
    for inputpath in imagePaths:
        # extract the file name of input image and extract the class label
        filename = inputpath.split(os.path.sep)[-1]
        label = filename[-5:-4]

        # build the path to label directory
        labelpath = os.path.sep.join([baseOutput, label])

        # if label output directory does not exists create it
        if not os.path.exists(labelpath):
            print(f'[INFO] Creating {labelpath} directory')
            os.makedirs(labelpath)

        # construct the path to destination image and copy image
        p = os.path.sep.join([labelpath, filename])
        shutil.copy2(inputpath, p)
