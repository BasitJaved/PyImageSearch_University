import os

# initialize path to original input directory of images
INPUT_DATASET = os.path.join('datasets', 'orig')

# initiate the path to new directory that will contain our images after computing training and testing split
Base_path = os.path.join('datasets', 'idc')

# derive the training, validation and testing directories
Train_path = os.path.sep.join([Base_path, 'training'])
Val_path = os.path.sep.join([Base_path, 'validation'])
Test_path = os.path.sep.join([Base_path, 'testing'])

# define the amount of data that will be used during Training
Train_split = 0.8

# the amount of validation data that will be percentage of training data
Val_split = 0.1

# define the input image dimensions
Image_size = (48, 48)

# initialize out no. of epochs, early stopping patience, initial learning rate  and batch size
Num_Epochs = 40
Early_stopping_patience = 5
Init_lr = 1e-2
Bs = 128


