import torch
import os

# specify path to datasets
dataset_path = 'flowers_photos'
mnist_dataset_path = 'mnist'
base_path = 'dataset'

# define validation split and paths to seperate train and validation splits
val_split = 0.1
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')

# specify ImageNet mean and standard deviation and image size
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_size = 224

# determine device to be  used for training and evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# specify training hyperparameters
feature_extraction_batch_size = 256
fine_tune_batch_size = 64
pred_batch_size = 4
epochs = 20
lr = 0.001
lr_fine_tune = 0.0005

# define paths to store training plots and trained model
warmup_plot = os.path.join('output', 'warmup.png')
fine_tune_plot = os.path.join('output', 'fine_tune.png')
warmup_model = os.path.join('output', 'warmup_model.pth')
fine_tune_model = os.path.join('output', 'fine_tune_model.pth')

# =========================================Below here Lecture 1 configs=====================================

# specify path to our training and validation set (for lecture 1 in this series)
train = 'train'
val = 'val'

# set the input height and width
input_height = 128
input_width = 128

# set the batch size and validation data split
batch_size = 8
val_split = 0.1