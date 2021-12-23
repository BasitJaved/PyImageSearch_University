import torch
import os

# define the base path to input dataset and use it to derive path to input image and annotation CSV files
base_path = 'dataset'
images_path = os.path.sep.join([base_path, 'images'])
annots_path = os.path.sep.join([base_path, 'annotations'])

# define the output path to base output directory
base_output = 'output'

# define the path to output model, label encoder, plots output directory and testing image paths
model_path = os.path.sep.join([base_output, 'detector.pth'])
le_path = os.path.sep.join([base_output, 'le.pickle'])
plots_path = os.path.sep.join([base_output, 'plots'])
test_path = os.path.sep.join([base_output, 'test_path.txt'])

# determine the current device and based on that set the pin memory flag
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pin_memory = True if device=='cuda' else False

# specify imagenet mean and standard deviation
mean = [0.485, 0.456, 0.406]
sd = [0.229, 0.224, 0.225]

# initialize our initial learning rate, number of epochs to train for and batch size
init_lr = 1e-4
num_epochs = 20
batch_size = 32

# specify the loss weights
labels = 1.0
bbox = 1.0