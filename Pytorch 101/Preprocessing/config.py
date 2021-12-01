import torch

# specify the image dimension
image_size = 224

# specify the imagenet mean and standard deviation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# determine the device that will be used for infering
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# specify path to imagenet labels
in_labels = 'ilsvrc2012_wordnet_lemmas.txt'