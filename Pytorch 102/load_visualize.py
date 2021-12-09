from Preprocessing import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

def visualize_batch(batch, classes, dataset_type):
    # initialize a figure
    fig = plt.figure(f'{dataset_type} batch', figsize=(config.batch_size, config.batch_size))

    # loop over the batch size
    for i in range(0, config.batch_size):
        # create a sub-plot
        ax = plt.subplot(2, 4, i+1)

        # grab image and convert it from channel first to channel last ordering, then scale the raw pixel
        # intensities to [0, 255]
        image = batch[0][i].cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = (image * 255).astype('uint8')

        #grab the label id and get label from classes list
        idx = batch[1][i]
        label = classes[idx]

        # show the image along with label
        plt.imshow(image)
        plt.title(label)
        plt.axis('off')

# initialize data augmentation function
resize = transforms.Resize(size=(config.input_height, config.input_width))
hflip = transforms.RandomHorizontalFlip(p=0.25)
vflip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)

# initialize out training and validation set data augmentation pipline
train_transforms = transforms.Compose([resize, hflip, vflip, rotate, transforms.ToTensor()])
val_transforms = transforms.Compose([resize, transforms.ToTensor()])

# initialize training and validation dataset
print('[INFO] loading training and validation dataset...')
train_dataset = ImageFolder(root=config.train, transform=train_transforms)
val_dataset = ImageFolder(root=config.val, transform=val_transforms)
print(f'[INFO] Training dataset contains {len(train_dataset)} samples...')
print(f'[INFO] Val dataset contains {len(val_dataset)} samples...')

# create training and validation set dataloaders
print('[INFO] creating training and validation set dataloaders...')
train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size)

# grab a batch from both training and validation dataloader
train_batch = next(iter(train_data_loader))
val_batch = next(iter(val_data_loader))

# visualize training and validation set batches
print('[INFO] visualizing training and validation batch...')
visualize_batch(train_batch, train_dataset.classes, 'train')
visualize_batch(val_batch, val_dataset.classes, 'val')