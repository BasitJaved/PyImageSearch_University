from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

def get_data_loader(root_dir, transforms, batch_size, shuffle=True):
    # create a dataset and use it to create a dataloader
    ds = datasets.ImageFolder(root=root_dir, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(),
                        pin_memory=True if config.device =='cuda' else False)

    return (ds, loader)