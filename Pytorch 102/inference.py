from Preprocessing import config
from Preprocessing import create_data_loader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import torch

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to trained model')
args = vars(ap.parse_args())

# build our data preprocessing piplines
test_transforms = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
])

# calculate inverse mean and standard deviation
inv_mean = [-m/s for (m, s) in zip(config.mean, config.std)]
inv_std = [1/s for s in config.std]

# define out de-normalization transform so we can visualize the image
de_normalize = transforms.Normalize(mean = inv_mean, std = inv_std)

# initialize our test set and test data loader
print('[INFO] loading dataset...')
(test_ds, test_loader) = create_data_loader.get_data_loader(config.val_path, transforms=test_transforms,
                                                            batch_size=config.pred_batch_size, shuffle=True)

# check if GPU is available, if so define map location accordingly
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()

else:
    map_location = 'cpu'

# load model
print('[INFO] loading model...')
model = torch.load(args['model'], map_location=map_location)

# move the model to device and set it in evaluation mode
model.to(config.device)
model.eval()

# grab a batch of test data
batch = next(iter(test_loader))
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure('Results', figsize=(10, 10))

# switch off autograd
with torch.no_grad():
    # send images to device
    images = images.to(config.device)

    # make predictions
    print('[INFO] preforming predictions...')
    preds = model(images)

    # loop over all the batch
    for i in range(0, config.pred_batch_size):
        # initialize a subplot
        ax = plt.subplot(config.pred_batch_size, 1, i+1)

        # grab the image, de_normalize it, scale the raw pixcel intensities to range [0, 255], and change
        # channel ordering from channel first to channel last
        image = images[i]
        image = de_normalize(image).cpu().numpy()
        image = (image*255).astype('uint8')
        image = image.transpose((1,2,0))

        # grab the ground truth label
        idx = labels[i].cpu().numpy()
        get_label = test_ds.classes[idx]

        # grab the predicted label
        pred = preds[i].argmax().cpu().numpy()
        pred_label = test_ds.classes[pred]

        # add results and image to plot
        info = f'Ground Truth: {get_label}, Predicted: {pred_label}'
        plt.imshow(image)
        plt.title(info)
        plt.axis('off')

    # show plot
    plt.tight_layout()
    plt.show()