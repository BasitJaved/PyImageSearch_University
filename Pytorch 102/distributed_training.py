from Preprocessing import config
from Preprocessing import create_data_loader
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# determine number of GPUs we have
num_gpu = torch.cuda.device_count()
print(f'[INFO] Number of GPUs found are: {num_gpu}')

# determine batch size based on number of GPUs
batch_size = config.batch_size * num_gpu
print(f'[INFO] using a batch size of {batch_size} ....')

# define augmentation piplines
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
])

val_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
])

# create data loader
(train_ds, train_loader) = create_data_loader.get_data_loader(config.train_path,
                                                              transforms=train_transform,
                                                              batch_size=config.feature_extraction_batch_size)
(val_ds, val_loader) = create_data_loader.get_data_loader(config.val_path,
                                                          transforms=val_transform,
                                                          batch_size=config.feature_extraction_batch_size,
                                                          shuffle=False)

# load up resnet50 model
model = resnet50(pretrained=True)
num_features = model.fc.in_features

# loop over the modules of model and set parameters of batch normalization modules as not trainable
for module, param in zip(model.modules(), model.parameters()):
    if isinstance(module, nn.BatchNorm2d):
        param.requires_grad = False

# define network head and attach it to model
head_model = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(train_ds.classes))
)
model.fc = head_model

# append a new classification top to our feature extractor and pop it on to the current device
model = model.to(config.device)

# if we have more then 1 GPU then parallelize the model
if num_gpu > 1:
    model = nn.DataParallel(model)

# initialize loss function and classifier
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.fc.parameters(), lr=config.lr * num_gpu)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# initialize a learning rate scheduler to decay it by a factor of 0.1 after every 10 epochs
lr_schedul = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

# calculate steps per epoch for training and validation set
train_step = len(train_ds)//config.feature_extraction_batch_size
val_step = len(val_ds)//config.feature_extraction_batch_size

# initialize a dictionary to store training history
H = {'training_loss': [],
     'train_acc': [],
     'val_loss': [],
     'val_acc': []}

# loop over epochs
print('[INFO] training the network ...')
start_time = time.time()
for e in tqdm(range(config.epochs)):

    # set model to training mode
    model.train()

    # initialize total training and validation loss
    total_train_loss = 0
    total_val_loss = 0

    # initialize number of correct predictions in training and validation set
    train_correct = 0
    val_correct = 0

    # loop over the training set
    for (x, y) in train_loader:
        with torch.cuda.amp.autocast(enabled=True):
            # send inout to device
            (x, y) = (x.to(config.device), y.to(config.device))

            # perform forward pass and calculate training loss
            pred = model(x)
            loss = loss_func(pred, y)

        # calculate the gradient
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

        # add the loss to total training loss so far and calculate total number of correct predictions
        total_train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # update LR scheduler
    lr_schedul.step()

    # switch off auto grad
    with torch.no_grad():
        # set model in evaluation mode
        model.eval()

        # loop over the validation set
        for (x, y) in val_loader:
            with torch.cuda.amp.autocast(enabled=True):
                # send input to device
                (x, y) = (x.to(config.device), y.to(config.device))

                # make predictions and calculate validation loss
                pred = model(x)
                total_val_loss += loss_func(pred, y)

            # calculate number of correct predictions
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # calculate average training and validation loss
    avg_train_loss = total_train_loss/train_step
    avg_val_loss = total_val_loss/val_step

    # calculate training and validation accuracy
    train_correct = train_correct / len(train_ds)
    val_correct = val_correct / len(val_ds)

    # update training history
    H['training_loss'].append(avg_train_loss)
    H['train_acc'].append(train_correct)
    H['val_loss'].append(avg_val_loss)
    H['val_acc'].append(val_correct)

    # print model training and validation information
    print(f'[INFO] Epoch: {e+1}/{config.epochs}')
    print(f'[INFO] Train Loss: {avg_train_loss:.6f}, Train Accuracy: {train_correct:.4f}')
    print(f'[INFO] Val Loss: {avg_val_loss:.6f}, Val Accuracy: {val_correct:.4f}')

# Display total time needed to perform training
end_time = time.time()
print(f'[INFO] Total time taken to train Model: {end_time - start_time: .2f} s')

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H['training_loss'], label='training_loss')
plt.plot(H['val_loss'], label='val_loss')
plt.plot(H['train_acc'], label='train_acc')
plt.plot(H['val_acc'], label='val_acc')
plt.title('Training Loss and accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(config.warmup_plot)

# serialize model to disk
torch.save(model.module.state_dict(), config.warmup_model)
