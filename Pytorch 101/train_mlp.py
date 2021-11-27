from Preprocessing import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch

def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # yield aq tuple of cuerrent inputs and batched data
        yield(inputs[i:i+batchSize], targets[i:i+batchSize])

# specify batch size, number of epochs and learning rate
batch_size = 64
epochs = 10
lr = 1e-2

# determine which device will be used for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] training using {device}...')

# generate a 3-class classification problem with 1024 data points where each datapoint is a 4d feature vector
print('[INFO] preparing data...')
(X, y) = make_blobs(n_samples=1024, n_features=4, centers=3, cluster_std=2.5, random_state=95)

#create training and testing splits and convert them to pytorch tensors
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

# initialize modle and display its architecture
mlp = mlp.get_training_model().to(device)
print(mlp)

# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr = lr)
lossFunc = nn.CrossEntropyLoss()

# create a templet to summarize current training process
trainTemplate = 'epoch: {}, test loss: {}, test accuracy: {}'

# loop through the epochs
for epoch in range(0, epochs):
    # initialize tracker variables and set our model to trainable
    print('[INFO] epoch: {}...')
    trainLoss = 0
    trainAcc = 0
    samples = 0
    mlp.train()

    # loop over current batch of data
    for (batchX, batchY) in next_batch(trainX, trainY, batch_size):
        # move data to device, run it through model and calculate loss
        (batchX, batchY) = (batchX.to(device), batchY.to(device))
        predictions = mlp(batchX)
        loss = lossFunc(predictions, batchY.long())

        # zero the gradient accumulated from previous steps, perform batck propagation and update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        # update training loss accuracy and number of samples visited
        trainLoss += loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        samples +=batchY.size(0)

        # display model progress on current training batch
    trainTemplate = 'epoch: {}, train loss: {:.3f},  train accuracy: {:.3f}'
    print(trainTemplate.format(epoch, (trainLoss/samples), (trainAcc/samples)))

    # initialize tracker variables for testing, then set model to evaluation mode
    testLoss = 0
    testAcc = 0
    samples = 0
    mlp.eval()

        # initialize a no-gradient context
    with torch.no_grad():
        # loop over the current batch of test data
        for (batchX, batchY) in next_batch(testX, testY, batch_size):
            # move data to device
            (batchX, batchY) = (batchX.to(device), batchY.to(device))

            # run data through model and calculate loss
            predictions = mlp(batchX)
            loss = lossFunc(predictions, batchY.long())

            #update test loss, accuracy, and the number of samples visited
            testLoss += loss.item() * batchY.size(0)
            testAcc += (predictions.max(1)[1] == batchY).sum().item()
            samples +=batchY.size(0)

        # display model progress on current test batch
        testTemplate = 'epoch: {}, test loss: {:.3f},  test accuracy: {:.3f}'
        print(testTemplate.format(epoch, (testLoss / samples), (testAcc / samples)))
        print('')
