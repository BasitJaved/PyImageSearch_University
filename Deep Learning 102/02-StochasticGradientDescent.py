from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # compute the derivative of sigmoid function Assuming that input x has already been passed through
    # sigmoid activation function
    return x * (1 - x)

def predict(X, W):
    # take dot product between feature and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply step function to threshold the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    # return the predictions
    return preds

def next_batch(X,y, batchSize):
    # loop over dataset X in mini-batches, yielding a tuple of current batched and data labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield(X[i:i+batchSize], y[i:i+batchSize])

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--alpha', type=float, default=0.1, help='Learning Rate')
ap.add_argument('-e', '--epochs', type=float, default=100, help='No. of Epochs')
ap.add_argument('-b', '--batchsize', type=int, default=32, help='size of SGD mini-batches')
args = vars(ap.parse_args())

# Generate a 2-Class Classification problem with 1000 data points where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# Bias Trick: insert a column of 1's as the last entry in the feature matrix, this trick allows us
# to treat the bias as a trainable parameter within the feature matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing split using 50% of data for training and remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize the weight matrix and list of losses
print('[INFO] training ...')
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):
    # take the dot product between feature X and weight matrix W then pass this value through our sigmoid activation
    # activation function thereby giving us our predictions on dataset
    epoch_loss = []

    # loop over data in batches
    for X_n, y_n in next_batch(trainX, trainY, args['batchsize']):
        preds = sigmoid_activation(X_n.dot(W))

        # now that we have our predictions we need to determine the error which is difference between predictions
        # and true values
        error = preds - y_n
        epoch_loss.append(np.sum(error**2))


        # the gradient descent update is dot product between
        # 1) features
        # 2) Error of sigmoid derivative of our predictions
        d = error * sigmoid_deriv(preds)
        gradient = X_n.T.dot(d)

        # in the update stage all we need to do is 'nudge' the weight matrix in the negative direction of
        # gradient (hence the term gradient descent) by taking a small step towards a set of more optimal parameters
        W += -args['alpha'] * gradient

    # update our loss history by tacking average loss across all batches
    loss = np.average(epoch_loss)
    losses.append(loss)
    # check to see if update needs to be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f'[INFO] epoch = {int(epoch+1)}, loss = {loss}')

# Evaluate our model
print('[INFO] evaluating ...')
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('loss')
plt.show()
