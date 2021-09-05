import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha = 0.1):

        #initialize the list of weight metrices then store the network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from index of first layer but stop before we reach last two layers
        for i in np.arange(0, len(layers) - 2):
            # Randomly initialize the weight matrix connecting the number of nodes in each respective layer
            # together adding an extra layer for bias
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))

        # the last two layers are a special case where the input connections need a bias term but output does not
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w/w/np.sqrt(layers[-2]))

    def __repr__(self):
        # Construct and return a string that represnet the network architecture
        return "Neural Network: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1.0/(1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # Compute the derivative of sigmoid activation function
        return  x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate = 100):
        # insert a column of 1's as the last entry in the feature matrix -- this little trick allows us to
        # treat the bias as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train network
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display an update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print(f'[INFO] epoch = {epoch + 1}, loss = {loss}')

    def fit_partial(self, x, y):
        # construct list of output activations for each layer as our data point flows through the network, the
        # first activation is a special case -- it's just  the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDForward
        # loop over thet layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward activation at current layer by taking the dot product between the activation and the weight
            #  matrix -- this is called the net input to the current layer
            net = A[layer].dot(self.W[layer])

            # applying the activation function
            out = self.sigmoid(net)

            # once we have output add it to the list of our activations
            A.append(out)

        #BACKPROPAGATION
        # the first phase of backpropagarion is to compute the difference between prediction and true target value
        error = A[-1] - y

        # from here we need to apply chain rule and build our list of deltas 'D'; first entry in the deltas is
        # simply error of output layer times the derivative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # now looping over the layers in reverse order
        for layer in np.arange(len(A)-2, 0, -1):
            # the Delta for currnet layer is equal to the delta of previous layer dotted with weight matrix of
            # current layer, followed by multiplying the delta by the derivative of non-linear activation function
            # for the activations of current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # Since we looped over layers in reverse order we need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE Phase
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update the weights by taking the dot product of layer  activations with their respective deltas,
            # then multiplying this value by some small learning rate and adding it to our weight matrix
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # initialize the output prediction as input feature this value will be forward propagated through the
        # network to obtain final prediction
        p = np.atleast_2d(X)

        # check to see if bias column should be added
        if addBias:
            # insert a column of 1's as last entry in the feature matrix(bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over the layers in network
        for layer in np.arange(0, len(self.W)):
            # compute the output prediction
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        # make predictions and calculate loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets)**2) # mean squared error

        return loss
