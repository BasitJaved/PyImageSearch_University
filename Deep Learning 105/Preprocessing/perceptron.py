import numpy as np

class Perceptron():
    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix and store learning rate
        self.W = np.random.randn(N + 1)/ np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs = 10):
        # inserts a column of 1s as the last entry in the feature matrix -- this little trick allows us to treat
        # the bias as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X, y):
                # take the dot product between input features and weight matrix, then pass this value through
                # the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))

                # only perform weight update if prediction does not match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * x * error

    def predict(self, X, addBias=True):
        # Ensure our input is a matrix
        X = np.atleast_2d(X)

        # check to see if Bias Column needs to be added
        if addBias:
            # insert a column of 1's as the last entry in features matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between input features and weight matrix then pass value through step function
        return self.step(np.dot(X, self.W))