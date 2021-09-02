from Preprocessing.perceptron import Perceptron
import numpy as np

# Construct the AND dataset
X = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define our perceptron and train it
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=50)

# now that our perceptron in trained we can evaluate it
print('[INFO] Testing Perceptron...')

# Loop over the data points
for (x, target) in zip(X, y):
    # make predictions and show result
    pred = p.predict(x)
    print(f'[INFO] data= {x}, Ground-Truth= {target[0]}, Pred= {pred}')