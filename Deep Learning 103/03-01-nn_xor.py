from Preprocessing.neuralnetwork import NeuralNetwork
import numpy as np

# Construct the AND dataset
X = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our 2-2-1 network and train it
p = NeuralNetwork([2, 2, 1], alpha=0.5)
p.fit(X, y, epochs=20000)

# now that our perceptron in trained we can evaluate it
print('[INFO] Testing Perceptron...')

# Loop over the data points
for (x, target) in zip(X, y):
    # make predictions and show result
    pred = p.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f'[INFO] data= {x}, Ground-Truth= {target[0]}, Pred= {pred}, Step= {step}')