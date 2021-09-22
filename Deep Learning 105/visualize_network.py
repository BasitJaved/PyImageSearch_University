from conv.lenet import LeNet
from tensorflow.keras.utils import plot_model

# initialize lenet and write network architecture visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file='LeNet.jpg', show_shapes=True)