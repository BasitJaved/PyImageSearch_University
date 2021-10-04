from Preprocessing.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

# initialize batch size and number of steps
BS = 64
Num_steps = 5000

# load Cifar 100 dataset
print('[INFO] loading the Cifar 100 dataset...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# create a standard image generator object
print('[INFO] Creating an ImageDataGenerator object...')
imageGen = ImageDataGenerator()
dataGen = imageGen.flow(x=trainX, y=trainY, batch_size=BS, shuffle=True)

# build a Tensorflow dataset from training data
dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

# build the data input pipline
print('[INFO] creating a tf.data input pipline...')
dataset = (dataset.shuffle(1024).cache().repeat().batch(BS).prefetch(tf.data.AUTOTUNE))

# benchmark the image data generator and display the number of data points generated, along with time taken
# to perform operation
totalTime = benchmark(dataGen, Num_steps)
print(f'[INFO] ImageDataGenerator generated {BS*Num_steps} images in {totalTime} seconds...')

# create a dataset iterator, benchmark the tf.data pipline and display the number of data points generated
# along with time taken
datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, Num_steps)
print(f'[INFO] tf.data generated {BS*Num_steps} images in {totalTime} seconds...')
