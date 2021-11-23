# specify the path of the dataset
CSV_PATH = 'abalone_train.csv'

# Specify Column names
COLS = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
        'Shell weight', 'Age']

# Configurations for Easy Hyperparameter Tuning with Keras Tuner lecture are starting from here
# define path to output directory
output_path = 'output'

# initiate input shape and number of classes
input_shape = (28, 28, 1)
num_classes = 10

# define epochs, batch size and early stopping patience
epochs = 50
batch_size = 32
early_stopping_patience = 5