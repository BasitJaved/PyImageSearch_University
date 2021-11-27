from collections import OrderedDict
import torch.nn as nn

#4-8-3 model
def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
    # construct a shallow, sequential network
    mlpModel = nn.Sequential(OrderedDict([
        ('hidden_layer_1', nn.Linear(inFeatures, hiddenDim)),
        ('activation_layer_1', nn.ReLU()),
        ('output_layer', nn.Linear(hiddenDim, nbClasses))
    ]))

    return mlpModel