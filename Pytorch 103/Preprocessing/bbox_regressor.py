from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid

class object_detector(Module):
    def __init__(self, base_model, num_classes):
        super(object_detector, self).__init__()

        # initiate the base model and number of classes
        self.base_model = base_model
        self.num_classes = num_classes
