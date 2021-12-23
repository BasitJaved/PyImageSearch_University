from torch.utils.data import Dataset

class custom_tensor_dataset(dataset):
    # initialize the constructor
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms

        