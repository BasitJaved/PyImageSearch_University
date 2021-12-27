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

        # build a regressor head for outputting bounding box coordinates
        self.regressor = Sequential(
            Linear(base_model.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )

        # build a classifier head to predict class labels
        self.classifier = Sequential(
            Linear(base_model.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.num_classes)
        )

        # set the classifier from our base model to produce outputs from last convoluttion block
        self.base_model.fc = Identity()

    def forward(self, x):

        # pass the input through the base model then obtain the predictions from two different branches of network
        features = self.base_model(x)
        bboxes = self.regressor(features)
        class_logits = self.classifier(features)

        return (bboxes, class_logits)
