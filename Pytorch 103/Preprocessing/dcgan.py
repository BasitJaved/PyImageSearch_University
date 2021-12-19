from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch.nn import Flatten
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=512, output_channels=1):
        super(Generator, self).__init__()

        # first set of Conv => Relu => BN
        self.ct1 = ConvTranspose2d(in_channels=input_dim, out_channels=128, kernel_size=4, stride=2,
                                   padding=0, bias=False)
        self.relu1 = ReLU()
        self.batch_norm1 = BatchNorm2d(128)

        # Second set of Conv => Relu => BN
        self.ct2 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                                   padding=1, bias=False)
        self.relu2 = ReLU()
        self.batch_norm2 = BatchNorm2d(64)

        # last set of Conv => Relu => BN
        self.ct3 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2,
                                   padding=1, bias=False)
        self.relu3 = ReLU()
        self.batch_norm3 = BatchNorm2d(32)

        # apply another upsample and transposed convolution but this time output the TanH activation
        self.ct4 = ConvTranspose2d(in_channels=32, out_channels=output_channels, kernel_size=4, stride=2,
                                   padding=1, bias=False)
        self.tanh = Tanh()

    def forward(self, x):
        # Pass the input through our first layer of CONVT => RELU => BN layers
        x = self.ct1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)

        # Pass the input through our Second layer of CONVT => RELU => BN layers
        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)

        # Pass the input through our last layer of CONVT => RELU => BN layers
        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)

        # finally pass the input through last CONVT => TANH layer to get the output
        x = self.ct4(x)
        output = self.tanh(x)

        return output

class Descriminator(nn.Module):
    def __init__(self, depth, alpha=0.2):
        super(Descriminator, self).__init__()

        # first set of CONV => RELU layers
        self.conv1 = Conv2d(in_channels=depth, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = LeakyReLU(alpha, inplace=True)

        # Second set of CONV => RELU layers
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu2 = LeakyReLU(alpha, inplace=True)

        # First set of FC => RELU layers
        self.fc1 = Linear(in_features=3136, out_features=512)
        self.leaky_relu3 = LeakyReLU(alpha, inplace=True)

        # Sigmoid layer outputting a single value
        self.fc2 = Linear(in_features=512, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):

        # passing input through first set of CONV => RELU layers
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        # passing input through second set of CONV => RELU layers
        x = self.conv2(x)
        x = self.leaky_relu2(x)

        # flatten the output and passing input through first set of FC => RELU layers
        x = Flatten(x, 1)
        x = self.fc1(x)
        x = self.leaky_relu3(x)

        # passing input through last set of FC => Sigmoid layer to get output
        x = self.fc2(x)
        output = self.sigmoid(x)

        return output