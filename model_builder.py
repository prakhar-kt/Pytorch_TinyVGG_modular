import torch
from torch import nn


class TinyVGG(nn.Module):

    """

    Replicates the TinyVGG architecture from here: https://poloclub.github.io/cnn-explainer/

    Args:
            
            input_shape (int): an integer indicating the number of input channels
            hidden_units (int): an integer indicating the number of hidden units between layers
            output_shape (int): Number of classes in the dataset.


    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, stride=1, padding=0),  # (W-F+2P)/S + 1 = (W-3)+1 = W-2

            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=0), # ((W-2) - 3)/1 + 1 = W-4

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2) # ((W-4) -2)/2 + 1 = W/2 - 2

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=0), # ((W/2 - 2) - 3)+ 1 = W/2 - 4

            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3, stride=1, padding=0), # (W/2 - 4 - 3) + 1 = W/2 - 6
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2) # (W/2 - 6 - 2)/2 + 1 = W/4 - 3

        )

        self.classifier = nn.Sequential(
            
            nn.Flatten(),

            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)

        )

    def forward(self, x: torch.Tensor):

            x = self.conv1(x)

            x = self.conv2(x)

            x = self.classifier(x)

            return x
        

        



