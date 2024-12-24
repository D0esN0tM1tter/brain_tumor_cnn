import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel, self).__init__()

        # defining the architecture of the convolutional block: 
        self.features = nn.Sequential(
            # first convolutional block:
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # second convolutional block:
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # third convolutional block:
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # flatten the final input:
            nn.Flatten()
        )

        # defining the architecture of the fully connected NN:
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),  # Adjusted the input size based on the calculated output size
            nn.ReLU(),
            nn.Linear(512, 4)  # Output layer for 4 classes
        )

    def forward(self, x):
        # forward the image through the feature extractor:
        x_features = self.features(x)
        # feed it to the classifier:
        classifier_result = self.classifier(x_features)

        return classifier_result

