import torch
import random
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn

# Define the CNN model (with the feature extractor)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)


def visualize_feature_maps(data_dir):
    # Load the CNN model
    model = CNNModel()
    model.eval()  # Set model to evaluation mode

    # Define the image transformations (resize, grayscale, etc.)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Assuming your input data is normalized
    ])

    # Get a random tumor type from the training directory
    tumor_types = os.listdir(f'{data_dir}/training')
    rand_type = random.choice(tumor_types)

    # Get a random image from the chosen tumor type
    tumor_type_dir = f'{data_dir}/training/{rand_type}'
    image_file = random.choice(os.listdir(tumor_type_dir))
    image_path = os.path.join(tumor_type_dir, image_file)

    # Load the image using PIL
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Pass the image through the feature extractor
    with torch.no_grad():
        feature_maps = model.features[:7](image_tensor)  # Extract features before flattening

    # Select the first 10 feature maps (channels)
    selected_feature_maps = feature_maps[0, :10, :, :]  # Select first 10 feature maps

    # Plot the selected feature maps
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))  # 2 rows, 5 columns for 10 maps
    axes = axes.flatten()  # Flatten axes array to easily iterate over

    for i, ax in enumerate(axes):
        ax.imshow(selected_feature_maps[i].cpu().detach().numpy(), cmap='viridis')  # Convert tensor to numpy and plot
        ax.axis('off')
        ax.set_title(f'Feature Map {i+1}')
    
    plt.tight_layout()
    plt.show()


# Test the visualization function
if __name__ == '__main__':
    data_dir = 'dataset'  # Specify your dataset path
    visualize_feature_maps(data_dir)
