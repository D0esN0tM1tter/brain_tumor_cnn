import os
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def visualize_transformed_image(data_dir):

    # Define the transformation pipeline
    transform = transforms.Compose(
        [
            # Resize the image
            transforms.Resize((300, 300)),
            # Convert to grayscale
            transforms.Grayscale(num_output_channels=1),
            # Convert into PyTorch tensor
            transforms.ToTensor(),
            # Normalize the tensor
            transforms.Normalize([0.5], [0.5])
        ]
    )

    # Picking a random tumor type
    tumor_types = os.listdir(f'{data_dir}/training')
    rand_type = random.choice(tumor_types)

    # Picking a random image from the chosen tumor type
    tumor_type_dir = f'{data_dir}/training/{rand_type}'
    image_file = random.choice(os.listdir(tumor_type_dir))
    image_path = os.path.join(tumor_type_dir, image_file)

    # Load the image using PIL
    image = Image.open(image_path)

    # Apply the transformations
    transformed_image = transform(image)

    transformed_image = transformed_image.squeeze(0)  # Remove the batch dimension (1, 128, 128) -> (128, 128)

    # Plot the transformed image
    plt.imshow(transformed_image, cmap='inferno')  # Display the image with gray colormap for single channel
    plt.title(f"Transformed Image - {rand_type}")
    plt.axis('off')  # Hide the axis for better visualization
    plt.show()

if __name__ == '__main__':
    data_dir = 'dataset'  # Replace with your dataset path
    visualize_transformed_image(data_dir=data_dir)
