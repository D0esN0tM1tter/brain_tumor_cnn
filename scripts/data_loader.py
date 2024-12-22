from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size):

    # Define a pipeline for transforming the images
    print("Creating the transformation pipeline...")
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Normalize([0.5], [0.5])  # Normalize with mean and std
        ]
    )

    # Loading the training and testing data
    print(f"Loading training data from {data_dir}/training...")
    train_data = datasets.ImageFolder(f'{data_dir}/training', transform=transform)
    print(f"Training data loaded. Total images: {len(train_data)}")

    print(f"Loading testing data from {data_dir}/testing...")
    test_data = datasets.ImageFolder(f'{data_dir}/testing', transform=transform)
    print(f"Testing data loaded. Total images: {len(test_data)}")

    # Creating data loaders
    print(f"Creating data loaders with batch size {batch_size}...")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    print(f"Data loaders created. Batch size: {batch_size}")
    print(f"Classes in the dataset: {train_data.classes}")

    # Return data loaders and class names
    return train_loader, test_loader, train_data.classes
