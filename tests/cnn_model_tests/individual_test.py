import torch
from torchvision import transforms
from PIL import Image
import random
import os
from scripts.cnn_model import CNNModel
import torch.nn.functional as f

# loading the model : 
model = CNNModel() 
model.load_state_dict(torch.load('models/cnn_model.pth' , weights_only=True))
model.eval() 

# preparing necessary image transformation : 
transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Normalize([0.5], [0.5])  # Normalize with mean and std
        ]
    )

# loading the image : 
data_dir = 'dataset'
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

with torch.no_grad() : 
    output = model(image_tensor)


probabilities = f.softmax(output[0] , dim=0)

class_names = ['glioma' , 'meningioma' , 'notumor' , 'pituitary']
predicted_class = class_names[probabilities.argmax().item()]

print(f'True class : {rand_type}') 
print(f'predicted class : {predicted_class}')