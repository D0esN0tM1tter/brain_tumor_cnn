import torch
from scripts.cnn_model import CNNModel
from scripts.data_loader import get_data_loaders


model = CNNModel() 
model.load_state_dict(torch.load('models/cnn_model.pth'))
model.eval() 

DATA_DIR = 'dataset' 
BATCH_SIZE = 32 

_ , test_loader , classes = get_data_loaders(DATA_DIR , BATCH_SIZE) 

# evaluating the model on the test dataset : 
correct = 0
total = 0
all_predictions = [] 
all_labels = []

# moving the model to the  CPU : 
device = torch.device("cpu") 
model.to(device)

#  no gradients needed for evaluation : 


with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):  # Add batch index for better tracking
        print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")

        # Move data to the same device as the model
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")

        # Get model outputs
        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")  # Should be (batch_size, num_classes)

        # Get predictions
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted: {predicted}")
        print(f"True labels: {labels}")

        # Update counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Correct predictions in this batch: {(predicted == labels).sum().item()}")

        # Store all predictions and labels for later analysis
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"Evaluation completed. Total samples: {total}, Total correct: {correct}")



accuracy = correct / total * 100
print(f"Accuracy on the test dataset: {accuracy:.2f}%")

