import torch 
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from cnn_model import CNNModel


# defining the data directory and hyper-parameters : 
DATA_DIR = 'dataset' 
BATCH_SIZE = 16
LEARNING_RATE = 0.001 
NUM_EPOCHS = 10
MODEL_PATH = 'cnn_model.pth' 

# define the device on which the calculations will be performed : 
device = torch.device("cpu") 

# loading data : 
train_loader , test_loader , classes = get_data_loaders(DATA_DIR , BATCH_SIZE)

# initialize the model, loss function and optimizer : 
model = CNNModel().to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , LEARNING_RATE) 

# Training loop : 

for epoch in range(NUM_EPOCHS):  # Use range(NUM_EPOCHS) to iterate properly
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Started")
    
    model.train() 
    running_loss = 0 
    correct = 0
    total = 0 

    for batch_idx, (images, labels) in enumerate(train_loader):  # Add batch_idx for logging
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Log batch details
        print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Log epoch summary
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Completed: Loss: {running_loss / len(train_loader):.4f}, "f"Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")