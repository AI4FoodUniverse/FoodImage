import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

label_name = 'Firmness'
batch = '4cbatch1'
num_epochs = 300


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, image_label_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_label_map = image_label_map
        self.image_names = list(self.image_label_map.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx] + ".jpg")  # Add .jpg extension
        
        image = Image.open(img_name).convert('RGB')
        label = self.image_label_map[self.image_names[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training dataset and data loader
train_image_label_map = np.load(f'dictfile/{batch}_train.npy', allow_pickle=True).item()
train_dataset = CustomDataset(root_dir=f'dataset/{batch}_train', image_label_map=train_image_label_map, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create validation dataset and data loader
val_image_label_map = np.load(f'dictfile/{batch}_validation.npy', allow_pickle=True).item()
val_dataset = CustomDataset(root_dir=f'dataset/{batch}_val', image_label_map=val_image_label_map, transform=transform)
val_data_loader = DataLoader(val_dataset, batch_size=32)

# Create testing dataset and data loader
test_image_label_map = np.load(f'dictfile/{batch}_test.npy', allow_pickle=True).item()
test_dataset = CustomDataset(root_dir=f'dataset/{batch}_test', image_label_map=test_image_label_map, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=32)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create ResNet model and move it to GPU
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 1)
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(resnet.parameters(),  lr=0.0001, weight_decay=1e-5)

import matplotlib.pyplot as plt

best_val_loss = float('inf')  # Initialize best validation loss to infinity
best_model_weights = None  # Initialize best model weights to None

train_losses = []
val_losses = []


# Train the model
for epoch in range(num_epochs):
    # Train the model
    resnet.train()
    for inputs, targets in train_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on validation set
    resnet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            outputs = resnet(inputs)
            val_loss += criterion(outputs.squeeze(), targets.float()).item()
    val_loss /= len(val_data_loader)

    train_losses.append(loss.item())
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}, Val Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = resnet.state_dict()
        torch.save(best_model_weights, f'Weights/{batch}_Best_{num_epochs}.pth')

torch.save(resnet.state_dict(), f'Weights/{batch}_Last_{num_epochs}.pth')




# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'LossFunction/{batch}_{num_epochs}.png')
plt.show()



