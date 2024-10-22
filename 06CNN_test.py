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
epochtype = "Last" #Last or Best



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

# Create testing dataset and data loader
test_image_label_map = np.load(f'dictfile/{batch}_test.npy', allow_pickle=True).item()
test_dataset = CustomDataset(root_dir=f'dataset/{batch}_test', image_label_map=test_image_label_map, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=32)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create ResNet model and move it to GPU
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 1)
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(resnet.parameters(),  lr=0.0001, weight_decay=1e-5)


model_weights = torch.load(f'Weights/{batch}_{epochtype}_{num_epochs}.pth')
resnet.load_state_dict(model_weights)  # load weights


resnet.eval()
test_losses = []
test_predictions = []
test_targets = []

with torch.no_grad():
    for inputs, targets in test_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到GPU上
        outputs = resnet(inputs)
        test_loss = criterion(outputs.squeeze(), targets.float()).item()
        test_losses.append(test_loss)
        test_predictions.extend(outputs.squeeze().cpu().tolist())  # 将预测值移回CPU
        test_targets.extend(targets.cpu().tolist())  # 将目标值移回CPU
print("predict",test_predictions)
print("true",test_targets)


test_mse = mean_squared_error(test_targets, test_predictions)

test_r2 = r2_score(test_targets, test_predictions)

print(f"Test RMSE: {np.sqrt(test_mse)}")
print(f"Test R^2: {test_r2}")

