import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

# Load pre-trained ResNet model and remove the top fully connected layer
base_model = resnet18(pretrained=True)
base_model = torch.nn.Sequential(*(list(base_model.children())[:-1]))

batch = "batch2"

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # the mean and std come from imagenet dataset
])

# Define dataset folder path
trainset = f'dataset_white/{batch}_train'
valset = f'dataset_white/{batch}_val'
testset = f'dataset_white/{batch}_test'

# Create an empty list to store features
train_features = []
val_features = []
test_features = []

# Iterate through all image files in the dataset folder
for filename in sorted(os.listdir(trainset)):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Construct image path
        image_path = os.path.join(trainset, filename)

        # Load image
        img = Image.open(image_path)

        # Preprocess image and add batch dimension
        img_tensor = preprocess(img).unsqueeze(0)

        # Set model to evaluation mode
        base_model.eval()

        # Use ResNet model for feature extraction
        with torch.no_grad():
            features = base_model(img_tensor)

        # Append features to the list
        train_features.append(features.squeeze())


# Iterate through all image files in the dataset folder
for filename in sorted(os.listdir(valset)):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Construct image path
        image_path = os.path.join(valset, filename)

        # Load image
        img = Image.open(image_path)

        # Preprocess image and add batch dimension
        img_tensor = preprocess(img).unsqueeze(0)

        # Set model to evaluation mode
        base_model.eval()

        # Use ResNet model for feature extraction
        with torch.no_grad():
            features = base_model(img_tensor)

        # Append features to the list
        val_features.append(features.squeeze())

# Iterate through all image files in the dataset folder
for filename in sorted(os.listdir(testset)):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Construct image path
        image_path = os.path.join(testset, filename)

        # Load image
        img = Image.open(image_path)

        # Preprocess image and add batch dimension
        img_tensor = preprocess(img).unsqueeze(0)

        # Set model to evaluation mode
        base_model.eval()

        # Use ResNet model for feature extraction
        with torch.no_grad():
            features = base_model(img_tensor)

        # Append features to the list
        test_features.append(features.squeeze())
import os
import pandas as pd

# Load label.csv file
label_df = pd.read_csv(f'label/{batch}label.csv')
labelname = "Firmness"

# Initialize lists to store labels
train_labels = []
val_labels = []
test_labels = []

# Iterate through all image files in the dataset folder
for filename in sorted(os.listdir(trainset)):
    if filename.endswith('.jpg'):
        # Extract filename without extension
        filename_without_extension = os.path.splitext(filename.replace("original_", ""))[0]

        # Extract label from label.csv based on filename
        label = label_df[label_df["Image_name"] == filename_without_extension][labelname].iloc[0]
        train_labels.append(label)

for filename in sorted(os.listdir(valset)):
    if filename.endswith('.jpg'):
        # Extract filename without extension
        filename_without_extension = os.path.splitext(filename)[0]
        
        
        # Extract label from label.csv based on filename
        label = label_df[label_df["Image_name"] == filename_without_extension][labelname].iloc[0]
        val_labels.append(label)

for filename in sorted(os.listdir(testset)):
    if filename.endswith('.jpg'):
        # Extract filename without extension
        filename_without_extension = os.path.splitext(filename)[0]
        
        # Extract label from label.csv based on filename
        label = label_df[label_df["Image_name"] == filename_without_extension][labelname].iloc[0]
        test_labels.append(label)

train_labels = [torch.tensor([float(train_labels[i])]) for i in range(len(train_labels))]
val_label = [torch.tensor([float(val_labels[i])]) for i in range(len(val_labels))]
test_label = [torch.tensor([float(test_labels[i])]) for i in range(len(test_labels))]

train_features_tensor = torch.stack(train_features)
train_labels_tensor = torch.tensor(train_labels)
val_features_tensor = torch.stack(val_features)
val_labels_tensor = torch.tensor(val_labels)
test_features_tensor = torch.stack(test_features)
test_labels_tensor = torch.tensor(test_labels)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import numpy as np

# Define the random forest model
# 定义随机森林模型
model = RandomForestRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
# 定义参数空间
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Create a GridSearchCV object with parallel computation
# 创建 GridSearchCV 对象，并设置并行计算
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
# 拟合网格搜索模型
grid_search.fit(train_features_tensor, train_labels_tensor)

# Output the best parameter combination
# 输出最佳参数组合
print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
best_model.fit(train_features_tensor, train_labels_tensor)

# Make predictions on the training and validation sets
# 在训练集和验证集上进行预测
train_predictions = best_model.predict(train_features_tensor)
val_predictions = best_model.predict(val_features_tensor)

# Calculate mean squared error and R^2 score
# 计算均方误差和 R^2 分数
train_loss = mean_squared_error(train_labels_tensor, train_predictions)
val_loss = mean_squared_error(val_labels_tensor, val_predictions)
train_r2 = r2_score(train_labels_tensor, train_predictions)
val_r2 = r2_score(val_labels_tensor, val_predictions)

# Output performance metrics on the training and validation sets
# 输出训练和验证集上的性能指标
print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
print(f'Train R-squared: {train_r2:.4f}, Validation R-squared: {val_r2:.4f}')

# Make predictions on the test set
# 在测试集上进行预测
test_predictions = best_model.predict(test_features_tensor)
mse = mean_squared_error(test_labels_tensor, test_predictions)
r_squared = r2_score(test_labels_tensor, test_predictions)

# Output predictions and performance metrics on the test set
# 输出测试集上的预测结果和性能指标
print("Predictions:", test_predictions)
print("True Labels:", test_labels_tensor.tolist())
print(f'Test RMSE: {np.sqrt(mse):.4f}')
print(f'Test R-squared: {r_squared:.4f}')

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the SVR model
# 定义 SVR 模型
svr = SVR()

# Define the parameter search space
# 定义参数搜索范围
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],  # Kernel types
    'C': [0.01, 0.1, 1, 10, 100],        # Regularization parameter
    'epsilon': [0.01, 0.1, 1, 10]        # Epsilon in the epsilon-SVR model
}

# Create the GridSearchCV object with parallel computation
# 创建 GridSearchCV 对象，并设置并行计算
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
# 拟合网格搜索模型
grid_search.fit(train_features_tensor, train_labels_tensor)

# Output the best parameter combination
# 输出最佳参数组合
print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
best_model.fit(train_features_tensor, train_labels_tensor)

# Make predictions on the training and validation sets
# 在训练集和验证集上进行预测
train_predictions = best_model.predict(train_features_tensor)
val_predictions = best_model.predict(val_features_tensor)

# Calculate mean squared error and R^2 score
# 计算均方误差和 R^2 分数
train_loss = mean_squared_error(train_labels_tensor, train_predictions)
val_loss = mean_squared_error(val_labels_tensor, val_predictions)
train_r2 = r2_score(train_labels_tensor, train_predictions)
val_r2 = r2_score(val_labels_tensor, val_predictions)

# Output performance metrics on the training and validation sets
# 输出训练和验证集上的性能指标
print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
print(f'Train R-squared: {train_r2:.4f}, Validation R-squared: {val_r2:.4f}')

# Make predictions on the test set
# 在测试集上进行预测
test_predictions = best_model.predict(test_features_tensor.numpy())
test_labels = test_labels_tensor.numpy()

# Calculate mean squared error and R^2 score on the test set
# 计算测试集上的均方误差和 R^2 分数
mse = mean_squared_error(test_labels, test_predictions)
r_squared = r2_score(test_labels, test_predictions)

# Output predictions and performance metrics on the test set
# 输出测试集上的预测结果和性能指标
print("Predictions:", test_predictions)
print("True Labels:", test_labels)
print(f'Test RMSE: {np.sqrt(mse):.4f}')
print(f'Test R-squared: {r_squared:.4f}')