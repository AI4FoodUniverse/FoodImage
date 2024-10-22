import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="split dataset and augment train set")
parser.add_argument('--batch',
        type=str,
        required=True,
        help='Batch identifier to replace "batch1+2" in file and folder names (e.g., "batch1").'
    )
args = parser.parse_args() 
batch_num = args.batch

csv_file = f'label/{batch_num}label.csv'
data = pd.read_csv(csv_file)

image_folder = f'dataset_white/{batch_num}_white'

train_folder = f'dataset_split/{batch_num}_train'
val_folder = f'dataset_split/{batch_num}_val'
test_folder = f'dataset_split/{batch_num}_test'
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

train_data, test_data = train_test_split(data, test_size=0.4, shuffle=True, stratify=data['Firmness'], random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.75, shuffle=True, stratify=test_data['Firmness'], random_state=42)

def augment_image(image):
    # Image translation
    tx = np.random.randint(-20, 20)  # Random translation distance
    ty = np.random.randint(-20, 20)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix
    translated_image = cv2.warpAffine(image, M_translate, (image.shape[1], image.shape[0]))

    # Image rotation
    angle = np.random.randint(-30, 30)  # Random rotation angle
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Rotation center
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotation matrix
    rotated_image = cv2.warpAffine(translated_image, M_rotate, (image.shape[1], image.shape[0]))
    
    return rotated_image

def copy_images(data, dest_folder):
    for index, row in data.iterrows():
        image_name = row['Image_name'] + ".jpg"  
        source_path = os.path.join(image_folder, image_name)
        dest_path = os.path.join(dest_folder, image_name)
        copyfile(source_path, dest_path)
        
def copy_images_with_augmentation(data, dest_folder):
    for index, row in data.iterrows():
        image_name = row['Image_name'] + ".jpg"  
        source_path = os.path.join(image_folder, image_name)
        dest_path = os.path.join(dest_folder, image_name)
        
        image = cv2.imread(source_path)  # Read image
        augmented_image = augment_image(image)  # Data augmentation
        
        cv2.imwrite(dest_path, augmented_image)  # Save augmented image
        
        # Also copy the original image to the training folder
        original_dest_path = os.path.join(dest_folder, "original_" + image_name)
        copyfile(source_path, original_dest_path)  # Copy original image

copy_images_with_augmentation(train_data, train_folder)
copy_images(val_data, val_folder)
copy_images(test_data, test_folder)
