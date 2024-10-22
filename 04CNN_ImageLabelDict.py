import os
import numpy as np
import pandas as pd
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="split dataset and augment train set")
parser.add_argument('--batch', type=str, required=True, help='Batch identifier to replace "batch1+2" in file and folder names (e.g., "batch1").')
args = parser.parse_args() 
batch = args.batch

label_name = "Firmness"
labelfile = f"label/{batch}label.csv"

# Read the labels once
labels_df = pd.read_csv(labelfile)


save_path = 'dictfile'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# List of dataset subsets
data_subsets = ['train', 'val', 'test']

for subset in data_subsets:
    # Get image names without extensions in the current subset directory
    image_names = set(
        os.path.splitext(filename)[0]
        for filename in os.listdir(f'dataset_split/{batch}_{subset}')
    )

    # Filter labels for images present in the current subset
    labels_subset_df = labels_df[labels_df['Image_name'].isin(image_names)]

    image_label_dict = {}
    for index, row in labels_subset_df.iterrows():
        image_name = row['Image_name']
        firmness = row[label_name]

        # Handle the special case for 'train' subset
        if subset == 'train':
            image_label_dict["original_" + image_name] = firmness
            image_label_dict[image_name] = firmness
        else:
            image_label_dict[image_name] = firmness

    # Adjust filename for 'val' subset to match 'validation'
    subset_save_name = 'validation' if subset == 'val' else subset

    # Save the dictionary as a .npy file
    np.save(f'{save_path}/{batch}_{subset_save_name}.npy', image_label_dict)
