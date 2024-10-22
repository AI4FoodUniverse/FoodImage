import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

batch = "4cbatch1"
label_name = "Firmness"
labelfile = f"label/{batch}label.csv"
labels_df = pd.read_csv(labelfile)
# keep file name only in train

train_image_names = set(os.path.splitext(filename)[0] for filename in os.listdir(f'dataset/{batch}_train'))
labels_df = labels_df[labels_df['Image_name'].isin(train_image_names)]
image_label_dict = {}
for index, row in labels_df.iterrows():
    image_name = row['Image_name']
    firmness = row[label_name]
    image_label_dict["original_"+image_name] = firmness
    image_label_dict[image_name] = firmness
np.save(f'dictfile/{batch}_train.npy', image_label_dict)

labels_df = pd.read_csv(labelfile)
val_image_names = set(os.path.splitext(filename)[0] for filename in os.listdir(f'dataset/{batch}_val'))
labels_df = labels_df[labels_df['Image_name'].isin(val_image_names)]
image_label_dict = {}
for index, row in labels_df.iterrows():
    image_name = row['Image_name']
    firmness = row[label_name]
    image_label_dict[image_name] = firmness
np.save(f'dictfile/{batch}_validation.npy', image_label_dict)


labels_df = pd.read_csv(labelfile)
test_image_names = set(os.path.splitext(filename)[0] for filename in os.listdir(f'dataset/{batch}_test'))
labels_df = labels_df[labels_df['Image_name'].isin(test_image_names)]
image_label_dict = {}
for index, row in labels_df.iterrows():
    image_name = row['Image_name']
    firmness = row[label_name]
    image_label_dict[image_name] = firmness
np.save(f'dictfile/{batch}_test.npy', image_label_dict)