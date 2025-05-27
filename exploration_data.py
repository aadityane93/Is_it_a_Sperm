# Get an overview of the dataset, including file sizes, dimensions, and label distribution.

import os
import pandas as pd
 
from collections import Counter
 
 
def create_image_dataframe(root_folder):
    data = []

    for label in os.listdir(root_folder):
        class_path = os.path.join(root_folder, label)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(class_path, fname)
                    data.append({'filepath': full_path, 'label': label})

    return pd.DataFrame(data)


import os
import pandas as pd
from PIL import Image







df = create_image_dataframe('resized')

normal_count = len(df[df['label'] == 'Normal_Sperm'])
abnormal_count = len(df[df['label'] == 'Abnormal_Sperm'])
non_sperm_count = len(df[df['label'] == 'Non-Sperm'])
class_counts = Counter(df['label'])

count_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])

print("----General Overview-----")
print("Shape of resized dataset:", df.shape)
print("---------")
print(f"First 5 rows of dataseting:", df.head(5))
print("---------")
print(f"Labels distribution: The dataset is well Balanced:\n ")

count_df['Diff_vs_Normal'] = count_df['Count'] - count_df.loc['Normal_Sperm', 'Count']

print(count_df)
print("---------")
print("Missing values:\n", df.isnull().sum())
print("---------")
file_sizes = df['filepath'].apply(lambda x: os.path.getsize(x) / 1024)  # in KB
print("Average file size: {:.2f} KB".format(file_sizes.mean()))
print("Max file size: {:.2f} KB".format(file_sizes.max()))
print("Min file size: {:.2f} KB".format(file_sizes.min()))
print("---------")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

#  2 for normal, not sperma 1, anormal 0 
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)
 



# Width and height of the datasrt
def get_image_dimensions(filepath):
    with Image.open(filepath) as img:
        return img.size  # returns tuple (width, height)

df[['width', 'height']] = df['filepath'].apply(get_image_dimensions).apply(pd.Series)

# Show summary of dimensions
print("---------")
print("Image dimension summary:")
print(df[['width', 'height']].describe())
print("---------")
print("Most common dimensions:")
print(df[['width', 'height']].value_counts().head(5))


import matplotlib.pyplot as plt

# Function to plot sample images
def plot_sample_images(df, n=5):
    labels = df['label'].unique()
    label_names = {v: k for k, v in label_mapping.items()}  # Reverse the label mapping
    
    plt.figure(figsize=(15, len(labels) * 3))

    for i, label in enumerate(labels):
        sample_df = df[df['label'] == label].sample(n=n, random_state=42)
        
        for j, row in enumerate(sample_df.itertuples()):
            plt_idx = i * n + j + 1
            img = Image.open(row.filepath)
            plt.subplot(len(labels), n, plt_idx)
            plt.imshow(img)
            plt.axis('off')
            plt.title(label_names[label])
    
    plt.tight_layout()
    plt.show()

 
plot_sample_images(df, n=5)