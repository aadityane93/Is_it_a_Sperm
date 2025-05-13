import os
import pandas as pd

image_paths = []
labels = []
for classes in os.listdir("data"):
    print (classes)

    for img_file in os.listdir(f"data/{classes}/"):
        image_paths.append(f"data/{classes}/{img_file}")
        labels.append(classes)

print("\n ")
# print(labels)

df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.head())