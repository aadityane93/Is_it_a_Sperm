import os
import pandas as pd
import cv2

image_paths = []
labels = []
for classes in os.listdir("data"):
    # print (classes)

    for img_file in os.listdir(f"data/{classes}/"):
        image_paths.append(f"data/{classes}/{img_file}")
        labels.append(classes)

# print("\n ")
# print(labels)

df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.head())

def load(path):
    print(".", end=' ')
    img= cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize(img, img_size=(224,224)):
    print(".", end=' ')
    img = cv2.resize(img, img_size)
    return img


print("loading and resizing images")
df["raw_image"] = df["image_path"].apply(load)
df["resized_image"] = df["raw_image"].apply(resize)


#finding








# print(df["image"][1])

print("\n")
print("Before Resize :", end= " ")
print(df["raw_image"][1].shape)
print("After Resize :", end= " ")
print(df["resized_image"][1].shape)


cv2.imshow('Before Resize', df["raw_image"][2])
cv2.imshow('After Resize', df["resized_image"][2])

cv2.waitKey(0)
#finding the smallest image
