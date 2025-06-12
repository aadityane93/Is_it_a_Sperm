import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

image_paths = []
labels = []
for classes in os.listdir("resized"):
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


# print(df["image"][1])

print("\n")
print("Before Resize :", end= " ")
print(df["raw_image"][1].shape)
print("After Resize :", end= " ")
print(df["resized_image"][1].shape)


cv2.imshow('Before Resize', df["raw_image"][2])
cv2.imshow('After Resize', df["resized_image"][2])

cv2.waitKey(0)

# converting the resized_image column to a single np array
x= np.array(df['resized_image'].tolist())
y= df['label']

# normalization of pixel values
x = x.astype('float32')/255.0

#encoding the labels
enco = LabelEncoder()
y_encoded = enco.fit_transform(y)
y_categorical = to_categorical(y_encoded)


#split data
x_train, x_test, y_train, y_test = train_test_split(x,y_categorical, test_size=0.2, random_state=1, stratify=y_encoded)

print("\n")
print("x train \n", x_train[0])
print("y train \n", y_train[0])


#base model
base_model = EfficientNetB0(
    include_top = False,
    weights = 'imagenet',
    input_shape = (224,224,3),
    pooling='avg'
)

base_model.trainable = False
#adding new layers

model = models.Sequential([
    base_model,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#image augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow(
    x_train, y_train,
    batch_size=32
)

#training
model.fit(
    train_generator,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        #to reduce learning rate
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]
)



# evaluating
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=enco.classes_))
