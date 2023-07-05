import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data loading and preprocessing
dataset = keras.utils.image_dataset_from_directory('./flower_photos', image_size=(180, 180), batch_size=32)
class_names = dataset.class_names
for data, labels in dataset:
    print(class_names)
    print(data.shape)
    break


#image classification
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './flower_photos',
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(180, 180),
    batch_size=32)

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
])

#viualize the data wit matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        plt.show()