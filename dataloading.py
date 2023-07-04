import numpy as np
import tensorflow as tf
from tensorflow import keras

#data loading and preprocessing
dataset = keras.utils.image_dataset_from_directory('./flower_photos', image_size=(180, 180), batch_size=32)
class_names = dataset.class_names
for data, labels in dataset:
    print(class_names)
    print(data.shape)
    break
