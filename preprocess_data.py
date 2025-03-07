import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def preprocess_dataset(data_dir, img_size=(128, 128)):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data

if __name__ == "__main__":
    data_dir =r"C:\Users\Lenovo\Desktop\Food_scan\data\images"
    train_data, val_data = preprocess_dataset(data_dir)
