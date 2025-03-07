import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess_data import preprocess_dataset

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    data_dir =r"C:\Users\Lenovo\Desktop\Food_scan\data\images"
    img_size = (128, 128)
    train_data, val_data = preprocess_dataset(data_dir, img_size=img_size)

    model = build_model(input_shape=(128, 128, 3), num_classes=len(train_data.class_indices))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)

    model.save('models/food_scan_model.h5')
