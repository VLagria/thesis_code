import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Path to your dataset directory

data = pd.read_csv('D:/Documents/Thesis/Code/DataSet/english.csv')

# Split data into features (paths) and labels
x_data = data['image'].values
y_data = data['label'].values

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Create an ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define image dimensions
image_height = 128 
image_width = 128  

# Define batch size
batch_size = 32  # You can change this value

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'path_column': x_train, 'label_column': y_train}),
    x_col='path_column',
    y_col='label_column',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'path_column': x_val, 'label_column': y_val}),
    x_col='path_column',
    y_col='label_column',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(62, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Plot Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Make Predictions (replace 'test_image_path' with the actual path to your test image)
test_image_path = 'rawData/3.png'  # Replace with the actual path to your test image
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(image_height, image_width))
test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255.0
test_image = np.expand_dims(test_image, axis=0)
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions)

# Display the test image and prediction
class_labels = list(train_generator.class_indices.keys())  # Get class labels

plt.figure(figsize=(6, 6))
plt.imshow(test_image[0])
plt.title(f"Predicted Class: {class_labels[predicted_class]}")
plt.axis('off')
plt.show()
