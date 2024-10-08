# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.layers as tfl
from sklearn.metrics import classification_report, confusion_matrix
import os

# Define the path to the brain tumor image directory
folder_path_cancer = 'E:\\projectPhase\\Brain Disease\\BrainCancer\\dataset\\cancer'
folder_path_health = 'E:\\projectPhase\\Brain Disease\\BrainCancer\\dataset\\healthy'

# List all files in the cancer directory and store them in a list
image_files_cancer = [file for file in os.listdir(folder_path_cancer)]

# Determine the number of images to display (up to 6 images)
num_images = min(6, len(image_files_cancer))
num_rows = (num_images - 1) // 3 + 1

# Create a grid of subplots to display cancer images
fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))

# Loop through and display each cancer image
for i in range(num_images):
    image_path = os.path.join(folder_path_cancer, image_files_cancer[i])
    img = mpimg.imread(image_path)
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')  # Display images in grayscale
    ax.axis('off')  # Hide the axis

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# List all files in the healthy directory and store them in a list
image_files_healthy = [file for file in os.listdir(folder_path_health)]

# Determine the number of images to display (up to 6 images)
num_images = min(6, len(image_files_healthy))
num_rows = (num_images - 1) // 3 + 1

# Create a grid of subplots to display healthy brain images
fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))

# Loop through and display each healthy brain image
for i in range(num_images):
    image_path = os.path.join(folder_path_health, image_files_healthy[i])
    img = mpimg.imread(image_path)
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')  # Display images in grayscale
    ax.axis('off')  # Hide the axis

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print the number of cancer images
print("Number of Cancer Images in the directory:", len(image_files_cancer))

# Print the number of healthy images
print("Number of Healthy Images in the directory:", len(image_files_healthy))

# Define the image size and batch size
img_height, img_width = 224, 224
batch_size = 32  # Adjust batch size to a more typical value

# Create an ImageDataGenerator object for data preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Reserve 20% of images for validation
)

# Load the training dataset using flow_from_directory method
train_dataset = datagen.flow_from_directory(
    'E:\\projectPhase\\Brain Disease\\BrainCancer\\dataset',
    target_size=(img_height, img_width),  # Resize images to 224x224
    batch_size=batch_size,
    class_mode='binary',  # Use binary labels (cancer vs. healthy)
    color_mode='grayscale',  # Load images in grayscale mode
    interpolation='bilinear',  # Resampling strategy when resizing
    subset='training'  # Specify that this is the training subset
)

# Load the validation dataset using flow_from_directory method
validation_dataset = datagen.flow_from_directory(
    'E:\\projectPhase\\Brain Disease\\BrainCancer\\dataset',
    target_size=(img_height, img_width),  # Resize images to 224x224
    batch_size=batch_size,
    class_mode='binary',  # Use binary labels (cancer vs. healthy)
    color_mode='grayscale',  # Load images in grayscale mode
    interpolation='bilinear',  # Resampling strategy when resizing
    subset='validation'  # Specify that this is the validation subset
)

# Function to plot training metrics (loss, accuracy, precision, recall)
def plot_metrics(history):
    metrics = ["loss", "accuracy", "Precision", "Recall"]
    plt.figure(figsize=(16, 10))
    
    for n, metric in enumerate(metrics):
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train', color='royalblue', marker='o')
        plt.plot(history.epoch, history.history['val_' + metric], linestyle='--', label='Val', color='seagreen', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])
        plt.legend()

# Function to build and train a CNN model for cancer detection
def build_model(metrics, epochs=20):
    tf.keras.backend.clear_session()  # Clear previous session

    # Initialize a Sequential model
    model = tf.keras.models.Sequential()

    # Add Convolutional layers and pooling
    model.add(tfl.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 1)))
    model.add(tfl.MaxPooling2D((2, 2)))
    model.add(tfl.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(tfl.MaxPooling2D((2, 2)))

    # Flatten the output to feed into Dense layers
    model.add(tfl.Flatten())
    model.add(tfl.Dense(8, activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.3))

    # Output layer with sigmoid activation for binary classification
    model.add(tfl.Dense(1, activation='sigmoid'))

    # Compile the model with binary_crossentropy loss and specified metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    model.summary()

    # Train the model on training and validation datasets
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, verbose=1)

    # Plot training metrics
    plot_metrics(history)

    return model, history

# Define metrics for model evaluation
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name='Precision'),
    tf.keras.metrics.Recall(name='Recall'),
]

# Build and train the model
model, history = build_model(metrics, epochs=25)

# Evaluate the model on the validation dataset for confusion matrix and classification report
steps = validation_dataset.samples // validation_dataset.batch_size
if validation_dataset.samples % validation_dataset.batch_size > 0:
    steps += 1

images_list = []
y_true = []

# Collect true labels from validation dataset
for i, (images, labels) in enumerate(validation_dataset):
    if i >= steps:
        break
    images_list.extend(images)
    y_true.extend(labels)

images_array = np.array(images_list)
y_true = np.array(y_true)

# Predict labels using the trained model
y_pred = model.predict(images_array)
y_pred = (y_pred >= 0.5).astype(int)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# Print classification report
print(classification_report(y_true, y_pred))

# Save the trained model
model.save('Brain.keras')
