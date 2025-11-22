import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Configuration
DATASET_PATH = 'dataset'
PROCESSED_DATA_PATH = 'processed_data.pkl'
IMAGE_SIZE = (128, 128)

def load_images_from_folder(folder):
    """Loads images from a folder, resizes them, and assigns labels."""
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))

    print(f"Found classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {class_name}")
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(class_name)

    return np.array(images), np.array(labels)

def preprocess_data():
    """Main function to load, preprocess, and save the dataset."""
    print("Starting data preprocessing...")

    X, y = load_images_from_folder(DATASET_PATH)

    if len(X) == 0:
        raise RuntimeError(f'No images found in {DATASET_PATH}. Please download the dataset and place it in this folder (each class in its own subdirectory).')

    unique_labels = sorted(list(set(y)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_encoded = np.array([label_map[label] for label in y])

    X = X.astype('float32') / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    data_to_save = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'label_map': label_map
    }

    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Data preprocessing complete. Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    preprocess_data()
