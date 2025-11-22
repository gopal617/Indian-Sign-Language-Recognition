import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
PROCESSED_DATA_PATH = 'processed_data.pkl'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
BATCH_SIZE = 32
EPOCHS = 25
IMG_SIZE = (128, 128)

def create_custom_cnn_model(input_shape, num_classes):
    """Builds a custom CNN model for image classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_transfer_learning_model(base_model, input_shape, num_classes):
    """Builds a model using transfer learning with a pre-trained base."""
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def plot_history(history, model_name):
    """Save training & validation loss/accuracy plots to the results folder."""
    hist = history
    acc = hist.get('accuracy', hist.get('acc'))
    val_acc = hist.get('val_accuracy', hist.get('val_acc'))
    loss = hist.get('loss')
    val_loss = hist.get('val_loss')

    if acc is None or loss is None:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(acc, label='train_acc')
    if val_acc:
        plt.plot(val_acc, label='val_acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_accuracy.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(loss, label='train_loss')
    if val_loss:
        plt.plot(val_loss, label='val_loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_loss.png'))
    plt.close()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, num_classes):
    """Trains, evaluates, and saves a given model."""
    print(f"\n--- Training {model_name} ---")

    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[early_stopping]
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy for {model_name}: {accuracy:.4f}")

    model_path = os.path.join(MODELS_DIR, f'{model_name}.h5')
    model.save(model_path)
    print(f"{model_name} saved to {model_path}")

    # save plots
    plot_history(history.history, model_name)

    return accuracy, history.history

def main():
    """Main function to run the training pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(PROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    label_map = data['label_map']
    num_classes = len(label_map)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    print(f"Successfully loaded data. Number of classes: {num_classes}")

    results = {}

    # Custom CNN
    custom_cnn = create_custom_cnn_model(input_shape, num_classes)
    acc, hist = train_and_evaluate_model(custom_cnn, X_train, y_train, X_test, y_test, 'custom_cnn', num_classes)
    results['Custom CNN'] = {'accuracy': acc, 'history': hist}

    # ResNet50
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    resnet_model = create_transfer_learning_model(resnet_base, input_shape, num_classes)
    acc, hist = train_and_evaluate_model(resnet_model, X_train, y_train, X_test, y_test, 'resnet50', num_classes)
    results['ResNet50'] = {'accuracy': acc, 'history': hist}

    # MobileNetV2
    mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    mobilenet_model = create_transfer_learning_model(mobilenet_base, input_shape, num_classes)
    acc, hist = train_and_evaluate_model(mobilenet_model, X_train, y_train, X_test, y_test, 'mobilenetv2', num_classes)
    results['MobileNetV2'] = {'accuracy': acc, 'history': hist}

    # Save result summary
    with open(os.path.join(RESULTS_DIR, 'summary.pkl'), 'wb') as f:
        import pickle
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
