import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Configuration
PROCESSED_DATA_PATH = 'processed_data.pkl'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

def load_data():
    """Loads preprocessed data from the pickle file."""
    print("Loading preprocessed data...")
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    X_test = data['X_test']
    y_test = data['y_test']

    return X_test, y_test

def load_all_models():
    """Loads all trained models from the models directory."""
    models = {}
    print("Loading trained models...")
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.h5'):
            model_name = os.path.splitext(filename)[0]
            model_path = os.path.join(MODELS_DIR, filename)
            models[model_name] = load_model(model_path)
            print(f"Loaded model: {model_name}")
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluates each model and returns a dictionary of accuracies."""
    accuracies = {}
    print("\n--- Evaluating models ---")
    for name, model in models.items():
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies[name] = accuracy
        print(f"Accuracy for {name}: {accuracy:.4f}")
    return accuracies

def plot_accuracy_comparison(accuracies):
    """Generates and saves a bar chart comparing the accuracy of all models."""
    models = list(accuracies.keys())
    values = list(accuracies.values())

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.bar(models, values)
    plt.title('Model Accuracy Comparison on Test Data')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

    plot_path = os.path.join(RESULTS_DIR, 'final_accuracy_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nFinal comparison chart saved to {plot_path}")

def main():
    """Main function to run the evaluation pipeline."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_test, y_test = load_data()
    models = load_all_models()

    accuracies = evaluate_models(models, X_test, y_test)
    plot_accuracy_comparison(accuracies)

if __name__ == '__main__':
    main()
