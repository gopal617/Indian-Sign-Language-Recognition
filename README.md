# Indian Sign Language (ISL) Recognition System

## Project Overview
This project aims to develop a robust deep learning model for recognizing **Indian Sign Language (ISL)** gestures from images. We use a custom Convolutional Neural Network (CNN) and compare its performance with pre-trained models like **ResNet50** and **MobileNetV2** using transfer learning. The goal is to identify the most accurate and efficient model for a potential real-time application.

## 🚀 Key Features
- **Custom CNN Model:** A custom-built deep learning model for ISL gesture recognition.
- **Transfer Learning:** Using pre-trained models (ResNet50, MobileNetV2) to enhance accuracy and reduce training time.
- **Comparative Analysis:** Detailed performance evaluation and comparison of all models on a test set.
- **Data Augmentation:** Techniques used to expand the dataset and improve model generalization.
- **Modular Codebase:** Clean, well-documented Python scripts for data processing, training, and evaluation.
- **Visualization:** Graphs to visualize model training progress and final accuracy comparison.

## ⚙️ Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/ISL_Recognition_Project.git
    cd ISL_Recognition_Project
    ```

2.  **Create a virtual environment and activate it (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux / macOS
    venv\Scripts\activate    # Windows (PowerShell)
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Dataset
The dataset for this project can be sourced from various public platforms. For this project, we recommend the **Indian Sign Language Alphabet** dataset from sites like Kaggle.

**Important:** Do NOT commit your dataset to GitHub if it is large. Add the `dataset/` folder to your `.gitignore` (already done here) and host the dataset externally (Kaggle, Google Drive, etc.). Place the dataset locally like this:

```
dataset/
├── A/ (images of sign 'A')
├── B/ (images of sign 'B')
└── ...
```

## 📝 Usage
To run the project, follow these steps in order:

1.  **Data Preprocessing:**
    ```bash
    python scripts/preprocess_data.py
    ```

2.  **Model Training:**
    ```bash
    python scripts/train.py
    ```

3.  **Model Evaluation:**
    ```bash
    python scripts/evaluate.py
    ```

## 📊 Results
After training, trained models will be saved to the `models/` folder and performance plots will be saved to the `results/` folder. Replace the sample accuracy table below with your actual results.

### Accuracy Comparison (example)
| Model | Test Accuracy |
| :--- | :--- |
| Custom CNN | 0.90 |
| ResNet50 (Transfer) | 0.95 |
| MobileNetV2 (Transfer) | 0.96 |

*(Note: These are example values. Your results will vary.)*

## 🗺️ Future Work
- Develop a real-time application to translate ISL gestures from a webcam feed.
- Expand the dataset to include more ISL signs and dynamic gestures.
- Deploy the final model as a web or mobile application.

## 📄 License
This project is licensed under the **MIT License**. See `LICENSE` for details.
