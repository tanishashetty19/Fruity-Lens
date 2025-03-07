# Fruit Recognition App

## Overview
The **Fruit Recognition App** is a deep learning-based image classification project designed to identify different types of fruits from images. This application is built using **TensorFlow/Keras** for model training and **Streamlit** for an interactive user interface.

## Features
- Upload an image of a fruit and receive a predicted class.
- Displays the confidence score for the prediction.
- Uses a **Convolutional Neural Network (CNN)** trained on a fruit dataset.
- Allows users to correct misclassified predictions.

## Technologies Used
### Programming Language
- **Python**

### Libraries & Frameworks
- **TensorFlow/Keras**: Model building and training
- **Streamlit**: Web application framework
- **Pillow (PIL)**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib** (optional): Data visualization

### Model Details
- **Architecture:** CNN (Convolutional Neural Network)
- **Input Shape:** (128, 128, 3)
- **Layers Used:** Convolutional, MaxPooling, Flatten, Dense
- **Activation Function:** ReLU (hidden layers), Softmax (output layer)
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam

## Installation & Setup
### Prerequisites
- Python 3.8 or above installed 
- Required libraries installed (use the command below)

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/fruit-recognition.git
   cd fruit-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in the browser.
2. Upload an image of a fruit.
3. View the predicted fruit name and confidence score.
4. (Optional) Correct the prediction if incorrect.

## Dataset
- Used **Indian Fruits Dataset** (or a similar fruit dataset).
- Images are resized to 128x128 pixels before training.

## Folder Structure
```
Fruit_Recognition/
│-- data/                     # Dataset directory
│   ├── Train/
│   ├── Test/
│-- models/                   # Trained model directory
│   ├── food_scan_model.h5
│-- static/                   # Background images for Streamlit
│-- app.py                     # Streamlit application
│-- preprocess_data.py         # Data preprocessing script
│-- train_model.py             # Model training script
│-- requirements.txt           # Dependencies
│-- README.md                  # Project documentation
```

## Future Enhancements
- Expand dataset to include food and its nutrients.
- Improve accuracy by fine-tuning the model with transfer learning.
- Add a feature to display nutritional information of detected food items and Deploy the model to a cloud platform.






