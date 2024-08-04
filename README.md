# Uber Estimated Time of Arrival (ETA) Prediction

This project leverages **Streamlit** for an interactive web interface and **TensorFlow** for machine learning to predict the **Estimated Time of Arrival (ETA)** for Uber rides. The model and application work together to provide real-time predictions and insightful data visualizations.

## Overview

### 🚀 **Streamlit** Application

**Streamlit** is used to build the web application that allows users to interact with the ETA prediction model. The app includes features like:
- **Input fields** to enter details of a new trip.
- **Real-time ETA prediction** using a pre-trained TensorFlow model.
- **Data visualization** for insightful analysis of Uber ride data.

### 🤖 **TensorFlow** Model

The **TensorFlow** model is used to predict the ETA based on various input features such as ride category, start location, stop location, miles, purpose, date, time, and shift. The model is trained on historical Uber data and employs a neural network architecture to make accurate predictions.

#### Key Features of the TensorFlow Model:
- **Neural Network Architecture**: Uses a sequential model with dense layers and dropout for regularization.
- **Learning Rate Scheduler**: Implements a learning rate scheduler to optimize training performance.
- **Model Saving**: The trained model is saved in HDF5 format for easy loading and inference.

### ⏱️ **Estimated Time of Arrival (ETA)**

**ETA** refers to the predicted time it will take to complete a ride based on historical data and input features. The model outputs ETA in minutes, providing users with an estimate of how long their trip might take.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Utkarshmishra2k2/Uber-Estimated-Time-of-Arrival-ETA-Prediction
    cd Uber-Estimated-Time-of-Arrival-ETA-Prediction
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Usage

- Open the Streamlit app in your browser.
- Enter trip details in the input fields.
- Click on the "Predict" button to get the ETA for the new trip.
- Explore various data visualizations and statistical insights provided in the application.

## Code Structure

- **`app.py`**: The main Streamlit application file.
- **`model_file.py`**: Contains TensorFlow model definition, training, and saving.
- **`backend.py`**: Includes data loading, preprocessing, and prediction functions.

## Acknowledgements

- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io)
- **TensorFlow**: [TensorFlow Documentation](https://www.tensorflow.org)
