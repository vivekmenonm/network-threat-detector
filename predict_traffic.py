import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import tensorflow as tf
from keras.models import load_model

# Load the dataset
data = pd.read_excel('android_traffic.xlsx')  # Replace 'your_dataset.csv' with the actual filename

# Separate the features and the target variable
X = data.drop(['type', 'name'], axis=1)
y = data['type']

# Encode the categorical target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to input features and return predicted class labels
def predict_traffic_class(features):
    # Load the saved model
    loaded_model = load_model('android_traffic_model.h5')

    # Preprocess the input features
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = loaded_model.predict(features_scaled)
    predicted_classes = np.round(predictions).astype(int)

    # Inverse transform the predicted classes to get the original labels
    predicted_labels = le.inverse_transform(predicted_classes)

    return predicted_labels

# Example usage of the predict_traffic_class function
# input_features = np.array([[100, 0.5, 3, 5000, 20, 1, 200, 150, 8000, 6000, 10, 10]])
# predicted_class = predict_traffic_class(input_features)
# print(f'Predicted class: {predicted_class[0]}')