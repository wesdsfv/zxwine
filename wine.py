import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Get the directory where this script is located
dir_path = os.path.dirname(os.path.realpath(_file_))

# Construct the full path to the model file
model_path = os.path.join(dir_path, 'kmeans_wine_clustering.joblib')
scaler_path = "scaler_model.joblib"  # Correct the path for your scaler

# Load the KMeans model
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Ensure the scaler is loaded

# Streamlit title and description
st.title("Wine Cluster Prediction")
st.write("Discover what type of wine you might prefer based on alcohol content and color intensity.")

# User input for alcohol content and color intensity
alcohol_content = st.number_input("Enter the alcohol content:", value=12.0, step=0.1)
color_intensity = st.number_input("Enter the color intensity:", value=5.0, step=0.1)

# Provide warnings if the user inputs values outside typical ranges
if alcohol_content < 11.0 or alcohol_content > 15.0:
    st.warning("Warning: Alcohol content is typically between 11 and 15.")
if color_intensity < 1.0 or color_intensity > 13.0:
    st.warning("Warning: Color intensity is typically between 1 and 13.")

# Create an input array with the user data
input_array = np.array([[alcohol_content, color_intensity]])

# Button to trigger the prediction
if st.button('Check Your Wine Preference'):
    # Scale the input
    input_array_scaled = scaler.transform(input_array)
    
    # Predict the cluster
    predicted_cluster = model.predict(input_array_scaled)[0]
    
    # Map the cluster to a description
    cluster_descriptions = {
        0: "You are a casual drinker. You seem to enjoy light, refreshing wines with bold color and acidity. You likely prefer crisp and lively wines.",
        1: "You are an experienced drinker. You have the refined palate of a connoisseur! You enjoy balanced, flavorful, and structured wines with higher alcohol content.",
        2: "You are health-conscious or an occasional drinker. You likely lean towards wines that are subtle and easy-going."
    }

    # Output the result
    cluster_description = cluster_descriptions.get(predicted_cluster, "Unknown Cluster")
    st.write(f"### {cluster_description}")