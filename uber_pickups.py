import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Define your neural network architecture (SimpleNN) here if not already defined
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
@st.cache_data()
def load_model(model_file):
    model = SimpleNN()
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()
    else:
        st.error(f"Error: Model file '{model_file}' not found.")
        st.stop()
    return model

# Prediction function
def predict_digit(image_path, model):
    try:
        image = Image.open(image_path).convert('L')  # Open image in grayscale
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        image = np.array(image)  # Convert to numpy array
        image = torch.tensor(image, dtype=torch.float32)  # Convert to PyTorch tensor
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        image = transforms.Normalize((0.5,), (0.5,))(image)  # Normalize pixel values

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit app
st.markdown('<a href="https://sohel-datta.vercel.app/" style="text-decoration: none; color: inherit;">'
            '<h1 style="text-align: center;">Digit Recognition by Sohel Datta</h1>'
            '</a>',
            unsafe_allow_html=True)

# Instructions and Information Section
st.markdown("""
Upload an image of a digit (0-9) to predict which digit it represents using a model trained by Sohel Datta.
""")
st.markdown("---")

# Load the model
model_file = 'mnist_model.pth'
model = load_model(model_file)

# Define the LinkedIn profile picture link as an icon
icon_url = "https://media.licdn.com/dms/image/D4E03AQFHPk7bC9ZC5Q/profile-displayphoto-shrink_200_200/0/1692881348340?e=2147483647&v=beta&t=h7VOH_d_jnS5ICwAARY9iAylcrHWKMHEgaNfEij7q8g"

# Add the icon in the top left corner
st.markdown(f"""
    <style>
        .icon {{
            display: block;
            position: fixed;
            top: 60px;
            left: 20px;
            width: 50px;
            height: 50px;
            background-image: url('{icon_url}');
            background-size: cover;
            background-repeat: no-repeat;
            border-radius: 50%;
        }}
    </style>
    <div class="icon"></div>
""", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    # Display uploaded image
    try:
        image = Image.open(uploaded_file)
        resized_image = image.resize((100, 100))
        st.image(resized_image, caption='Uploaded Image', use_column_width=False)

        # Make prediction and display result
        if st.button('Predict'):
            prediction = predict_digit(uploaded_file, model)
            if prediction is not None:
                st.success(f'Prediction: {prediction}')
            else:
                st.warning('Unable to make a prediction. Please try again.')

    except OSError as e:
        st.error(f"Error opening uploaded file: {e}")
