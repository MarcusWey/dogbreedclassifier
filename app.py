import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import gdown  # Import gdown to download from Google Drive
import os

# Define the same image transformation used during training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to download the model from Google Drive
def download_model_from_drive():
    model_file = 'vgg16_best_model_1.pth'
    
    # Check if the model file already exists to avoid downloading again
    if not os.path.exists(model_file):
        # Google Drive file ID (from the shareable link)
        file_id = '1--Tbq0QWkXY2yk_5dOIrG35s2L1sOUrz'
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, model_file, quiet=False)

# Load the trained model (VGG16 in this case)
def load_model():
    # Download the model file from Google Drive if not already present
    download_model_from_drive()
    
    model = models.vgg16(pretrained=False)
    
    # Modify the fully connected layer to match your number of classes
    num_classes = 14  # Update with your actual number of classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    # Load saved weights
    model.load_state_dict(torch.load("vgg16_best_model_1.pth", map_location=torch.device('cpu')))
    
    model.eval()  # Set model to evaluation mode
    return model

# Function to make predictions
def predict(image, model):
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Streamlit interface
st.title("Dog Breed Classifier")
st.write("Upload an image and the model will classify the dog breed.")

# Upload image section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load the model
    model = load_model()

    # Predict the class
    st.write("Classifying...")
    predicted_class = predict(image, model)
    
    # Map class index to actual breed name (update this list with your classes)
    class_names = ['Affenhuahua', 'Afgan Hound', 'Akita', 'Alaskan Malamute', 'American Bulldog', 
                   'Auggie', 'Beagle', 'Belgian Tervuren', 'Bichon Frise', 'Bocker', 
                   'Borzoi', 'Boxer', 'Bugg', 'Bulldog']
    
    st.write(f"Predicted Dog Breed: {class_names[predicted_class]}")
