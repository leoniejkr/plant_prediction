import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

# Load the PyTorch model
@st.cache_resource
def load_model(model_path, type):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    num_classes = checkpoint['num_classes']
    transform = checkpoint['transform']
    class_names = checkpoint['class_names']
    
    if type == "resnet":
        model = models.resnet50(pretrained=False)  
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)  
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    if type == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    else: # default 
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
    return model, transform, class_names


# Preprocess the image
def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)  # add batch dimension

# Classify image
def classify_image(image, model, class_names):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

# Streamlit UI
def main():
    st.set_page_config("Image Classifier", layout='wide')
    st.title("Image Classification with PyTorch")
    
    model, transform, class_names = load_model("saved_models/plant_classifier.pth", "efficientnet")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display the uploaded image with a smaller and consistent size
        st.image(image, caption='Uploaded Image', use_container_width=False, width=400)
        
        # Preprocess and classify
        processed_image = preprocess_image(image, transform)
        prediction = classify_image(processed_image, model, class_names)
        st.write(f"### Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
