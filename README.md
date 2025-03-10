# Plant Species Classification Project
## Overview
The Plant Species Classification project leverages machine learning techniques to classify plant species from images of their leaves. The primary objective is to build a robust classifier that can accurately identify plant species based on their visual features. The project uses the Kaggle dataset kacpergregorowicz/house-plant-species to train the model, and users can interact with the classifier through a Streamlit-based web application.

### Features
Image Classification: Classifies plant species based on leaf images.
Pretrained Models: Fine-tune models such as ResNet50 or EfficientNetB0 for improved accuracy.
Web Interface: A user-friendly interface built with Streamlit for easy image uploads and predictions.

### Requirements
Before you run the project, ensure you have requirements intsalled with:

```pip install -r requirements.txt```


### Dataset
The model is trained using the Kaggle dataset kacpergregorowicz/house-plant-species. You can access this dataset from Kaggle to explore further or train locally.

# Training
to train the model, you can use one of the following scripts:

```python train_method.py```

or

```python train_simple.py```,

which will fine-tune a model using the dataset and save the trained model for later use


Once training is complete, the model weights and other relevant information are saved in a checkpoint file located in the saved_models/ directory.

# Start the Web Application


To interact with the trained model through a web interface, you can use the provided Streamlit app. This app allows you to upload leaf images and receive predictions about the plant species.

To start the app, run:

```streamlit run main.py```

This will launch a web interface where you can upload an image and get the predicted plant species