# main class

from dataset import DatasetSplitter, PlantDataset
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# ------------------------------------------------------------------------------------------------------------------------ #
# define dataset + loader
data_splitter = DatasetSplitter("kacpergregorowicz/house-plant-species")  # dataset download from kaggle
train_dir, val_dir, test_dir = data_splitter.get_split_paths()

train_df = data_splitter.create_dataframe(train_dir)
val_df = data_splitter.create_dataframe(val_dir)
test_df = data_splitter.create_dataframe(test_dir)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

train_ds = PlantDataset(train_df, transform= transform)
val_ds = PlantDataset(val_df, transform= transform)
test_ds = PlantDataset(test_df, transform= transform)

BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count()
APPLY_SHUFFLE = True

train_loader = DataLoader(
    dataset = train_ds,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = APPLY_SHUFFLE
)

val_loader = DataLoader(
    dataset = val_ds,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = APPLY_SHUFFLE
)

test_loader = DataLoader(
    dataset = test_ds,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = APPLY_SHUFFLE
)

# ------------------------------------------------------------------------------------------------------------------------ #
# load pre-trained model

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(pretrained=True)

num_ftrs = model.fc.in_features
num_classes = len(train_df["label"].unique()) 
model.fc = nn.Linear(num_ftrs, num_classes)

# move model to device
model = model.to(device)


# define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# ------------------------------------------------------------------------------------------------------------------------ #
# define loss functions, optimizer, training loop & evaluation

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    print("Training Complete!")

# ------------------------------------------------------------------------------------------------------------------------ #
# train & save the model

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)

MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)  # Create folder if it doesn't exist

MODEL_PATH = os.path.join(MODEL_DIR, "plant_classifier.pth")

torch.save(model.state_dict(), MODEL_PATH)
