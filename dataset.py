import os
import shutil
from sklearn.model_selection import train_test_split
import kagglehub
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import torch

class DatasetSplitter:
    def __init__(self, dataset_name, base_dir="split_data", test_size=0.3, val_size=0.2, random_state=42):
        dataset_path = kagglehub.dataset_download(dataset_name)
        data_path = os.path.join(dataset_path, "house_plant_species") 

        print("Path to dataset files:", dataset_path)

        # Define new directories
        base_dir = f"{dataset_path}/split_data"
        self.train_dir = f"{base_dir}/train"
        self.val_dir = f"{base_dir}/val"
        self.test_dir = f"{base_dir}/test"

    def split_and_move_images(self):
        for split in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(split, exist_ok=True)

        class_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]

        # create train, val, and test folders
        for class_name in class_dirs:
            # print(class_name)
            os.makedirs(os.path.join(self.train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_dir, class_name), exist_ok=True)

            # Get all images for this class
            class_path = os.path.join(self.data_path, class_name)
            images = os.listdir(class_path)
            
            # Split into train (70%), val (20%), test (10%)
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1/3, random_state=42)  # 20% val, 10% test

            # Move images
            for img in train_imgs:
                shutil.move(os.path.join(class_path, img), os.path.join(self.train_dir, class_name, img))
            
            for img in val_imgs:
                shutil.move(os.path.join(class_path, img), os.path.join(self.val_dir, class_name, img))

            for img in test_imgs:
                shutil.move(os.path.join(class_path, img), os.path.join(self.test_dir, class_name, img))

        print("Dataset split completed!")
        print("Train Path:", self.train_dir)
        print("Validation Path:", self.val_dir)
        print("Test Path:", self.test_dir)

    def get_split_paths(self):
        """Returns paths for train, validation, and test splits."""
        return self.train_dir, self.val_dir, self.test_dir

    def create_dataframe(self, data_dir):
        data = []
        class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        if not class_folders:
            print(f"No classes found in {data_dir}.")
            return pd.DataFrame(columns=["image_path", "label"])  # Return empty DataFrame

        for class_name in class_folders:
            class_path = os.path.join(data_dir, class_name)
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            for img in images:
                image_path = os.path.join(class_path, img)
                data.append({"image_path": image_path, "label": class_name})  # Append image path and label

        return pd.DataFrame(data)


class PlantDataset(Dataset):
    """Custom Dataset for loading plant images."""
    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        super().__init__()

        self.paths = df['image_path'].to_list()
        self.labels = df['label'].to_list()
        self.transform = transform

        self.classes = sorted(list(df['label'].unique()))
        self.class_to_idx = {cls_name: _ for _, cls_name in enumerate(self.classes)}

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        class_name = self.labels[index]
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(image), class_idx
        else:
            return image, class_idx