from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def _load(image_path, as_tensor=True):
    """
    Load an image from the given path.

    Args:
        image_path (str): Path to the image file.
        as_tensor (bool): Whether to convert the image to a tensor.

    Returns:
        Image or Tensor: Loaded image.
    """
    image = Image.open(image_path)
    if as_tensor:
        converter = transforms.ToTensor()
        return converter(image)
    else:
        return image

def view_sample(image, label, color_map='rgb', fig_size=(8,10)):
    """
    Display a single sample image with its label.

    Args:
        image (Tensor or PIL Image): Image to display.
        label (str): Label of the image.
        color_map (str): Color map for displaying the image.
        fig_size (tuple): Figure size for the plot.
    """
    plt.figure(figsize=fig_size)

    if isinstance(image, Image.Image):  # Check if the image is a PIL image
        image = transforms.ToTensor()(image)  # Convert to tensor for consistency

    if color_map == 'rgb':
        plt.imshow(image.permute(1, 2, 0))  # Permute to HWC format for RGB
    else:
        plt.imshow(image.squeeze(), cmap=color_map)

    plt.title(f'Label: {label}', fontsize=16)
    plt.axis('off')  # H
