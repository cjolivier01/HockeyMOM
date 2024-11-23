import torch
import torchvision.transforms as transforms
from PIL import Image


def is_jersey_dark(image_tensor, bbox, threshold=0.5):
    """
    Determine if the jersey is dark or light.
    Args:
    image_tensor (torch.Tensor): Tensor of the image.
    bbox (tuple): Bounding box (x1, y1, x2, y2).
    threshold (float): Threshold to classify dark or light. Range [0, 1].

    Returns:
    bool: True if dark, False if light.
    """
    # Crop the image
    x1, y1, x2, y2 = bbox
    cropped_image = image_tensor[:, y1:y2, x1:x2]

    # Convert to PIL for easier manipulation
    to_pil = transforms.ToPILImage()
    cropped_image_pil = to_pil(cropped_image)

    # Convert to LAB color space where L stands for lightness
    cropped_image_lab = cropped_image_pil.convert("LAB")

    # Convert back to tensor
    to_tensor = transforms.ToTensor()
    lab_tensor = to_tensor(cropped_image_lab)

    # Calculate the average lightness
    lightness = lab_tensor[0, :, :].mean().item()

    # Determine if dark or light based on the threshold
    return lightness < threshold


if __name__ == "__main__":
    # Example usage
    # Load your image
    image_path = "path_to_your_image.jpg"
    image_pil = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image_pil)

    # Define your bounding box coordinates (x1, y1, x2, y2)
    bbox = (50, 100, 200, 400)

    # Set a threshold, e.g., 0.5 for LAB lightness (scale 0 to 1)
    threshold = 0.5

    # Check if the jersey is dark
    dark_jersey = is_jersey_dark(image_tensor, bbox, threshold)
    print("The jersey is dark:", dark_jersey)
