import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import numpy as np


def ensure_pil(img):
    """
    Ensures the input is a PIL.Image.
    - If it's already a PIL image, returns it as-is.
    - If it's a torch.Tensor, converts it using to_pil_image.
    - If it's a np.ndarray, converts it using Image.fromarray.
    - If it's a list with exactly one element, recursively converts that element.
    - Otherwise, raises an error.
    """
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, torch.Tensor):
        return to_pil_image(img)
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img.astype('uint8'))
    elif isinstance(img, list):
        if len(img) == 1:
            return ensure_pil(img[0])
        else:
            raise ValueError(f"Cannot convert list of length {len(img)} to PIL.")
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")


def robust_transform(example, transform):
    """
    Applies the given transform to the "image" field in the example.
    Handles both individual examples and batched examples (i.e. when
    example["image"] is a list).
    """
    # If the image field is a list, process each image in the list.
    if isinstance(example["image"], list):
        new_images = []
        for img in example["image"]:
            img = ensure_pil(img)
            img = img.convert("RGB")
            new_images.append(transform(img))
        example["image"] = new_images
    else:
        img = ensure_pil(example["image"])
        img = img.convert("RGB")
        example["image"] = transform(img)
    return example


def get_transforms(img_size=224):
    """Get training and validation transforms"""
    # Define base transforms (without forcing RGB conversion; that's done in robust_transform)
    base_train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    base_val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Define final transforms using our robust_transform function.
    train_transform = lambda ex: robust_transform(ex, base_train_transforms)
    val_transform = lambda ex: robust_transform(ex, base_val_transforms)
    
    return train_transform, val_transform


def create_data_loaders(dataset_name, train_transform, val_transform, batch_size=32):
    """Create training and validation data loaders"""
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Split into train and validation
    labeled_train = dataset['train']
    labeled_val = dataset['validation'] if 'validation' in dataset else dataset['test']
    
    # Set transforms on processed datasets
    labeled_train.set_transform(train_transform)
    labeled_val.set_transform(val_transform)
    
    # Create DataLoaders using default collate (transforms return proper tensors)
    train_loader = DataLoader(labeled_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(labeled_val, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, labeled_train, labeled_val
