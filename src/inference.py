import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from .model import create_model


class MultiTaskPredictor:
    def __init__(self, model_path, obj_label_mapping_path, num_obj_classes=494, device=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model weights
            obj_label_mapping_path: Path to the object label mapping JSON file
            num_obj_classes: Number of object classes
            device: Device to use for inference
        """
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Load model
        self.model = create_model(num_obj_classes, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        with open(obj_label_mapping_path, "r") as f:
            self.obj_label_to_idx = json.load(f)
        self.idx_to_obj_label = {v: k for k, v in self.obj_label_to_idx.items()}
        
        self.bin_label_names = ["AI-Generated", "Real"]
        
        # Define transforms for inference
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Make predictions on a single image
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            dict: Dictionary containing predictions
        """
        # Handle image input
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or path to image file")
        
        # Preprocess image
        image = image.convert("RGB")
        img_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            obj_logits, bin_logits = self.model(img_tensor)
        
        # Get predictions
        obj_pred = torch.argmax(obj_logits, dim=1).item()
        bin_pred = torch.argmax(bin_logits, dim=1).item()
        
        # Get confidence scores
        obj_probs = torch.softmax(obj_logits, dim=1)
        bin_probs = torch.softmax(bin_logits, dim=1)
        
        obj_confidence = obj_probs[0, obj_pred].item()
        bin_confidence = bin_probs[0, bin_pred].item()
        
        # Map predictions to labels
        obj_name = self.idx_to_obj_label.get(obj_pred, "Unknown")
        bin_name = self.bin_label_names[bin_pred]
        
        return {
            "object_prediction": obj_name,
            "object_confidence": obj_confidence,
            "authenticity_prediction": bin_name,
            "authenticity_confidence": bin_confidence,
            "formatted_prediction": f"Prediction: {obj_name} ({bin_name})"
        }
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images
        
        Args:
            images: List of PIL Images or paths to image files
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for image in images:
            results.append(self.predict(image))
        return results


def load_predictor(model_path, obj_label_mapping_path, num_obj_classes=494, device=None):
    """
    Convenience function to load a predictor
    
    Args:
        model_path: Path to the saved model weights
        obj_label_mapping_path: Path to the object label mapping JSON file
        num_obj_classes: Number of object classes
        device: Device to use for inference
        
    Returns:
        MultiTaskPredictor: Initialized predictor
    """
    return MultiTaskPredictor(model_path, obj_label_mapping_path, num_obj_classes, device)
