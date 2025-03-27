import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

########################################
# 1. Define the Model Architecture
########################################
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_obj_classes):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.obj_head = nn.Linear(feature_dim, num_obj_classes)
        self.bin_head = nn.Linear(feature_dim, 2)
    
    def forward(self, x):
        feats = self.backbone(x)
        obj_logits = self.obj_head(feats)
        bin_logits = self.bin_head(feats)
        return obj_logits, bin_logits

########################################
# 2. Reconstruct the Model and Load Weights
########################################
num_obj_classes = 494  # Make sure this matches your training
device = torch.device("cpu")

resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Identity()
feature_dim = 2048
model = MultiTaskModel(resnet, feature_dim, num_obj_classes)
model.to(device)

repo_id = "Abdu07/multitask-model"
filename = "Yolloplusclassproject_weights.pth"
weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

########################################
# 3. Load Label Mapping and Define Transforms
########################################
# Load the saved mapping from JSON
with open("obj_label_mapping.json", "r") as f:
    obj_label_to_idx = json.load(f)
# Create the inverse mapping
idx_to_obj_label = {v: k for k, v in obj_label_to_idx.items()}

bin_label_names = ["AI-Generated", "Real"]

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

########################################
# 4. Define the Inference Function
########################################
def predict_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        obj_logits, bin_logits = model(img_tensor)
    obj_pred = torch.argmax(obj_logits, dim=1).item()
    bin_pred = torch.argmax(bin_logits, dim=1).item()
    obj_name = idx_to_obj_label.get(obj_pred, "Unknown")
    bin_name = bin_label_names[bin_pred]
    return f"Prediction: {obj_name} ({bin_name})"

########################################
# 5. Create Gradio UI
########################################
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="DualSight: Multi-Task Image Classifier for Content Verification Trained by Abdellahi El Moustapha",
    description="Upload an image to receive two predictions:\n1) The primary object in the image,\n2) Whether the image is AI-generated or Real."
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)