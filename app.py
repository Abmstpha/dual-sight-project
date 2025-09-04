import gradio as gr
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from src.inference import load_predictor

########################################
# 1. Load the Model and Setup Predictor
########################################
num_obj_classes = 494  # Make sure this matches your training
device = torch.device("cpu")

# Download weights from Hugging Face Hub
repo_id = "Abdu07/multitask-model"
filename = "Yolloplusclassproject_weights.pth"
weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the predictor using the modular structure
predictor = load_predictor(
    model_path=weights_path,
    obj_label_mapping_path="data/obj_label_mapping.json",
    num_obj_classes=num_obj_classes,
    device=device
)

########################################
# 2. Define the Inference Function
########################################
def predict_image(img: Image.Image) -> str:
    """Predict using the modular inference system"""
    result = predictor.predict(img)
    return result["formatted_prediction"]

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