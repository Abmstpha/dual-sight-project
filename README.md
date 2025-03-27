---
datasets:
- Hemg/AI-Generated-vs-Real-Images-Datasets(HUGGINGFACE)
metrics:
- accuracy
base_model:
- microsoft/resnet-50
pipeline_tag: image-classification
---

# DualSight: A Multi-Task Image Classifier for Object Recognition and Authenticity Verification

## Model Overview
This model is a **Multi-Task Image Classifier** that performs two tasks simultaneously:
1. **Object Recognition:** Identifies the primary objects in an image (e.g., "cat," "dog," "car," etc.) using pseudo-labels generated through a YOLO-based object detection approach.
2. **Authenticity Classification:** Determines whether the image is AI-generated or a real photograph.

The model uses a **ResNet-50** backbone with two heads: one for multi-class object recognition and another for binary classification (AI-generated vs. Real). It was trained on a subset of the [Hemg/AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets) and leverages YOLO for improved pseudo-labeling across the entire dataset.

## Model Details
- **Trained by:** [Abdellahi El Moustapha](https://abmstpha.github.io/)
- **Programming Language:** Python
- **Base Model:** ResNet-50
- **Datasets:** Hemg/AI-Generated-vs-Real-Images-Datasets
- **Library:** PyTorch
- **Pipeline Tag:** image-classification
- **Metrics:** Accuracy for both binary classification and multi-class object recognition
- **Version:** v1.0


## Intended Use
This model is designed for:
- **Digital Content Verification:** Detecting AI-generated images to help prevent misinformation.
- **Social Media Moderation:** Automatically flagging images that are likely AI-generated.
- **Content Analysis:** Assisting researchers in understanding the prevalence of AI art versus real images in digital media.

## How to Use
You can use this model locally or via the provided Hugging Face Space. For local usage, load the state dictionary into the model architecture using PyTorch. For example:
```python
import torch
from model import MultiTaskModel  # Your model definition

# Instantiate your model architecture (must match training)
model = MultiTaskModel(...)


# Load the saved state dictionary (trained weights)
model.load_state_dict(torch.load("DualSight.pth", map_location="cpu"))
model.eval()
```
Alternatively, you can test the model directly via our interactive demo:
[Test the Model Here(CLICK)](https://huggingface.co/spaces/Abdu07/DualSight-Demo)  

## Training Data and Evaluation
- **Dataset:** The model was trained on a subset of the [Hemg/AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets) comprising approximately 152k images.
- **Metrics:**
  - **Authenticity (AI vs. Real):** Validation accuracy reached around 85% after early epochs.
  - **Object Recognition:** Pseudo-label accuracy started at around 38–40% and improved during training.
- **Evaluation:** Detailed evaluation metrics and loss curves are available in our training logs.



## Limitations and Ethical Considerations
- **Pseudo-Labeling:** The object recognition task uses pseudo-labels generated from a pretrained model, which may introduce noise or bias.
- **Authenticity Sensitivity:** The binary classifier may face challenges with highly realistic AI-generated images.
- **Usage:** This model is intended for research and prototyping purposes. Additional validation is recommended before deploying in high-stakes applications.

## How to Cite
If you use this model, please cite:
```bibtex
@misc{multitask_classifier,
  title={Multi-Task Image Classifier},
  author={Abdellahi El Moustapha},
  year={2025},
  howpublished={\url{https://huggingface.co/Abdu07/multitask-model}}
}
```
