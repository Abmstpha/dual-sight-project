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

## Project Structure
```
project/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # MultiTaskModel architecture
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── train.py             # Training script and functions
│   └── inference.py         # Inference and prediction classes
├── notebooks/
│   └── yolloplusclassproject.ipynb  # Original training notebook
├── assets/
│   ├── andrejAI.png         # Sample AI-generated image
│   ├── andrejreal.jpeg      # Sample real image
│   └── logo.png             # Project logo
├── data/
│   └── obj_label_mapping.json  # Object class mappings
├── docs/
│   └── Dualsight_slides.pdf    # Project presentation
├── models/                  # Trained models save here
├── logs/                    # Training logs save here
├── app.py                   # Gradio web interface
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## How to Use

### Option 1: Use the Gradio Web Interface
Run the interactive web app:
```bash
python app.py
```

### Option 2: Use the Modular API
```python
import torch
from src.inference import load_predictor

# Load the predictor
predictor = load_predictor(
    model_path="path/to/weights.pth",
    obj_label_mapping_path="data/obj_label_mapping.json",
    num_obj_classes=494
)

# Make predictions
result = predictor.predict("path/to/image.jpg")
print(result["formatted_prediction"])
```

### Option 3: Train Your Own Model
```python
from src.train import train_model

# Train the model
model = train_model(
    dataset_name="your_dataset",
    num_obj_classes=494,
    epochs=35,
    batch_size=32
)
```

### Option 4: Test Online Demo
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
