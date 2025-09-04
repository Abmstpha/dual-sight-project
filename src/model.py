import torch
import torch.nn as nn
import torchvision.models as models


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


def create_model(num_obj_classes, pretrained=True):
    """Create and return the multi-task model"""
    # Load pre-trained ResNet-50 and modify
    resnet = models.resnet50(pretrained=pretrained)
    resnet.fc = nn.Identity()
    feature_dim = 2048
    
    model = MultiTaskModel(resnet, feature_dim, num_obj_classes)
    return model
