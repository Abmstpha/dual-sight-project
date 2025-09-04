import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from .model import create_model
from .data_loader import get_transforms, create_data_loaders


def train_model(dataset_name, num_obj_classes, epochs=35, patience=7, batch_size=32, 
                device=None, save_dir="models", log_dir="logs"):
    """
    Train the multi-task model
    
    Args:
        dataset_name: Name of the dataset to load
        num_obj_classes: Number of object classes
        epochs: Maximum number of epochs
        patience: Early stopping patience
        batch_size: Batch size for training
        device: Device to use for training
        save_dir: Directory to save model weights
        log_dir: Directory to save logs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create model
    model = create_model(num_obj_classes, pretrained=True)
    model.to(device)
    
    # Get transforms and data loaders
    train_transform, val_transform = get_transforms()
    train_loader, val_loader, _, _ = create_data_loaders(
        dataset_name, train_transform, val_transform, batch_size
    )
    
    # Define loss functions and optimizer
    criterion_obj = nn.CrossEntropyLoss()
    criterion_bin = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.obj_head.parameters(), "lr": 5e-4},
        {"params": model.bin_head.parameters(), "lr": 5e-4}
    ])
    
    # Setup tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "multitask_experiment"))
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)  # [B, 3, 224, 224]
            obj_labels = batch["obj_label"].to(device)
            bin_labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            obj_logits, bin_logits = model(images)
            loss1 = criterion_obj(obj_logits, obj_labels)
            loss2 = criterion_bin(bin_logits, bin_labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_obj = 0
        total_obj = 0
        correct_bin = 0
        total_bin = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                obj_labels = batch["obj_label"].to(device)
                bin_labels = batch["label"].to(device)
                obj_logits, bin_logits = model(images)
                loss1 = criterion_obj(obj_logits, obj_labels)
                loss2 = criterion_bin(bin_logits, bin_labels)
                loss = loss1 + loss2
                val_loss += loss.item()
                _, obj_preds = torch.max(obj_logits, dim=1)
                _, bin_preds = torch.max(bin_logits, dim=1)
                correct_obj += (obj_preds == obj_labels).sum().item()
                total_obj += obj_labels.size(0)
                correct_bin += (bin_preds == bin_labels).sum().item()
                total_bin += bin_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_obj_acc = correct_obj / total_obj
        val_bin_acc = correct_bin / total_bin
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, " +
              f"Val Obj Acc = {val_obj_acc:.4f}, Val AI/Real Acc = {val_bin_acc:.4f}")
        
        # Log metrics
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Object", val_obj_acc, epoch)
        writer.add_scalar("Accuracy/RealVsAI", val_bin_acc, epoch)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_path = os.path.join(save_dir, "Yolloplusclassproject_weights.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    writer.close()
    
    # Load best model weights
    best_model_path = os.path.join(save_dir, "Yolloplusclassproject_weights.pth")
    model.load_state_dict(torch.load(best_model_path))
    
    return model


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # You would need to specify your dataset name and number of classes
    # model = train_model("your_dataset_name", num_obj_classes=494, device=device)
