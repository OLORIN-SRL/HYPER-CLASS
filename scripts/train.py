import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(train_dir, val_dir, model_save_path,
                img_size=(384, 384),
                batch_size=32,
                epochs=20,
                learning_rate=1e-4,
                patience=5):
    """
    1) Loads data from train/val directories (folder-per-class).
    2) Builds a pretrained EfficientNetV2S, freezes base layers, replaces final layer.
    3) Trains using Adam, logs metrics each epoch, saves best model, and early-stops.
    4) Returns the final model and a dict of all metrics.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Count number of classes from subfolders in train_dir
    num_classes = len(next(os.walk(train_dir))[1])

    # 2. Transforms
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # 3. Datasets & loaders
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset   = ImageFolder(root=val_dir,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 4. Build model (pretrained EfficientNetV2S)
    model = efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # final layer
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(device)

    # 5. Define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Setup to track best model + early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0

    # 7. Data structures to log metrics (like the custom Keras callback)
    metrics_logger = {
        "train_metrics": {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []},
        "val_metrics":   {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []},
    }

    # 8. Training loop
    for epoch in range(epochs):
        # === Train Phase ===
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.detach().cpu().numpy())

        train_loss = running_loss / len(train_dataset)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_recall    = recall_score(all_train_labels,  all_train_preds, average='weighted', zero_division=0)
        train_f1        = f1_score(all_train_labels,      all_train_preds, average='weighted', zero_division=0)

        # === Validation Phase ===
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.detach().cpu().numpy())

        val_loss = val_running_loss / len(val_dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_recall    = recall_score(all_val_labels,   all_val_preds, average='weighted', zero_division=0)
        val_f1        = f1_score(all_val_labels,       all_val_preds, average='weighted', zero_division=0)

        # === Store metrics for this epoch ===
        metrics_logger["train_metrics"]["loss"].append(train_loss)
        metrics_logger["train_metrics"]["accuracy"].append(train_accuracy)
        metrics_logger["train_metrics"]["precision"].append(train_precision)
        metrics_logger["train_metrics"]["recall"].append(train_recall)
        metrics_logger["train_metrics"]["f1"].append(train_f1)

        metrics_logger["val_metrics"]["loss"].append(val_loss)
        metrics_logger["val_metrics"]["accuracy"].append(val_accuracy)
        metrics_logger["val_metrics"]["precision"].append(val_precision)
        metrics_logger["val_metrics"]["recall"].append(val_recall)
        metrics_logger["val_metrics"]["f1"].append(val_f1)

        # === Print epoch summary ===
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, "
              f"Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, "
              f"Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")

        # === Checkpoint saving ===
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  [*] Best model saved to: {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered (no improvement in {patience} epochs).")
                break

    # === Load best model weights back ===
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    # === Save the collected metrics ===
    with open("training_metrics.pkl", "wb") as f:
        pickle.dump(metrics_logger, f)

    print("Training complete.")
    return model, metrics_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EfficientNetV2S classifier with PyTorch.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--val_dir",   type=str, required=True, help="Path to the validation data directory.")
    parser.add_argument("--model_save_path", type=str, default="best_model.pth", help="Filename to save the best model.")
    parser.add_argument("--img_size",  type=int, nargs=2, default=[300, 300], help="Target image size (width height).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs",     type=int, default=20, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_save_path=args.model_save_path,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience
    )
