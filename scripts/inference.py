import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torchvision.models import efficientnet_v2_s

def evaluate_model(model_path, test_dir, img_size=(224,224), batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Figure out how many classes from test_dir
    num_classes = len(next(os.walk(test_dir))[1])

    # 2. Rebuild the same model architecture
    model = efficientnet_v2_s(weights=None)  # no pretrained weights
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # 3. Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. Data transforms
    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # 5. Test dataset/loader
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Run inference, gather predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. Compute metrics
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # 8. Print results
    print("Test Performance Metrics:")
    print("-------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    class_names = test_dataset.classes
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a test set and compute performance metrics (PyTorch).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved PyTorch model (e.g., best_model.pth).")
    parser.add_argument("--test_dir",   type=str, required=True, help="Path to the test directory (subfolders per class).")
    parser.add_argument("--img_size",   type=int, nargs=2, default=[300, 300], help="Target image size (width height).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )