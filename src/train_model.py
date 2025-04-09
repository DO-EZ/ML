import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from data_loader import get_dataloaders
from eval import evaluate
from models import HybridCNN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train(num_epochs=20, learning_rate=0.001, batch_size=64, seed=42):

    set_seed(seed)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mnist-digit-captcha")    
    with mlflow.start_run():    
        device = get_device()
        print(f"Using device: {device}")

        train_loader, test_loader = get_dataloaders(batch_size=batch_size)
        model = HybridCNN().to(device)   
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
   
        best_accuracy = 0.0
    
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0
            all_preds, all_labels = [], []
            for images, labels in train_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())    
            train_acc = correct / total
            avg_loss = running_loss / len(train_loader)
            val_acc = evaluate(model, test_loader, device)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)    
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")    
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("precision", precision, step=epoch)
            mlflow.log_metric("recall", recall, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)   
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), "../tests/best_model.pth")
                mlflow.log_artifact("../tests/best_model.pth")
                print(f"Best model saved and logged with val acc: {val_acc:.4f}")   
            scheduler.step()    
        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name="HybridCNN")
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("weight_decay", 1e-4)
        mlflow.log_param("model_architecture", "HybridCNN")
        mlflow.log_param("seed", seed)

if __name__ == "__main__":
    train()
