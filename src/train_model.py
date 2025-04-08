import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from eval import evaluate
from models import HybridCNN

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train(num_epochs=20, learning_rate=0.001, batch_size=64):
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

            train_acc = correct / total
            avg_loss = running_loss / len(train_loader)
            val_acc = evaluate(model, test_loader, device)

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), "../tests/best_model.pth")
                print(f"✅ Best model saved with val acc: {val_acc:.4f}")

            scheduler.step()

        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name="HybridCNN")
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("weight_decay", 1e-4)

if __name__ == "__main__":
    train()
