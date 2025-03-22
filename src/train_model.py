import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from eval import evaluate
from models import SimpleCNN


def train():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mnist-digit-captcha")

    with mlflow.start_run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, test_loader = get_dataloaders()
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            val_accuracy = evaluate(model, test_loader, device)

            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        # 하이퍼파라미터 로깅
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("lr", 0.001)

        # 모델 저장
        mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    train()
