import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score

import mlflow
import mlflow.pytorch

from fetch_user_images import fetch_and_extract_user_images
from data_loader import get_combined_dataloaders, get_mnist_only_dataloaders
from models import HybridCNN
from eval import evaluate
from mlflow.tracking import MlflowClient
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def train(
    be_base_url: str,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 1,
    seed: int = 42
):
    # ─── 랜덤 시드 고정 ───────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_experiment("mnist-digit-captcha")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "be_base_url": be_base_url
        })

        # ─── 1) BE로부터 사용자 캡차 데이터 받아오기 ────────────────────────────────
        user_data_available = True
        try:
            fetch_and_extract_user_images(be_base_url)
        except Exception as e:
            print(f"[Train] 사용자 데이터 가져오기 실패 → MNIST 전용으로 학습: {e}")
            user_data_available = False

        # ─── 2) DataLoader 준비 ─────────────────────────────────────────────────
        if user_data_available:
            try:
                train_loader, test_loader = get_combined_dataloaders(
                    batch_size=batch_size
                )
            except Exception as e:
                print(f"[Train] 사용자 데이터 처리 실패 → MNIST 전용: {e}")
                train_loader, test_loader = get_mnist_only_dataloaders(
                    batch_size=batch_size
                )
        else:
            train_loader, test_loader = get_mnist_only_dataloaders(
                batch_size=batch_size
            )

        # ─── 3) 모델/손실함수/옵티마이저/스케줄러 설정 ──────────────────────────────
        model = HybridCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # ─── 예시 입력 생성 (input_example) ──────────────────────────────────────
        example_input = torch.randn(1, 1, 28, 28, dtype=torch.float32).to(device)
        example_input_np = example_input.cpu().numpy()
        
        best_f1 = 0.0
        best_model_path = None

        # ─── 4) 학습 루프 ─────────────────────────────────────────────────────────
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []

            for images, labels in train_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

            # ─── 학습 정확도 계산 ────────────────────────────────────────────────
            epoch_loss = epoch_loss / len(train_loader.dataset)
            train_acc = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)

            # ─── 검증 정확도 계산 ────────────────────────────────────────────────
            val_acc = evaluate(model, test_loader, device)

            # ─── Precision, Recall, F1 계산 ─────────────────────────────────────
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy().tolist())
                    val_labels.extend(labels.cpu().numpy().tolist())

            precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            recall    = recall_score(val_labels, val_preds, average='macro', zero_division=0)
            f1        = f1_score(val_labels, val_preds, average='macro', zero_division=0)

            # ─── MLflow 로깅 ───────────────────────────────────────────────────
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }, step=epoch)

            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Loss: {epoch_loss:.4f} "
                  f"Train Acc: {train_acc:.4f} "
                  f"Val Acc: {val_acc:.4f} "
                  f"F1: {f1:.4f}")

            # ─── 베스트 F1 모델 저장 ───────────────────────────────────────────
            if f1 > best_f1:
                best_f1 = f1
                best_model_path = f"best_model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), best_model_path)
                # mlflow.log_artifact(best_model_path)

            scheduler.step()

        # ─── 5) 학습 완료 후 MLflow에 베스트 모델만 등록 ────────────────────────────
        if best_model_path is None:
            raise RuntimeError("학습 도중 best_model_path가 설정되지 않았습니다.")

        # 1) 로컬에 저장된 최고 성능 모델 로드
        print(f"[Train] Best F1: {best_f1:.4f} → 모델 '{best_model_path}' 로드 중...")
        model.load_state_dict(torch.load(best_model_path))

        # 2) MLflow에 베스트 모델만 등록
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="HybridCNN"
        )
        client = MlflowClient() 
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(name="HybridCNN", source=model_uri, run_id=run_id)
        version = mv.version
        client.set_model_version_tag(name="HybridCNN", version=version, key="timestamp", value=timestamp)
        client.set_registered_model_alias(name="HybridCNN", alias=timestamp, version=version)
        
        print("[Train] MLflow에 HybridCNN (best) 버전으로 등록 완료")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--be_base_url", type=str, default=None, help="BE 서버 베이스 URL")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="학습률")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=20, help="에폭 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    train(
        be_base_url=args.be_base_url,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
