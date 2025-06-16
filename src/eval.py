import torch

def evaluate(model: torch.nn.Module, dataloader, device) -> float:
    """
    Validation/Test 시 모델 정확도를 계산.
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0
