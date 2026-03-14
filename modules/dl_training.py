"""PyTorch training utilities for image and text models."""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any
import os


def set_seed(seed: int = 42) -> None:
    """Set random seed cho random, numpy, torch, CUDA.

    Args:
        seed: default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Class Early stopping

    Args:
        patience: số lần không improve tối đa, default 5
        min_delta: Delta metric nhỏ nhất để tính là improve, default 0.0
        mode: "min" cho loss, "max" cho accuracy,... , default "min"
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = None

        if mode == "min":
            self.best_metric = float('inf')
            self.is_improvement = lambda current, best: current < (best - min_delta)
        elif mode == "max":
            self.best_metric = float('-inf')
            self.is_improvement = lambda current, best: current > (best + min_delta)
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, metric: float) -> bool:
        """Kiểm tra dừng (số lần không improve lớn hơn patience)

        Args:
            metric: giắ trị metric mới nhận được

        Returns:
            True nếu dừng, False nếu không
        """
        if self.best_metric is None:
            self.best_metric = metric
            return False

        if self.is_improvement(metric, self.best_metric):
            self.best_metric = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Train 1 epoch.

    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (cpu or cuda)

    Returns:
        Tuple (average_loss, average_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        if outputs.dim() > 1:
            _, predicted = torch.max(outputs.data, 1)
        else:
            predicted = outputs.data
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate.

    Args:
        model: PyTorch model
        loader: Validation/test data loader
        criterion: Loss function
        device: Device (cpu or cuda)

    Returns:
        Tuple (average_loss, average_accuracy, y_true, y_pred)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy and predictions
            if outputs.dim() > 1:
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = outputs.data
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(predicted.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_correct / total_samples
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    return avg_loss, avg_accuracy, y_true, y_pred


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    early_stopping: Optional[EarlyStopping] = None,
    scheduler: Optional[LRScheduler] = None
) -> Dict[str, List[float]]:
    """Training loop với logging và early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        epochs: Số epoch train
        device: Device (cpu or cuda)
        early_stopping: EarlyStopping instance
        scheduler: LR scheduler

    Returns:
        History dict cho từng epoch train và val {"train_loss", "val_loss", "train_acc", "val_acc"}
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # Training epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation epoch
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step()

        # In epoch
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    return history


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """Plot training and validation loss and accuracy curves.

    Args:
        history: History dict with keys: "train_loss", "val_loss", "train_acc", "val_acc"
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(history["val_loss"], label="Val Loss", marker='s')
    best_epoch = np.argmin(history["val_loss"])
    axes[0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history["train_acc"], label="Train Acc", marker='o')
    axes[1].plot(history["val_acc"], label="Val Acc", marker='s')
    best_epoch_acc = np.argmax(history["val_acc"])
    axes[1].axvline(x=best_epoch_acc, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    path: str,
    device: torch.device
) -> int:
    """Load model checkpoint for resuming training or inference.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to checkpoint
        device: Device to load to

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch+1}")

    return epoch
