from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spiking_neuron import LIFNeuron, TernaryLIFNeuron
from spiking_transformer import SpikingTransformer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description='Train a Spiking Transformer on CIFAR-10.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--timesteps', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='checkpoints/best_spiking_transformer.pt',
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Select the best available training device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def reset_neurons(model: nn.Module) -> None:
    """Reset all stateful spiking neurons in a model.

    Args:
        model: Model containing spiking neuron modules.
    """
    for module in model.modules():
        if isinstance(module, (LIFNeuron, TernaryLIFNeuron)):
            module.reset()


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 training and validation data loaders.

    Args:
        data_dir: Root directory for dataset storage.
        batch_size: Mini-batch size.
        num_workers: Number of dataloader workers.
        device: Active training device.

    Returns:
        Pair of train and validation data loaders.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=True,
    )

    pin_memory = device.type == 'cuda'
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation loader.

    Args:
        model: Model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Active training device.

    Returns:
        Tuple of average loss and top-1 accuracy in percent.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            reset_neurons(model)
            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = 100.0 * total_correct / total_examples
    return average_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full training epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Active training device.

    Returns:
        Tuple of average loss and top-1 accuracy in percent.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        reset_neurons(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = 100.0 * total_correct / total_examples
    return average_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_accuracy: float,
    checkpoint_path: Path,
) -> None:
    """Persist the best checkpoint to disk.

    Args:
        model: Trained model.
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        epoch: Current epoch number.
        best_val_accuracy: Best validation accuracy so far.
        checkpoint_path: Destination path for the checkpoint file.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_accuracy': best_val_accuracy,
        },
        checkpoint_path,
    )


def main() -> None:
    """Train the Spiking Transformer on CIFAR-10."""
    args = parse_args()
    device = get_device()
    print(f'Using device: {device}')

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = SpikingTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        T=args.timesteps,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_path = Path(args.checkpoint_path)

    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        print(
            f'Epoch [{epoch:03d}/{args.epochs:03d}] '
            f'train_loss={train_loss:.4f} '
            f'train_acc@1={train_accuracy:.2f}% '
            f'val_loss={val_loss:.4f} '
            f'val_acc@1={val_accuracy:.2f}%'
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                checkpoint_path=checkpoint_path,
            )
            print(
                f'Saved new best checkpoint to {checkpoint_path} '
                f'(val_acc@1={best_val_accuracy:.2f}%).'
            )


if __name__ == '__main__':
    main()
