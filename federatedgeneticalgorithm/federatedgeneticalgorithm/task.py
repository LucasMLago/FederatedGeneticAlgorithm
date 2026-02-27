import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from lion_pytorch import Lion

from torch.utils.data import Dataset, DataLoader, random_split, Subset

from typing import Optional, Literal, Tuple, Dict
import numpy as np


class CNN(nn.Module):
    """CNN for CIFAR-10."""

    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.4)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        #Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.drop4(x)
        x = self.fc2(x)
        return x


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)


def get_partition(dataset: Dataset, partition_id: int, num_partitions: int, seed: int = 42) -> Subset:
    """Return an IID dataset partition (the last partition gets the remainder)."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    partition_size = len(dataset) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = len(dataset) if partition_id == num_partitions - 1 else start_idx + partition_size

    return Subset(dataset, indices[start_idx:end_idx].tolist())


def build_dataloaders(
    trainset: Dataset, testset: Dataset, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val (80/20 split) and test DataLoaders."""
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(trainset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, generator=gen)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, generator=gen)
    return train_loader, val_loader, test_loader


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    optimizer: Literal["adam", "adamw", "radam", "sgd", "lion"],
    weight_decay: float,
    momentum: Optional[float] = None,
    mu: float = 0.0,
    global_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> dict:
    """Train the model; if `mu>0`, add a proximal term w.r.t. `global_state_dict`."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    global_params = {}
    if mu > 0 and global_state_dict is not None:
        global_params = {k: v.to(device) for k, v in global_state_dict.items()}

    if optimizer == "adam":
        local_optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        local_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "radam" and hasattr(torch.optim, "RAdam"):
        local_optimizer = torch.optim.RAdam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        local_optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "lion":
        local_optimizer = Lion(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            local_optimizer.zero_grad()
            logits = net(x)

            loss = criterion(logits, y)
            
            # FedProx: Add proximal term to constrain local updates to global weights
            if mu > 0 and global_params:
                proximal_term = 0.0
                for name, param in net.named_parameters():
                    if name in global_params:
                        proximal_term += (param - global_params[name]).norm(2) ** 2
                loss += (mu / 2) * proximal_term

            loss.backward()
            local_optimizer.step()

            running_loss += loss.item() * y.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

    return {"loss": running_loss / total, "accuracy": correct / total}


def test(net: nn.Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Evaluate on `testloader` and return (avg_loss, accuracy)."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss_batch = criterion(logits, y)
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted.eq(y).sum().item()) * 1.0
            loss += loss_batch.item() * y.size(0)
    avg_loss = loss / len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return float(avg_loss), float(accuracy)
