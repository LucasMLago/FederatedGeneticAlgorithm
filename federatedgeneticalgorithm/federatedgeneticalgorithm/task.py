"""FederatedGeneticAlgorithm: A Flower / PyTorch app."""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, random_split, Subset

from federatedgeneticalgorithm.genetic_algorithm import GeneticAlgorithm

from typing import Optional, Literal, Tuple
import numpy as np

class Net(nn.Module):
    """Simple CNN (LeNet/AlexNet based) model for MNIST classification."""

    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)

        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        # Fully coonnected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.bn(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(
        root="../data",
        train=True,
        transform=transform,
        download=True
    )

testset = torchvision.datasets.MNIST(
    root="../data",
    train=False,
    transform=transform,
    download=True
)

def genetic_algorithm(model: nn.Module, trainset: Dataset, testset: Dataset):
    ga = GeneticAlgorithm(model, trainset, testset)
    pop, log = ga.run()
    best_individual = ga.get_best_individuals(pop, k=1)[0]
    return best_individual

def get_partition(dataset: Dataset, partition_id: int, num_partitions: int, seed: int = 42) -> Subset:
    """Return an IID partition of a dataset."""
    rng = np.random.default_rng(seed)    
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    partition_size = len(dataset) // num_partitions
    start_idx = partition_id * partition_size
    
    # Last partition gets remaining samples
    if partition_id == num_partitions - 1:
        end_idx = len(dataset)
    else:
        end_idx = start_idx + partition_size

    partition_indices = indices[start_idx:end_idx].tolist()
    
    return Subset(dataset, partition_indices)

def build_dataloaders(trainset: Dataset, testset: Dataset, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, generator=torch.Generator().manual_seed(seed))

    return train_loader, val_loader, test_loader

def train(net: nn.Module, 
          trainloader: DataLoader, 
          epochs: int, 
          lr: float, 
          device: str, 
          optimizer: Literal["adam", "rmsprop", "sgd"], 
          weight_decay: float, 
          momentum: Optional[float] = None
        ) -> dict:
    
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if optimizer == "adam":
        local_optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        local_optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "sgd":
        local_optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

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
            loss.backward()
            local_optimizer.step()
            running_loss += loss.item() * y.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    avg_trainloss = running_loss / total
    avg_accuracy = correct / total
    
    return {
        "loss": avg_trainloss,
        "accuracy": avg_accuracy
    }


def test(net: nn.Module, testloader: DataLoader, device: str):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = criterion(logits, y)
            predicted = torch.argmax(logits, dim=1)
            accuracy = (predicted.eq(y).sum().item()) / y.shape[0]
            correct += accuracy * y.size(0)
            loss += loss.item() * y.size(0)
    avg_loss = loss / len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy
