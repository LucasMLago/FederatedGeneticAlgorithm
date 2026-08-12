import gc
import os
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from lion_pytorch import Lion

from torch.utils.data import Dataset, DataLoader, random_split, Subset

from typing import Optional, Literal, Tuple, Dict, List
import numpy as np

from federatedgeneticalgorithm.config import config


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# from the LEAF preprocessing
FEMNIST_MEAN = (0.9637,)
FEMNIST_STD = (0.1591,)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# old name for CIFARResNet, unused inside the package
CNN = CIFARResNet


class CIFARSmallCNN(nn.Module):
    """2-conv CNN, ~530K params (CIFARResNet has ~11M)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FEMNISTCNN(nn.Module):
    """CNN from the LEAF benchmark (Caldas et al. 2018)."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


_cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_cifar_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# no augmentation here, a flipped or cropped character can become another one
_femnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(FEMNIST_MEAN, FEMNIST_STD),
])

# old names, unused inside the package
transform_train = _cifar_transform_train
transform_test = _cifar_transform_test


def _resolve_data_root() -> str:
    # must be absolute, Ray workers inherit env but run in a different cwd
    env_root = os.environ.get("FGA_DATA_ROOT")
    if env_root:
        return str(Path(env_root).expanduser().resolve())
    # Repo layout: <repo>/federatedgeneticalgorithm/federatedgeneticalgorithm/task.py
    repo_default = Path(__file__).resolve().parents[2] / "data"
    return str(repo_default)


_CIFAR_ROOT = _resolve_data_root()


class _FEMNISTSubset(Dataset):
    """Random subsample of a HF FEMNIST split, kept in memory."""

    def __init__(self, hf_split, n: int, seed: int, transform):
        rng = np.random.default_rng(seed)
        total = len(hf_split)
        if n >= total:
            indices = np.arange(total)
        else:
            indices = rng.choice(total, size=n, replace=False)
        indices = np.sort(indices)  # sequential access is faster for HF datasets
        # 60k * 28*28 uint8 is ~47MB, fine to keep in memory
        images = []
        labels = []
        for i in indices.tolist():
            sample = hf_split[i]
            images.append(np.array(sample["image"], dtype=np.uint8))
            labels.append(int(sample["character"]))
        self._images = np.stack(images, axis=0)  # (N, 28, 28) uint8
        self._labels = np.asarray(labels, dtype=np.int64)
        self.targets = self._labels  # used by _dataset_labels() for Dirichlet partitioning
        self._transform = transform

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(self, idx):
        img = self._images[idx]
        # back to PIL so the transforms get the same input type as torchvision datasets
        from PIL import Image
        pil = Image.fromarray(img, mode="L")
        return self._transform(pil), int(self._labels[idx])


def _load_cifar10() -> Tuple[Dataset, Dataset]:
    train = torchvision.datasets.CIFAR10(
        root=_CIFAR_ROOT, train=True, download=True, transform=_cifar_transform_train
    )
    test = torchvision.datasets.CIFAR10(
        root=_CIFAR_ROOT, train=False, download=True, transform=_cifar_transform_test
    )
    return train, test


def _load_femnist() -> Tuple[Dataset, Dataset]:
    from datasets import load_dataset

    # HF cache under the data root, so reruns don't download again
    hf_cache = os.environ.get("FGA_DATA_ROOT") or str(Path(_CIFAR_ROOT))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(hf_cache) / "hf_datasets"))

    train_full = load_dataset("flwrlabs/femnist", split="train")
    # flwrlabs/femnist only has a train split, the test set is carved out of it below
    n_train = int(getattr(config, "FEMNIST_TRAIN_SAMPLES", 60000))
    n_test = int(getattr(config, "FEMNIST_TEST_SAMPLES", 10000))
    seed = int(getattr(config, "SEED", 0))

    rng = np.random.default_rng(seed)
    total = len(train_full)
    take = min(n_train + n_test, total)
    idxs = rng.choice(total, size=take, replace=False)
    rng.shuffle(idxs)
    train_idx = np.sort(idxs[:n_train])
    test_idx = np.sort(idxs[n_train : n_train + n_test])

    train_split = train_full.select(train_idx.tolist())
    test_split = train_full.select(test_idx.tolist())

    train = _FEMNISTSubset(train_split, n=n_train, seed=seed, transform=_femnist_transform)
    test = _FEMNISTSubset(test_split, n=n_test, seed=seed + 1, transform=_femnist_transform)
    return train, test


_DATASET_LOADERS = {
    "cifar10": _load_cifar10,
    "cifar10_small": _load_cifar10,  # same data, smaller model
    "femnist": _load_femnist,
}

_MODEL_FACTORIES = {
    "cifar10": lambda: CIFARResNet(num_classes=10),
    "cifar10_small": lambda: CIFARSmallCNN(num_classes=10),
    "femnist": lambda: FEMNISTCNN(num_classes=62),
}


def _dataset_name() -> str:
    name = str(getattr(config, "DATASET_NAME", "cifar10")).lower()
    if name not in _DATASET_LOADERS:
        raise ValueError(
            f"Unsupported DATASET_NAME={name!r}; choose one of {sorted(_DATASET_LOADERS)}"
        )
    return name


def build_model() -> nn.Module:
    return _MODEL_FACTORIES[_dataset_name()]()


# loaded at import time because client_app and server_app import trainset/testset directly
trainset, testset = _DATASET_LOADERS[_dataset_name()]()


# get_partition runs for every client every round, so each split is computed once
_PARTITION_CACHE: Dict[Tuple, List[List[int]]] = {}


def _iid_indices(num_samples: int, num_partitions: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.arange(num_samples)
    rng.shuffle(shuffled)
    part_size = num_samples // num_partitions
    result = []
    for p in range(num_partitions):
        start = p * part_size
        end = num_samples if p == num_partitions - 1 else start + part_size
        result.append(shuffled[start:end].tolist())
    return result


def _dirichlet_indices(
    labels: np.ndarray, num_partitions: int, alpha: float, seed: int
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]

    for c in classes:
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet([alpha] * num_partitions)
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)
        for i, s in enumerate(splits):
            partitions[i].extend(s.tolist())

    # otherwise each partition comes out sorted by class
    for p in partitions:
        rng.shuffle(p)
    return partitions


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    raise ValueError("Dataset has no .targets attribute; cannot Dirichlet-partition.")


def _compute_partitions(
    dataset: Dataset, num_partitions: int, mode: str, alpha: float, seed: int
) -> List[List[int]]:
    key = (id(dataset), num_partitions, mode, alpha, seed)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    if mode == "dirichlet":
        labels = _dataset_labels(dataset)
        indices = _dirichlet_indices(labels, num_partitions, alpha, seed)
    else:  # "iid"
        indices = _iid_indices(len(dataset), num_partitions, seed)

    _PARTITION_CACHE[key] = indices
    return indices


def get_partition(
    dataset: Dataset,
    partition_id: int,
    num_partitions: int,
    seed: int = 42,
    force_iid: bool = False,
) -> Subset:
    # force_iid is for the test set, so eval isn't skewed by the train partitioning
    if force_iid:
        mode = "iid"
    else:
        mode = getattr(config, "PARTITION_MODE", "iid")
    alpha = float(getattr(config, "DIRICHLET_ALPHA", 0.3))
    partitions = _compute_partitions(dataset, num_partitions, mode, alpha, seed)
    return Subset(dataset, partitions[partition_id])


def partition_class_distribution(
    dataset: Dataset,
    num_partitions: int,
    seed: int = 42,
    force_iid: bool = False,
) -> List[Dict[int, int]]:
    mode = "iid" if force_iid else getattr(config, "PARTITION_MODE", "iid")
    alpha = float(getattr(config, "DIRICHLET_ALPHA", 0.3))
    partitions = _compute_partitions(dataset, num_partitions, mode, alpha, seed)
    labels = _dataset_labels(dataset)
    histograms = []
    for idxs in partitions:
        part_labels = labels[idxs]
        classes, counts = np.unique(part_labels, return_counts=True)
        histograms.append({int(c): int(n) for c, n in zip(classes, counts)})
    return histograms


def build_dataloaders(
    trainset: Dataset, testset: Dataset, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val (80/20 split) and test DataLoaders."""
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(trainset, [train_size, val_size], generator=gen)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        generator=gen,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        generator=gen,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        generator=gen,
    )
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
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    global_params = {}
    if mu > 0 and global_state_dict is not None:
        global_params = {k: v.to(device) for k, v in global_state_dict.items()}

    local_optimizer = _build_optimizer(net, optimizer, lr, weight_decay, momentum)

    net.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(local_optimizer, T_max=epochs)
    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            local_optimizer.zero_grad()
            logits = net(x)

            loss = criterion(logits, y)
            
            # FedProx term, pulls the weights back toward the global model
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

        scheduler.step()

    metrics = {"loss": running_loss / total, "accuracy": correct / total}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return metrics


def _build_optimizer(
    net: nn.Module,
    optimizer: Literal["adam", "adamw", "radam", "sgd", "lion"],
    lr: float,
    weight_decay: float,
    momentum: Optional[float],
):
    if optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "radam" and hasattr(torch.optim, "RAdam"):
        return torch.optim.RAdam(net.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if optimizer == "lion":
        return Lion(net.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer}")


def test(net: nn.Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Returns (loss, accuracy)."""
    net.to(device)
    net.eval()
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return float(avg_loss), float(accuracy)
