"""Model definitions and data loading for VFL with images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path
from vfedentity.utils import VFLConfig


class ClientModel(nn.Module):
    """Client-side model for processing vertical feature splits."""
    
    def __init__(self, input_channels: int, embedding_dim: int = 32):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
        # model.fc.in_features = 512 for resnet18
    def forward(self, xb):
        return self.model(xb)


class ServerModel(nn.Module):
    """Server-side model for final classification."""
    
    def __init__(self, total_embedding_dim: int, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(total_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, xb):
        return self.network(xb)


class VFLDataset(Dataset):
    """Custom dataset for VFL with entity tracking."""
    
    def __init__(self, images, labels, entity_ids, transform=None):
        self.images = images
        self.labels = labels
        self.entity_ids = entity_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.entity_ids)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        entity_id = self.entity_ids[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, entity_id


def load_cifar10(data_path: str = "./data") -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset."""
    
    # Download if not exists
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True
    )
    
    # Convert to numpy
    X_train = train_dataset.data  # Shape: (50000, 32, 32, 3)
    y_train = np.array(train_dataset.targets)
    
    X_test = test_dataset.data  # Shape: (10000, 32, 32, 3)
    y_test = np.array(test_dataset.targets)
    
    return (X_train, y_train), (X_test, y_test)


def partition_cifar10_vertical(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    overlap_ratio: float = 1.0,
    only_train_with_overlap: bool = False, # use True for exp that use only overlapping samples for training, hence will coem into play when overlap ratio is < 1
    seed: int = 42
) -> List[Dict]:
    """
    Partition CIFAR-10 vertically (spatially) across clients with controlled overlap.
    
    Args:
        X: Images array [num_samples, 32, 32, 3]
        y: Labels array [num_samples]
        num_clients: Number of vertical partitions (spatial splits)
        overlap_ratio: Ratio of entities common to all clients (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        List of dicts with keys: 'images', 'labels', 'entity_ids', 'v_split_id'
    """
    np.random.seed(seed)
    num_samples = len(X)
    height, width, channels = X.shape[1], X.shape[2], X.shape[3]
    
    # Define spatial splits for each client # Need to create a generalized partitioner
    # For 2 clients: left half and right half
    # For 3 clients: left third, middle third, right third
    # For 4 clients: top-left, top-right, bottom-left, bottom-right quadrants
    
    spatial_splits = []
    
    if num_clients == 2:
        # Vertical split: left and right halves
        mid_width = width // 2
        spatial_splits = [
            (0, height, 0, mid_width),           # Client 0: left half
            (0, height, mid_width, width)        # Client 1: right half
        ]
    
    elif num_clients == 3:
        # Vertical split into thirds
        third_width = width // 3
        spatial_splits = [
            (0, height, 0, third_width),                    # Client 0: left third
            (0, height, third_width, 2*third_width),        # Client 1: middle third
            (0, height, 2*third_width, width)               # Client 2: right third
        ]
    
    elif num_clients == 4:
        # Quadrant split
        mid_height = height // 2
        mid_width = width // 2
        spatial_splits = [
            (0, mid_height, 0, mid_width),              # Client 0: top-left
            (0, mid_height, mid_width, width),          # Client 1: top-right
            (mid_height, height, 0, mid_width),         # Client 2: bottom-left
            (mid_height, height, mid_width, width)      # Client 3: bottom-right
        ]
    
    else:
        raise ValueError(f"num_clients={num_clients} not supported. Use 2, 3, or 4 clients.")
    
    print(f"\nSpatial splits per client:")
    for i, (h_start, h_end, w_start, w_end) in enumerate(spatial_splits):
        print(f"  Client {i}: rows [{h_start}:{h_end}], cols [{w_start}:{w_end}]")
    
    # Step 2: Create entity overlaps based on overlap_ratio
    all_entity_ids = np.arange(num_samples)
    np.random.shuffle(all_entity_ids)
    
    num_overlap = int(num_samples * overlap_ratio)
    num_non_overlap = num_samples - num_overlap

    # Overlapping entities (common to all clients)
    overlapping_entities = all_entity_ids[:num_overlap]

    if only_train_with_overlap:
        num_non_overlap = 0
    
    # Non-overlapping pool
    non_overlapping_pool = all_entity_ids[num_overlap:]
    # Distribute non-overlapping entities among clients
    non_overlap_per_client = num_non_overlap // num_clients
    
    client_data = []
    
    for client_id in range(num_clients):
        # Step 3: Determine which entities this client has
        client_entities = overlapping_entities.copy()
        
        if num_non_overlap > 0: # if only_train_with_overlap flag is True, thus is 0 and this block wont execute, so non_overlapping pool is not added to cleint entities
            start_idx = client_id * non_overlap_per_client
            end_idx = start_idx + non_overlap_per_client
            
            # Handle remaining entities for last client
            if client_id == num_clients - 1:
                end_idx = len(non_overlapping_pool)
            
            client_specific = non_overlapping_pool[start_idx:end_idx]
            client_entities = np.concatenate([client_entities, client_specific])
        
        # Sort to maintain order (important for aligned batching)
        client_entities = np.sort(client_entities)
        
        # Step 4: Extract spatial region for this client
        h_start, h_end, w_start, w_end = spatial_splits[client_id]
        
        # Crop images to this client's spatial region
        client_images = X[client_entities, h_start:h_end, w_start:w_end, :]
        client_labels = y[client_entities]
        
        client_data.append({
            'images': client_images,
            'labels': client_labels,
            'entity_ids': client_entities,
            'v_split_id': client_id,
            'spatial_region': (h_start, h_end, w_start, w_end)
        })
        
        print(f"Client {client_id}: {len(client_entities)} entities, "
              f"shape {client_images.shape}, "
              f"overlap in the partitions created: {len(np.intersect1d(client_entities, overlapping_entities))}/{len(overlapping_entities)}")
    
    return client_data


def create_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    entity_ids: np.ndarray,
    batch_size: int,
    shuffle: bool = False
) -> DataLoader:
    """Create DataLoader for VFL with entity tracking."""
    
    # Transform: Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Works for any number of channels
    ])
    
    dataset = VFLDataset(images, labels, entity_ids, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False  # Keep all data
    )
    return dataloader


def load_partition(partition_id: int, num_partitions: int, config: VFLConfig):
    """
    Load data partition for a specific client.
    
    Args:
        partition_id: Client ID
        num_partitions: Total number of clients
        config
    """
    # Load full dataset
    (X_train, y_train), (X_test, y_test) = load_cifar10(config.data_path)
    
    # Partition training data
    train_partitions = partition_cifar10_vertical(
        X_train,
        y_train,
        num_clients=num_partitions,
        overlap_ratio=config.overlap_ratio,
        only_train_with_overlap=config.only_train_with_overlap,
        seed=config.seed,
    )
    
    # Partition test data (with 100% overlap for fair evaluation)
    test_partitions = partition_cifar10_vertical(
        X_test,
        y_test,
        num_clients=num_partitions,
        overlap_ratio=1.0,  # Full overlap for test
        only_train_with_overlap=config.only_train_with_overlap,
        seed=config.seed + 1
    )
    
    # Get this client's partition
    client_train_data = train_partitions[partition_id]
    client_test_data = test_partitions[partition_id]
    
    # Create DataLoaders
    train_loader = create_dataloader(
        client_train_data['images'],
        client_train_data['labels'],
        client_train_data['entity_ids'],
        batch_size=config.batch_size,
        shuffle=False  # CRITICAL: False , Shuffling already done via indices thorugh numpy
    )
    
    test_loader = create_dataloader(
        client_test_data['images'],
        client_test_data['labels'],
        client_test_data['entity_ids'],
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return (
        train_loader,
        test_loader,
        client_train_data['v_split_id'],
        client_train_data['channels']
    )
