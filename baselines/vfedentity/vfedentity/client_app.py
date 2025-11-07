"""Flower client for VFL with batching support."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, NDArrays, Scalar
from typing import Dict, Tuple
from collections import OrderedDict

from vfedentity.task import ClientModel, load_partition
from vfedentity.utils import load_config, VFLConfig

class VFLClient(NumPyClient):
    """VFL Client with batch-level training."""
    
    def __init__(
        self,
        train_loader = None,
        test_loader = None,
        v_split_id: int = None,
        config: VFLConfig = None, 
        device: str = None,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.v_split_id = v_split_id
        self.device = device,
        self.config = config

        self.num_channels = config.num_channels
        self.mode = config.mode
        self.embedding_dim = config.client_embedding_dim 

        # Initialize model
        self.model = ClientModel(
            input_channels=self.num_channels,
            embedding_dim=self.embedding_dim
        ).to(device)
        
        if self.mode == "train" or self.mode == "train-val":
            self.train_iter = iter(self.train_loader)
            if self.test_loader is not None:  
                self.test_iter = iter(self.test_loader)
            else:
                self.test_iter = None
            
            if config.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            elif config.optimizer =="sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
            self.is_testing = False # flag

        if self.mode == "test":
            self.test_iter = iter(self.test_loader)
            self.is_testing = True # flag

        self.current_batch_data = None
        self.current_embedding = None
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return empty - VFL doesn't share model parameters."""
        return []
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Forward pass for training and testing."""

        self.is_testing = config["is_testing"]

        try:
            if self.is_testing:
                self.model.eval()
                with torch.no_grad():
                    try:
                        images, labels, entity_ids = next(self.test_iter)
                    except StopIteration:
                        self.test_iter = iter(self.test_loader)
                        images, labels, entity_ids = next(self.test_iter)
                    
                    images = images.to(self.device)
                    embeddings = self.model(images)  

            else:
                self.model.train()
                try:
                    images, labels, entity_ids = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    images, labels, entity_ids = next(self.train_iter)
            
                images = images.to(self.device)
                embeddings = self.model(images)  
                self.current_embeddings = embeddings # Only store if training (for backward)
                
            # detach just copies the tensor, computational graph of original tensor is not broken(verify?)
            return (
                    [
                        embeddings.detach().cpu().numpy(),
                        entity_ids.numpy(),
                        labels.numpy()
                    ],
                    len(images),
                    {
                        "v_split_id": float(self.v_split_id),
                        "is_testing": float(self.is_testing) 
                    }
                )

        except Exception as e:
            print(f"ERROR in Client {self.v_split_id} fit(): {e}", flush=True)
            raise
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """Backward pass during training, nothing during testing."""

        self.is_testing = config["is_testing"]
        
        try:
            if self.is_testing: 
                # Testing phase - nothing to do (embeddings already computed in fit)
                return 0.0, 0, {}
            else:
                # Training phase - backward pass
                if self.current_embeddings is None:
                    return 0.0, 0, {}
                
                self.optimizer.zero_grad()
                
                if len(parameters) > 0:
                    gradients = torch.from_numpy(parameters[0]).float().to(self.device)
                    self.current_embeddings.backward(gradients)
                    self.optimizer.step()
                
                batch_size = len(self.current_embeddings)
                self.current_embeddings = None
                
                return 0.0, batch_size, {}

        except Exception as e:
            print(f"ERROR in Client {self.v_split_id} evaluate(): {e}", flush=True)
            raise


def client_fn(context: Context):
    """Create VFL client instance."""
    
    # Get configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load config from YAML
    config_path = context.run_config.get("config-path", "config.yaml")
    config = load_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load partition
    train_loader, test_loader, v_split_id, channels = load_partition(
        partition_id, num_partitions, config
    )
    
    return VFLClient(
        train_loader,
        test_loader,
        v_split_id,
        config,
        device
    ).to_client()


app = ClientApp(client_fn=client_fn)
