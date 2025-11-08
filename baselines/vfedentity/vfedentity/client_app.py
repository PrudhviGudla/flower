"""Flower client for VFL with batching support."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, NDArrays, Scalar
from typing import Dict, Tuple
from collections import OrderedDict

from vfedentity.task import ClientModel, load_partition
from vfedentity.utils import load_config, VFLConfig
import json
from pathlib import Path
import uuid

# Create client state file
CLIENT_STATE_DIR = Path("/tmp/vfl_client_states")
CLIENT_STATE_DIR.mkdir(exist_ok=True)

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

        self.client_id = str(uuid.uuid4())[:8]
        self.state_file = CLIENT_STATE_DIR / f"client_{self.client_id}_{v_split_id}.json"
        
        self._write_state("init_started", {"v_split_id": v_split_id})


        self.train_loader = train_loader
        self.test_loader = test_loader
        self.v_split_id = v_split_id
        self.device = device
        self.config = config

        self.num_channels = config.num_channels
        self.mode = config.mode
        self.embedding_dim = config.client_embedding_dim 



        # Initialize model

        try:
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
            self.current_embeddings = None
            self._write_state("init_complete", {"device": str(device), "mode": self.mode})
        

        except Exception as e:
            self._write_state("init_error", {"error": str(e)})
            raise

    def _write_state(self, status, data=None):
        """Write client state to file for debugging."""
        state = {
            "status": status,
            "v_split_id": self.v_split_id,
            "timestamp": str(__import__('datetime').datetime.now()),
            "data": data or {}
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return empty - VFL doesn't share model parameters."""
        return []
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Forward pass for training and testing."""
        self._write_state("fit_started", {"round": config.get("server_round")})

        try:
            self.is_testing = config.get("is_testing", False)
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
            self._write_state("fit_error", {"error": str(e)})
            raise
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """Backward pass during training, nothing during testing."""

        self.is_testing = config.get("is_testing", False)
        
        try:
            if self.is_testing: 
                # Testing phase - nothing to do (embeddings already computed in fit)
                return 0.0, 0, {}
            else:
                # Training phase - backward pass
                if self.current_embeddings is None:
                    return 0.0, 1, {}
                
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
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    state_file = CLIENT_STATE_DIR / f"client_fn.json"
    state = {"log": "start", "partition": partition_id}
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    # print(f"\n[CLIENT_FN] Creating client - partition {partition_id}/{num_partitions}", flush=True)
    
    try:
        # Load config from YAML
        config_path = context.run_config.get("config-path", "config.yaml")
        # print(f"[CLIENT_FN] Loading config from: {config_path}", flush=True)
        config = load_config(config_path)
        # print(f"[CLIENT_FN] Config loaded successfully", flush=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"[CLIENT_FN] Device: {device}", flush=True)

        state = {"log": "config loaded"}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Load partition
        # print(f"[CLIENT_FN] Loading partition {partition_id}/{num_partitions}...", flush=True)
        train_loader, test_loader, v_split_id, channels = load_partition(
            partition_id, num_partitions, config
        )
        # print(f"[CLIENT_FN] Partition loaded - v_split_id={v_split_id}, channels={channels}", flush=True)
        
        # print(f"[CLIENT_FN] Creating VFLClient instance...", flush=True)

        state = {"log": "partition loaded", "vsplitid": v_split_id}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        client = VFLClient(
            train_loader=train_loader,
            test_loader=test_loader,
            v_split_id=v_split_id,
            config=config,
            device=device
        )
        # print(f"[CLIENT_FN] VFLClient created successfully", flush=True)
        
        # print(f"[CLIENT_FN] Converting to Flower client...", flush=True)
        return client.to_client()
    
    except Exception as e:
        # print(f"[CLIENT_FN] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


app = ClientApp(client_fn=client_fn)
