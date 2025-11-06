"""Flower server for VFL."""

import torch
from flwr.server import ServerApp, ServerConfig
from flwr.common import Context
from vfedentity.strategy import VFLStrategy
from vfedentity.utils import load_config


def server_fn(context: Context):
    """Create VFL server from YAML config."""
    
    # Load config from YAML
    config_path = context.run_config.get("config-path", "config.yaml")
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create strategy
    strategy = VFLStrategy(config=config, device=device)
    server_config = ServerConfig(num_rounds=config.total_rounds)
    
    return strategy, server_config


app = ServerApp(server_fn=server_fn)
