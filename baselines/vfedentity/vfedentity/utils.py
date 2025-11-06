"""Utility functions for VFL experiments."""

import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

load_dotenv()


class VFLConfig:
    """Configuration class for VFL experiments (flat structure)."""
    
    def __init__(
        self,
        # Experiment
        experiment_name: str = "vfl_baseline",
        experiment_description: str = "VFL experiment",
        seed: int = 42,       
        # Data
        dataset: str = "cifar10",
        data_path: str = "./data",
        num_train_samples: int = 50000,
        num_test_samples: int = 0,
        num_classes: int = 10,
        num_channels: int = 3,        
        # Model - Client
        client_architecture: str = "resnet18",
        client_pretrained: bool = False,
        client_embedding_dim: int = 128,        
        # Model - Server
        server_architecture: str = "mlp",
        server_hidden_dims: list = None,
        server_dropout: float = 0,   
        server_loss_method: str = "ce"   ,  
        # Training
        num_clients: int = 3,
        num_epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0,
        overlap_ratio: float = 1.0,
        only_train_with_overlap: bool = False,
        mode: str = "train-val",        
        # Paths
        exp_dir: str = "./experiments",
        resume_from: Optional[str] = None,       
        # Checkpoint
        save_every_n_epochs: int = 1,
        keep_last_n: int = 5,
        # Logging
        log_to_wandb: bool = True,
        wandb_project: str = "vfl-entity-augmentation",
        wandb_entity: Optional[str] = None,
        log_interval: int = 10,
        # Resources 
        client_num_cpus: int = 2,
        client_num_gpus: float = 0.3,
        server_num_cpus: int = 3,
        server_num_gpus: float = 0.3,
        
        **kwargs
    ):
        # Experiment
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.seed = seed        
        # Data
        self.dataset = dataset
        self.data_path = data_path
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.num_classes = num_classes
        self.num_channels = num_channels         
        # Model - Client
        self.client_architecture = client_architecture
        self.client_pretrained = client_pretrained
        self.client_embedding_dim = client_embedding_dim        
        # Model - Server
        self.server_architecture = server_architecture
        self.server_hidden_dims = server_hidden_dims if server_hidden_dims else [256, 128]
        self.server_loss_method = server_loss_method
        self.server_dropout = server_dropout       
        # Training
        self.num_clients = num_clients
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.overlap_ratio = overlap_ratio
        self.only_train_with_overlap = only_train_with_overlap
        self.mode = mode       
        # Paths
        self.exp_dir = exp_dir
        self.resume_from = resume_from        
        # Checkpoint
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n        
        # Logging
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.log_interval = log_interval        
        # Computed values (will be filled by _compute_derived_values)
        self.batches_per_epoch = 0
        self.total_rounds = 0
        self.evaluate_every_n_rounds = 0       
        # Paths (will be filled by _create_experiment_dirs)
        self.exp_dir = ""
        self.checkpoint_path = ""
        self.metrics_path = ""
        self.plots_path = ""
        self.run_name = ""
        # resources
        self.client_num_cpus = client_num_cpus
        self.client_num_gpus = client_num_gpus
        self.server_num_cpus = server_num_cpus
        self.server_num_gpus = server_num_gpus
                
        # Compute derived values
        self._compute_derived_values()
        self._create_experiment_dirs()
    
    def _compute_derived_values(self):
        """Compute derived configuration values based on mode."""
        
        num_train_batches = self.num_train_samples // self.batch_size
        num_test_batches = self.num_test_samples // self.batch_size
        
        if self.mode == "train":
            # Only training
            self.batches_per_epoch = num_train_batches
            self.total_rounds = num_train_batches * self.num_epochs
            self.evaluate_every_n_rounds = float('inf')  # Never evaluate
            
        elif self.mode == "train-val":
            # Training + Testing each epoch
            self.batches_per_epoch = num_train_batches + num_test_batches
            self.total_rounds = self.batches_per_epoch * self.num_epochs
            self.evaluate_every_n_rounds = num_train_batches  # Switch to testing after train batches
            
        elif self.mode == "test":
            # Only testing
            self.batches_per_epoch = num_test_batches
            self.total_rounds = num_test_batches * self.num_epochs
            self.evaluate_every_n_rounds = float('inf')  # Never evaluate since already eval

        print(f"Mode: {self.mode}")
        print(f"Train batches per epoch: {num_train_batches}")
        print(f"Test batches per epoch: {num_test_batches}")
        print(f"Batches per epoch: {self.batches_per_epoch}")
        print(f"Total rounds: {self.total_rounds}")
        print(f"Evaluate every N rounds: {self.evaluate_every_n_rounds}\n")
    
    def _create_experiment_dirs(self):
        """Create experiment directories and paths."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.experiment_name}_{timestamp}"
        # self.run_name = f"{self.experiment_name}_overlap{int(self.overlap_ratio*100)}_{timestamp}"

        self.exp_run_dir = os.path.join(self.exp_dir, self.run_name) # saves all artifacts of the run in this directory in th experiments directory(self.exp_dir)
        self.checkpoint_path = os.path.join(self.exp_run_dir, "checkpoints")
        self.metrics_path = os.path.join(self.exp_run_dir, "metrics")
        self.plots_path = os.path.join(self.exp_run_dir, "plots")
        
        # Create directories
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.metrics_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> VFLConfig:
    """
    Load configuration from YAML file and return VFLConfig instance.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        VFLConfig instance with all parameters loaded
    """
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested dictionary into kwargs for VFLConfig
    kwargs = {}
    
    # Experiment section
    if 'experiment' in config_dict:
        exp = config_dict['experiment']
        kwargs['experiment_name'] = exp.get('name', 'vfl_baseline')
        kwargs['experiment_description'] = exp.get('description', 'VFL experiment')
        kwargs['seed'] = exp.get('seed', 42)
    
    # Data section
    if 'data' in config_dict:
        data = config_dict['data']
        kwargs['dataset'] = data.get('dataset', 'cifar10')
        kwargs['data_path'] = data.get('data_path', './data')
        kwargs['num_train_samples'] = data.get('num_train_samples', 50000)
        kwargs['num_test_samples'] = data.get('num_test_samples', 10000)
        kwargs['num_classes'] = data.get('num_classes', 10)
        kwargs['num_channels'] = data.get('num_channels', 3)
    
    # Model section
    if 'model' in config_dict:
        model = config_dict['model']
        
        # Client model
        if 'client' in model:
            client = model['client']
            kwargs['client_architecture'] = client.get('architecture', 'resnet18')
            kwargs['client_pretrained'] = client.get('pretrained', False)
            kwargs['client_embedding_dim'] = client.get('embedding_dim', 128)
        
        # Server model
        if 'server' in model:
            server = model['server']
            kwargs['server_architecture'] = server.get('architecture', 'mlp')
            kwargs['server_hidden_dims'] = server.get('hidden_dims', [256, 128])
            kwargs['server_dropout'] = server.get('dropout', 0)
    
    # Training section
    if 'training' in config_dict:
        training = config_dict['training']
        kwargs['num_clients'] = training.get('num_clients', 3)
        kwargs['num_epochs'] = training.get('num_epochs', 10)
        kwargs['batch_size'] = training.get('batch_size', 64)
        kwargs['learning_rate'] = training.get('learning_rate', 0.001)
        kwargs['optimizer'] = training.get('optimizer', 'adam')
        kwargs['weight_decay'] = training.get('weight_decay', 0)
        kwargs['overlap_ratio'] = training.get('overlap_ratio', 1.0)
        kwargs['only_train_with_overlap'] = training.get('only_train_with_overlap', False
        kwargs['mode'] = training.get('mode', 'train-val')
    
    # Paths section
    if 'paths' in config_dict:
        paths = config_dict['paths']
        kwargs['exp_dir'] = paths.get('exp_dir', './experiments')
        kwargs['resume_from'] = paths.get('resume_from', None)
    
    # Checkpoint section
    if 'checkpoint' in config_dict:
        checkpoint = config_dict['checkpoint']
        kwargs['save_every_n_epochs'] = checkpoint.get('save_every_n_epochs', 1)
        kwargs['keep_last_n'] = checkpoint.get('keep_last_n', 2)
    
    # Logging section
    if 'logging' in config_dict:
        logging = config_dict['logging']
        kwargs['log_to_wandb'] = logging.get('log_to_wandb', True)
        kwargs['wandb_project'] = logging.get('wandb_project', 'vfl-entity-augmentation')
        kwargs['wandb_entity'] = logging.get('wandb_entity', None)
        kwargs['log_interval'] = logging.get('log_interval', 10)

    if 'resources' in config_dict:
        resources = config_dict['resources']
        if 'client' in resources:
            kwargs['client_num_cpus'] = resources['client'].get('num_cpus', 2)
            kwargs['client_num_gpus'] = resources['client'].get('num_gpus', 0.25)
        if 'server' in resources:
            kwargs['server_num_cpus'] = resources['server'].get('num_cpus', 3)
            kwargs['server_num_gpus'] = resources['server'].get('num_gpus', 0.2)
    
    # Create VFLConfig instance
    config = VFLConfig(**kwargs)
    
    # Save config copy to experiment directory
    config.save(os.path.join(config.exp_dir, "config.yaml"))
    
    return config


def setup_wandb(config: VFLConfig):
    """Initialize Weights & Biases logging with authentication."""
    
    if not config.log_to_wandb:
        return None
    
    try:        
        # Login with API key from .env
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        
        # Check if resuming
        resume_mode = "allow" if config.resume_from else None
        resume_id = None
        
        if config.resume_from and os.path.exists(config.resume_from):
            # Try to extract run_id from checkpoint
            try:
                checkpoint = torch.load(config.resume_from, map_location='cpu')
                resume_id = checkpoint.get('wandb_run_id', None)
            except:
                pass
        
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity or os.getenv("WANDB_ENTITY"),
            name=config.run_name,
            id=resume_id,
            resume=resume_mode,
            config={
                'experiment': config.experiment_name,
                'overlap_ratio': config.overlap_ratio,
                'client_model': config.client_architecture,
                'server_model': config.server_architecture,
                'num_epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
            }
        )
        print(f"Weights & Biases initialized (Run ID: {wandb.run.id})")
        return wandb
    
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        return None


def save_metrics(metrics: Dict, config: VFLConfig):
    """Save metrics to JSON file."""
    
    metrics_file = os.path.join(config.metrics_path, "metrics.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved")


def plot_training_curves(train_losses = None, val_losses = None, val_accuracies = None, config: VFLConfig):
    """Plot and save training curves."""
    
    epochs = range(1, len(config.num_epochs) + 1)
    i = 0

    if config.mode == "train":
        k = 1
    elif config,mode == "train-val":
        k = 3
    elif config.mode == "test":
        k = 2
    
    fig, axes = plt.subplots(1, k, figsize=(15, 4))

    if train_losses and i<k:
        axes[i].plot(train_losses, linewidth=2)
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel('Loss', fontsize=12)
        axes[i].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        i+=1

    if val_losses and i<k:
        axes[i].plot(val_losses, linewidth=2)
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel('Loss', fontsize=12)
        axes[i].set_title('Validation/Test Loss', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        i+=1

    if val_accuracies and i<k:
        axes[i].plot(epochs, test_accuracies, linewidth=2, marker='o', color='red')
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel('Accuracy', fontsize=12)
        axes[i].set_title('Validation/Test Accuracy', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config.plots_path, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics  Curves saved")


# ============== LATER ===========================================================================

# def save_checkpoint(server_model, optimizer, epoch: int, round_num: int,
#                    metrics: Dict, config: VFLConfig, is_best: bool = False):
#     """Save model checkpoint."""
    
#     checkpoint = {
#         'epoch': epoch,
#         'round': round_num,
#         'server_model_state_dict': server_model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'metrics': metrics,
#         'config_snapshot': {
#             'overlap_ratio': config.overlap_ratio,
#             'num_clients': config.num_clients,
#             'learning_rate': config.learning_rate,
#         }
#     }
    
#     checkpoint_path = os.path.join(
#         config.checkpoint_path,
#         f"checkpoint_epoch_{epoch}.pt"
#     )
#     torch.save(checkpoint, checkpoint_path)
#     print(f"✓ Checkpoint saved: epoch {epoch}")
    
#     if is_best:
#         best_path = os.path.join(config.checkpoint_path, "best_model.pt")
#         torch.save(checkpoint, best_path)
#         print(f"✓ Best model updated")
    
#     # Save latest
#     latest_path = os.path.join(config.checkpoint_path, "latest.pt")
#     torch.save(checkpoint, latest_path)
    
#     # Cleanup old checkpoints
#     _cleanup_old_checkpoints(config)


# def _cleanup_old_checkpoints(config: VFLConfig):
#     """Keep only last N checkpoints."""
    
#     checkpoint_dir = Path(config.checkpoint_path)
    
#     checkpoints = sorted(
#         [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")],
#         key=lambda x: x.stat().st_mtime
#     )
    
#     # Keep best, latest, and last N
#     for ckpt in checkpoints[:-config.keep_last_n]:
#         if ckpt.name not in ["best_model.pt", "latest.pt"]:
#             ckpt.unlink()


# def load_checkpoint(checkpoint_path: str, server_model, optimizer, device):
#     """Load checkpoint from file."""
    
#     if not os.path.exists(checkpoint_path):
#         print(f"⚠ Checkpoint not found: {checkpoint_path}")
#         return None
    
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     server_model.load_state_dict(checkpoint['server_model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
#     print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
    
#     return checkpoint

    # def to_dict(self) -> Dict:
    #     """Convert config to dictionary for saving the finalconfig in exp directory"""
    #     return {
    #         'experiment': {
    #             'name': self.experiment_name,
    #             'description': self.experiment_description,
    #             'seed': self.seed,
    #         },
    #         'data': {
    #             'dataset': self.dataset,
    #             'data_path': self.data_path,
    #             'num_train_samples': self.num_train_samples,
    #             'num_test_samples': self.num_test_samples,
    #             'num_classes': self.num_classes,
    #         },
    #         'model': {
    #             'client': {
    #                 'architecture': self.client_architecture,
    #                 'pretrained': self.client_pretrained,
    #                 'embedding_dim': self.client_embedding_dim,
    #             },
    #             'server': {
    #                 'architecture': self.server_architecture,
    #                 'hidden_dims': self.server_hidden_dims,
    #                 'dropout': self.server_dropout,
    #             }
    #         },
    #         'training': {
    #             'num_clients': self.num_clients,
    #             'num_epochs': self.num_epochs,
    #             'batch_size': self.batch_size,
    #             'learning_rate': self.learning_rate,
    #             'optimizer': self.optimizer,
    #             'weight_decay': self.weight_decay,
    #             'overlap_ratio': self.overlap_ratio,
    #             'only_train_with_overlap': self.only_train_with_overlap,
    #             'mode': self.mode,
    #         },
    #         'paths': {
    #             'exp_dir': self.exp_dir,
    #             'resume_from': self.resume_from,
    #             'exp_run_dir': self.exp_run_dir,
    #         },
    #         'checkpoint': {
    #             'save_every_n_epochs': self.save_every_n_epochs,
    #             'keep_last_n': self.keep_last_n,
    #         },
    #         'logging': {
    #             'log_to_wandb': self.log_to_wandb,
    #             'wandb_project': self.wandb_project,
    #             'wandb_entity': self.wandb_entity,
    #             'log_interval': self.log_interval,
    #         },
    #         'computed': {
    #             'batches_per_epoch': self.batches_per_epoch,
    #             'total_rounds': self.total_rounds,
    #             'evaluate_every_n_rounds': self.evaluate_every_n_rounds,
    #         }
    #     }
    
    # def save(self, path: str):
    #     """Save config to YAML file."""
    #     config_dict = self.to_dict()
    #     with open(path, 'w') as f:
    #         yaml.dump(config_dict, f, default_flow_style=False)