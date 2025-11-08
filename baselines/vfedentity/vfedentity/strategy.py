import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import numpy as np
import flwr as fl
from vfedentity.task import ServerModel
from vfedentity.utils import VFLConfig, setup_wandb, save_metrics, plot_training_curves
# from flwr.common import ndarrays_to_parameters


class VFLStrategy(fl.server.strategy.FedAvg):
    """Server Strategy"""
    
    def __init__(self, config: VFLConfig, device: str):
        self.config = config
        self.device = device
        self.num_clients = config.num_clients
        self.embedding_dim = config.client_embedding_dim
        self.num_classes = config.num_classes
        self.evaluate_every_n_rounds = config.evaluate_every_n_rounds
        self.loss_method = config.server_loss_method
        self.mode = config.mode
        self.num_train_batches = config.num_train_batches
        self.num_test_batches = config.num_test_batches
        self.batches_per_epoch = config.batches_per_epoch

        total_embedding_dim = self.embedding_dim * self.num_clients
        
        self.server_model = ServerModel(
            total_embedding_dim=total_embedding_dim,
            num_classes=self.num_classes
        ).to(device)

        # Optimizer
        if self.mode == "train" or self.mode == "train-val":
            if config.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            elif config.optimizer =="sgd":
                self.optimizer = torch.optim.SGD(self.server_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)

        if self.mode == "test":
            self.is_testing = True # flag
        
        if self.loss_method == "kl":
            self.loss = nn.KLDivLoss(reduction='batchmean')
        elif self.loss_method == "ce":
            self.loss = nn.CrossEntropyLoss()


        # server_config = {
        #     'architecture': config.server_architecture,
        #     'hidden_dims': config.server_hidden_dims,
        #     'dropout': config.server_dropout
        # }
        # self.server_model = create_server_model(
        #     server_config,
        #     total_embedding_dim,
        #     self.num_classes
        # ).to(device)
        
        # Tracking
        self.train_loss = 0
        self.train_epoch_losses = []
        self.val_loss = 0
        self.val_epoch_losses = []
        self.val_accuracies = []
        self.alignment_stats = []  # Track alignment per round
        self.current_phase = "train"  # "train" or "test"
        self.current_epoch = 0
        self.train_batch_counter = 0
        self.best_test_acc = 0.0
        self.test_batch_counter = 0
        self.test_correct = 0
        self.test_total = 0
        # self.results = None
        self.wandb = setup_wandb(config)
        # if config.resume_from:
        #     self._load_checkpoint(config.resume_from)

    # def initialize_parameters(self, client_manager):
    # 	return ndarrays_to_parameters([])

    def _determine_phase(self, server_round: int) -> Tuple[str, int]:
        """
        Determine current phase (train/test) based on round number.
        
        Returns:
            (phase, round_in_epoch): "train" or "test", and position within epoch
        """
        
        if self.mode == "train":
            return "train", server_round % self.num_train_batches
        
        elif self.mode == "test":
            return "test", server_round % self.num_test_batches
        
        elif self.mode == "train-val":
            # Calculate position within epoch
            # batches_per_epoch = self.num_train_batches + self.num_test_batches
            round_in_epoch = (server_round - 1) % self.batches_per_epoch
            
            if round_in_epoch < self.num_train_batches:
                return "train", round_in_epoch
            else:
                return "test", round_in_epoch - self.num_train_batches
        
        return "train", 0


    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, Dict]]:
        """Configure training round based on mode and phase."""

        # could use same formual to calculate num batches of test will be 0 in train mode and vice versa
        if self.mode == "train":
            self.current_epoch = server_round // (self.num_train_batches) + 1
        elif self.mode == "train-val":
            self.current_epoch = server_round // (self.num_train_batches + self.num_test_batches) + 1
        elif self.mode == "test":
            self.current_epoch = 1

        
        clients = client_manager.sample(num_clients=self.num_clients)
        
        # Determine phase based on mode and round number
        phase, round_in_epoch = self._determine_phase(server_round)
        
        # Check if transitioning between phases (for logging)
        if phase != self.current_phase:
            self.current_phase = phase
            
            if phase == "test":
                # Transitioning to test phase
                self.test_batch_counter = 0
                self.test_correct = 0
                self.test_total = 0
                self.val_loss = 0
                
                if self.mode == "train-val":
                    print(f"\n{'='*60}")
                    print(f"Epoch {self.current_epoch} training complete - Starting Validation")
                    print(self.results)
                    print(f"{'='*60}\n")
            
            elif phase == "train":
                # Transitioning back to train phase

                if self.mode == "train-val":
                    print(f"\n{'='*60}")
                    print(f"Validation complete - Resuming training")
                    print(f"{'='*60}\n")
        
        is_testing = (phase == "test")
        
        config = {
            "server_round": server_round,
            "is_testing": is_testing,
            "phase": phase,
            "round_in_epoch": round_in_epoch
        }
        
        return [(client, config) for client in clients]


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        self.results = results

        if not results:
            return None, {}

        # Determine current phase (Redoing it even after configure fit for safety)
        phase, _ = self._determine_phase(server_round)
        is_testing = (phase == "test")

        # Collect data from all clients (ordered by v_split_id)
        client_data = OrderedDict()
        for client_proxy, fit_res in results:
            v_split_id = int(fit_res.metrics["v_split_id"])
            params = parameters_to_ndarrays(fit_res.parameters)
            
            client_data[v_split_id] = {
                "embeddings": torch.from_numpy(params[0]).float().to(self.device),
                "entity_ids": params[1],
                "labels": params[2]
            }
        
        # Get minimum batch size (may need to add check for asserting that embeddings from all clients should be of same dim)
        batch_sizes = [len(data["embeddings"]) for data in client_data.values()]
        min_batch_size = min(batch_sizes)
        
        if min_batch_size == 0:
            print(f"Empty batch in round {server_round}")
            return None, {"loss": 0.0}
        
        #================================================================================================
        # ALIGNMENT VERIFICATION , LATER CAN REMOVE
        # Check how many samples have aligned entity_ids and labels
        aligned_count = 0
        sample_info = []
        
        for i in range(min_batch_size):
            # Get entity_ids and labels for sample i across all clients
            entity_ids_at_i = []
            labels_at_i = []
            
            for v_split_id in sorted(client_data.keys()):
                data = client_data[v_split_id]
                entity_ids_at_i.append(data["entity_ids"][i])
                labels_at_i.append(data["labels"][i])
            
            # Check if all clients have same entity_id AND same label
            all_same_entity = len(set(entity_ids_at_i)) == 1
            all_same_label = len(set(labels_at_i)) == 1
            
            is_aligned = all_same_entity and all_same_label
            
            if is_aligned:
                aligned_count += 1
            
            sample_info.append({
                "position": i,
                "entity_ids": entity_ids_at_i,
                "labels": labels_at_i,
                "aligned": is_aligned
            })
        
        alignment_ratio = aligned_count / min_batch_size
        self.alignment_stats.append(alignment_ratio)
        
        print(f"Round {server_round}: Alignment check: {aligned_count}/{min_batch_size} "
              f"samples aligned ({alignment_ratio:.2%})")
        
        # Optional: Log first few samples
        if server_round % 100 == 0:  
            print(f"  Sample details (first 5):")
            for info in sample_info[:5]:
                print(f"    Pos {info['position']}: "
                      f"entity_ids={info['entity_ids']}, "
                      f"labels={info['labels']}, "
                      f"aligned={info['aligned']}")

        #==================================================================================================
        
        # Concatenate embeddings directly by spatial order
        embeddings_list = []
        labels_list = []
        for v_split_id in sorted(client_data.keys()):
            data = client_data[v_split_id]
            embeddings_list.append(data["embeddings"][:min_batch_size])
            labels_list.append(data["labels"][:min_batch_size])
        
        # Concatenate embeddings: [batch_size, total_embedding_dim]
        batch_embeddings = torch.cat(embeddings_list, dim=1)
        # Convert labels to one-hot, average, convert to mixed targets
        labels_array = np.stack(labels_list, axis=0)  # [num_clients, batch_size]
        
        # Convert to one-hot encoding
        one_hot_labels = np.zeros((self.num_clients, min_batch_size, self.num_classes))
        for client_idx in range(self.num_clients):
            for sample_idx in range(min_batch_size):
                label = int(labels_array[client_idx, sample_idx])
                one_hot_labels[client_idx, sample_idx, label] = 1.0
        
        # Average across clients: [batch_size, num_classes]
        mixed_labels = np.mean(one_hot_labels, axis=0)
        mixed_labels = torch.from_numpy(mixed_labels).float().to(self.device)
        
        # ================= TRAINING PHASE ==============================

        if not is_testing:
            self.server_model.train()
            self.train_batch_counter += 1
            self.optimizer.zero_grad()
            batch_embeddings.requires_grad = True
            # Forward pass
            outputs = self.server_model(batch_embeddings)  # [batch_size, num_classes]
            
            if self.loss_method == "kl":
                log_outputs = F.log_softmax(outputs, dim=1)
                loss = self.loss(log_outputs, mixed_labels)
            elif self.loss_method == "ce":
                loss = self.loss(outputs, mixed_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            embedding_grad = batch_embeddings.grad  # [batch_size, total_embedding_dim]
            # Split gradients back to clients by spatial order
            client_grads = OrderedDict()
            for v_split_id in sorted(client_data.keys()):
                # Extract gradient portion for this client
                start = v_split_id * self.embedding_dim
                end = start + self.embedding_dim
                grad_for_client = embedding_grad[:, start:end]  # [batch_size, embedding_dim]
                
                # Pad with zeros if client had more samples than min_batch_size
                actual_batch_size = len(client_data[v_split_id]["embeddings"])
                if actual_batch_size > min_batch_size:
                    print("padding is done")
                    padding = torch.zeros(
                        (actual_batch_size - min_batch_size, self.embedding_dim)
                    ).to(self.device)
                    grad_for_client = torch.cat([grad_for_client, padding], dim=0)
                
                client_grads[v_split_id] = grad_for_client.cpu().numpy()
            

            # epoch = server_round // (self.num_train_batches + self.num_test_batches) + 1
            # self.round_losses.append(loss.item())

            self.train_loss += loss.item()

            if self.train_batch_counter >= self.num_train_batches:
                self.final_train_loss = self.train_loss / self.num_train_batches
                self.train_epoch_losses.append(self.final_train_loss)
                if self.wandb and server_round % self.config.log_interval == 0:
                    self.wandb.log({
                        "train/loss": self.final_train_loss,
                        "train/alignment_ratio": alignment_ratio,
                        "epoch": self.current_epoch
                    })

                print(f"Epoch {self.current_epoch}: Loss={self.final_train_loss:.4f}, "
                    f"Batch={min_batch_size}, "
                    f"Alignment={alignment_ratio:.2%}\n")

                self.train_batch_counter = 0
                self.train_loss = 0

            # Package gradients
            parameters_grad = [
                ndarrays_to_parameters([client_grads[vid]])
                for vid in sorted(client_grads.keys())
            ]
            
            # Metrics
            metrics = {
                "batch_size": min_batch_size
            }

            if server_round == self.config.total_rounds:
                print(f"\n{'='*60}")
                print(f"All rounds complete - Finalizing...")
                print(f"{'='*60}\n")
                self.finalize()

            return parameters_grad[0] if parameters_grad else None, metrics

    
        # ========== TESTING PHASE ==========
        else:
            self.server_model.eval()
            self.test_batch_counter += 1
            
            with torch.no_grad():
                outputs = self.server_model(batch_embeddings)

                if self.loss_method == "kl":
                    log_outputs = F.log_softmax(outputs, dim=1)
                    loss = self.loss(log_outputs, mixed_labels)
                elif self.loss_method == "ce":
                    loss = self.loss(outputs, mixed_labels)

                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(mixed_labels, dim=1)
                batch_correct = (predicted == labels).sum().item()
                self.test_correct += batch_correct
                self.test_total += len(mixed_labels)
                self.val_loss += loss.item()
            
            # Check if test epoch complete
            if self.test_batch_counter >= self.num_test_batches:
                # Test epoch complete
                final_test_acc = self.test_correct / self.test_total
                final_test_loss = self.val_loss / self.num_test_batches
                self.val_accuracies.append(final_test_acc)
                self.val_epoch_losses.append(final_test_loss)
                
                print(f"\n{'='*60}")
                print(f"Test Epoch {self.current_epoch} Complete")
                print(f"Test Loss: {final_test_loss:.4f}")
                print(f"Test Accuracy: {final_test_acc:.4f} ({self.test_correct}/{self.test_total})")
                print(f"{'='*60}\n")
                
                # Log to wandb
                if self.wandb:
                    self.wandb.log({
                        "test/accuracy": final_test_acc,
                        "test/loss": final_test_loss,
                        "test/samples": self.test_total,
                        "epoch": self.current_epoch
                    })
                
                # Save checkpoint
                # self._save_checkpoint(server_round, final_test_acc)
            
            metrics = {
                "phase": "test",
                "test_batch": self.test_batch_counter
            }
            
            if server_round == self.config.total_rounds:
                print(f"\n{'='*60}")
                print(f"All rounds complete - Finalizing...")
                print(f"{'='*60}\n")
                self.finalize()

            return None, metrics


    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        """Configure evaluation - always return clients for backward/processing."""
        
        clients = client_manager.sample(num_clients=self.num_clients)
        
        phase, _ = self._determine_phase(server_round)
        is_testing = (phase == "test")
        
        config = {
            "server_round": server_round,
            "is_testing": is_testing,
            "phase": phase
        }
        
        return [(client, config) for client in clients]
    

    # def aggregate_evaluate(self, server_round, results, failures):
    #     """Aggregate evaluation - minimal implementation."""
    #     return None, {}

    # def evaluate(self, server_round: int, parameters: Parameters):
    #     """Server-side evaluation - minimal implementation."""
    #     return None


    def finalize(self):
        """Save final metrics and plots after training completes."""

        metrics = {
            'train_losses' : self.train_epoch_losses,
            'val_losses' : self.val_epoch_losses,
            'test_accuracies': self.val_accuracies,
            'alignment_stats': self.alignment_stats,
            'best_test_accuracy': self.best_test_acc,
            'config': {
                'overlap_ratio': self.config.overlap_ratio,
                'num_clients': self.num_clients,
                'num_epochs': self.config.num_epochs,
            }
        }
        
        save_metrics(metrics, self.config)
        
        # Plot curves
        plot_training_curves(
            self.train_epoch_losses,
            self.val_epoch_losses,
            self.val_accuracies,
            self.config
        )
        
        # Close wandb
        if self.wandb:
            self.wandb.finish()
            print("Weights & Biases run finished")
        
        print(f"\nBest Test Accuracy: {self.best_test_acc:.4f}")
        print(f"Results saved in: {self.config.metrics_path}\n")


# ===============================================================================================================
    #     def _load_checkpoint(self, checkpoint_path: str):
    #     """Load checkpoint and resume training."""
        
    #     checkpoint = load_checkpoint(
    #         checkpoint_path,
    #         self.server_model,
    #         self.optimizer,
    #         self.device
    #     )
        
    #     if checkpoint:
    #         self.current_epoch = checkpoint['epoch']
    #         self.round_losses = checkpoint.get('metrics', {}).get('round_losses', [])
    #         self.round_accuracies = checkpoint.get('metrics', {}).get('round_accuracies', [])
    #         self.test_accuracies = checkpoint.get('metrics', {}).get('test_accuracies', [])
    #         self.best_test_acc = max(self.test_accuracies) if self.test_accuracies else 0.0
    #         print(f"✓ Resumed from epoch {self.current_epoch}, best test acc: {self.best_test_acc:.4f}")


    # # ADD checkpoint saving method:

    # def _save_checkpoint(self, server_round: int, current_test_acc: float):
    #     """Save checkpoint at epoch boundaries."""
    #     from vfedentity.utils import save_checkpoint
        
    #     epoch = server_round // self.evaluate_every_n_rounds
        
    #     # Skip if not at save interval
    #     if epoch % self.config.save_every_n_epochs != 0:
    #         return
        
    #     metrics = {
    #         'round_losses': self.round_losses,
    #         'round_accuracies': self.round_accuracies,
    #         'test_accuracies': self.test_accuracies,
    #         'alignment_stats': self.alignment_stats,
    #     }
        
    #     is_best = current_test_acc > self.best_test_acc
    #     if is_best:
    #         self.best_test_acc = current_test_acc
        
    #     save_checkpoint(
    #         self.server_model,
    #         self.optimizer,
    #         epoch,
    #         server_round,
    #         metrics,
    #         self.config,
    #         self.wandb,
    #         is_best=is_best
    #     )
