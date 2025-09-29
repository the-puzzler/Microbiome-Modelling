"""
train_funcs.py - Training and evaluation functions for EBM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm


def cross_entropy_loss_microbiome(
    model: nn.Module,
    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Cross-entropy loss for discriminating between correct embeddings and noise embeddings.
    Class-balanced by averaging positive and negative class losses separately.
    
    Args:
        model: The model to compute outputs
        batch: Dict containing:
            - 'embeddings_type1': (batch_size, seq_len1, input_dim_type1)
            - 'embeddings_type2': (batch_size, seq_len2, input_dim_type2)
            - 'mask': (batch_size, total_seq_len) - which positions are valid (not padding)
            - 'type_indicators': (batch_size, total_seq_len) - 0 for type1, 1 for type2
            - 'correct_mask': (batch_size, total_seq_len) - which positions are correct embeddings (vs noise)
        
    Returns:
        Class-balanced cross-entropy loss value
    """
    # Get model outputs (assuming model outputs logits for binary classification)
    outputs = model(batch)  # (batch_size, total_seq_len)
    
    mask = batch['mask']  # (batch_size, total_seq_len) - valid positions
    correct_mask = batch['correct_mask']  # (batch_size, total_seq_len) - correct vs noise
    
    # Create binary labels: 1 for correct embeddings, 0 for noise embeddings
    labels = correct_mask.float()  # (batch_size, total_seq_len)
    
    # Only compute loss on valid (non-padded) positions
    valid_outputs = outputs[mask]  # (num_valid_positions,)
    valid_labels = labels[mask]    # (num_valid_positions,)
    
    # Separate positive and negative examples
    positive_mask = valid_labels == 1.0
    negative_mask = valid_labels == 0.0
    
    # Compute loss for each class separately
    positive_outputs = valid_outputs[positive_mask]
    negative_outputs = valid_outputs[negative_mask]
    positive_labels = valid_labels[positive_mask]
    negative_labels = valid_labels[negative_mask]
    
    # Compute average loss for each class
    positive_loss = F.binary_cross_entropy_with_logits(positive_outputs, positive_labels)
    negative_loss = F.binary_cross_entropy_with_logits(negative_outputs, negative_labels)
    
    # Average the two class losses
    balanced_loss = (positive_loss + negative_loss) / 2.0
    
    return balanced_loss

# no text emebdding consideration
def cross_entropy_loss_microbiome(
    model: nn.Module,
    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Cross-entropy loss for discriminating between correct and noise embeddings.
    Only evaluates on microbiome embeddings (type1), ignores text embeddings (type2).
    Class-balanced by averaging positive and negative class losses separately.
    """
    # Get model outputs (assuming model outputs logits for binary classification)
    outputs = model(batch)  # (batch_size, total_seq_len)
    
    mask = batch['mask']  # (batch_size, total_seq_len) - valid positions
    correct_mask = batch['correct_mask']  # (batch_size, total_seq_len) - correct vs noise
    type_indicators = batch['type_indicators']  # 0 for type1 (microbiome), 1 for type2 (text)
    
    # Only consider microbiome positions (type1 = 0) that are valid
    microbiome_mask = mask & (type_indicators == 0)
    
    # Create binary labels: 1 for correct embeddings, 0 for noise embeddings
    labels = correct_mask.float()  # (batch_size, total_seq_len)
    
    # Only compute loss on valid microbiome positions
    valid_outputs = outputs[microbiome_mask]  # (num_valid_microbiome_positions,)
    valid_labels = labels[microbiome_mask]    # (num_valid_microbiome_positions,)
    
    # Separate positive and negative examples
    positive_mask = valid_labels == 1.0
    negative_mask = valid_labels == 0.0
    
    # Compute loss for each class separately
    positive_outputs = valid_outputs[positive_mask]
    negative_outputs = valid_outputs[negative_mask]
    positive_labels = valid_labels[positive_mask]
    negative_labels = valid_labels[negative_mask]
    
    # Compute average loss for each class
    positive_loss = F.binary_cross_entropy_with_logits(positive_outputs, positive_labels)
    negative_loss = F.binary_cross_entropy_with_logits(negative_outputs, negative_labels)
    
    # Average the two class losses
    balanced_loss = (positive_loss + negative_loss) / 2.0
    
    return balanced_loss


def energy_based_loss_microbiome(
    model: nn.Module,
    batch: Dict[str, torch.Tensor], 
    w: float = 1.0) -> torch.Tensor:
    """
    Loss function for discriminating between correct embeddings and noise embeddings.
    
    Args:
        model: The model to compute outputs
        batch: Dict containing:
            - 'embeddings_type1': (batch_size, seq_len1, input_dim_type1)
            - 'embeddings_type2': (batch_size, seq_len2, input_dim_type2)
            - 'mask': (batch_size, total_seq_len) - which positions are valid (not padding)
            - 'type_indicators': (batch_size, total_seq_len) - 0 for type1, 1 for type2
            - 'correct_mask': (batch_size, total_seq_len) - which positions are correct embeddings (vs noise)
        w: regularization parameter
        
    Returns:
        Loss value: log(w + exp(avg(correct_outputs) - avg(noise_outputs)))
    """
    # Get model outputs
    outputs = model(batch)  # (batch_size, total_seq_len)
    
    mask = batch['mask']  # (batch_size, total_seq_len) - valid positions
    correct_mask = batch['correct_mask']  # (batch_size, total_seq_len) - correct vs noise
    
    # Create masks for correct and noise embeddings (only considering valid positions)
    correct_positions = mask & correct_mask  # (batch_size, total_seq_len)
    noise_positions = mask & (~correct_mask)  # (batch_size, total_seq_len)
    
    # Calculate averages for correct and noise embeddings
    correct_outputs = outputs * correct_positions.float()  # Zero out non-correct positions
    noise_outputs = outputs * noise_positions.float()      # Zero out non-noise positions
    
    # Sum and count for averaging
    correct_sum = torch.sum(correct_outputs, dim=1)  # (batch_size,)
    correct_count = torch.sum(correct_positions.float(), dim=1)  # (batch_size,)
    
    noise_sum = torch.sum(noise_outputs, dim=1)  # (batch_size,)
    noise_count = torch.sum(noise_positions.float(), dim=1)  # (batch_size,)
    
    # Calculate averages (add small epsilon to avoid division by zero)
    eps = 1e-8
    avg_correct = correct_sum / (correct_count + eps)  # (batch_size,)
    avg_noise = noise_sum / (noise_count + eps)        # (batch_size,)
    
    # Calculate energy difference: avg(correct) - avg(noise)
    energy_diff = avg_correct - avg_noise  # (batch_size,)
    
    # Compute loss: log(w + exp(energy_diff))
    # Use logsumexp for numerical stability: log(w + exp(x)) = logsumexp([log(w), x])
    log_w = torch.log(torch.tensor(w, dtype=torch.float32, device=energy_diff.device))
    stable_log_terms = torch.logsumexp(
        torch.stack([log_w.expand_as(energy_diff), energy_diff], dim=0), 
        dim=0
    )  # (batch_size,)
    
    # Take mean across batch
    loss = torch.mean(stable_log_terms)
    
    return loss




def eval_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_params: Dict[str, float]) -> Tuple[float, torch.Tensor]:
    """
    Single evaluation step for microbiome data
    """
    model.eval()
    
    with torch.no_grad():
        # Compute loss
        # loss = energy_based_loss_microbiome(model, batch, w=loss_params.get('w', 1.0))
        loss = cross_entropy_loss_microbiome(model, batch)
        
        # Get predictions (energy scores for all embeddings)
        predictions = model(batch)  # (batch_size, total_seq_len)
    
    return loss.item(), predictions


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_params: Dict[str, float],
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    epoch: int = 0,
    save_every: int = 35_000,
    accumulation_steps: int = 32,
    log_every: int = 10_000) -> float:
    """
    Train for one epoch with gradient accumulation, periodic checkpointing, and batch metric logging
    """
    import os
    import csv
    from tqdm import tqdm
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup CSV logging
    log_file = os.path.join(save_dir, f'training_log_epoch_{epoch}.csv')
    
    # Storage for metrics over log_every steps
    metrics_buffer = []
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0
    meta_batch_count = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    # Initialize gradients
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        # loss = energy_based_loss_microbiome(model, batch, w=loss_params.get('w', 1.0))
        loss = cross_entropy_loss_microbiome(model, batch)
        # Add this check:
        if torch.isnan(loss):
            print(f"NaN loss at batch {batch_idx}, skipping update...")
            continue  # Skip this entire batch
        # Scale loss by accumulation steps to maintain average
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        # Accumulate loss for logging (use original unscaled loss)
        accumulated_loss += loss.item()
        total_loss += loss.item()
        num_batches += 1
        
        # Store metrics for this batch
        current_overall_loss = total_loss / num_batches
        current_meta_loss = accumulated_loss / ((batch_idx % accumulation_steps) + 1)
        
        metrics_buffer.append({
            'epoch': epoch,
            'batch_num': batch_idx + 1,
            'overall_loss': current_overall_loss,
            'current_loss': loss.item(),
            'meta_loss': current_meta_loss,
            'meta_batches': meta_batch_count,
            'effective_batch_size': accumulation_steps * batch['embeddings_type1'].shape[0],  # accumulation_steps * batch_size
            'is_meta_step': False
        })
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            meta_batch_count += 1
            
            # Mark this as a meta-step and update meta_loss
            avg_accumulated_loss = accumulated_loss / accumulation_steps
            # scheduler.step(avg_accumulated_loss)
            metrics_buffer[-1]['meta_loss'] = avg_accumulated_loss
            metrics_buffer[-1]['is_meta_step'] = True
            
            # Update progress bar only on meta-batch completion
            overall_avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{overall_avg_loss:.4f}',
                'meta_loss': f'{avg_accumulated_loss:.4f}',
                'meta_batches': meta_batch_count
            })
            
            # Reset accumulated loss
            accumulated_loss = 0.0
        
        # Write metrics to CSV every log_every batches
        if (batch_idx + 1) % log_every == 0:
            # Write all buffered metrics to CSV
            file_exists = os.path.exists(log_file)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write header if file is new
                if not file_exists:
                    writer.writerow([
                        'epoch', 'batch_num', 'overall_loss', 'current_loss', 
                        'meta_loss', 'meta_batches', 'effective_batch_size', 'is_meta_step'
                    ])
                
                # Write all buffered metrics
                for metrics in metrics_buffer:
                    writer.writerow([
                        metrics['epoch'],
                        metrics['batch_num'],
                        f"{metrics['overall_loss']:.6f}",
                        f"{metrics['current_loss']:.6f}",
                        f"{metrics['meta_loss']:.6f}",
                        metrics['meta_batches'],
                        metrics['effective_batch_size'],
                        metrics['is_meta_step']
                    ])
            
            print(f"\nLogged {len(metrics_buffer)} metrics to {log_file} (batches {batch_idx + 1 - log_every + 1} to {batch_idx + 1})")
            
            # Clear the buffer
            metrics_buffer = []
        
        # Save checkpoint every save_every batches (based on actual batches processed)
        if (batch_idx + 1) % save_every == 0 or batch_idx == 10000:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'batch': batch_idx + 1,
                'loss': total_loss / num_batches,
                'loss_params': loss_params,
                'accumulation_steps': accumulation_steps
            }
            
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx + 1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Handle remaining gradients if num_batches not divisible by accumulation_steps
    if num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        meta_batch_count += 1
        
        # Final update for remaining batches
        remaining_batches = num_batches % accumulation_steps
        avg_remaining_loss = accumulated_loss / remaining_batches
        overall_avg_loss = total_loss / num_batches
        progress_bar.set_postfix({
            'loss': f'{overall_avg_loss:.4f}',
            'final_meta_loss': f'{avg_remaining_loss:.4f}',
            'meta_batches': meta_batch_count
        })
        
        print(f"\nFinal gradient step with {remaining_batches} remaining batches")
    
    # Write any remaining buffered metrics
    if metrics_buffer:
        file_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    'epoch', 'batch_num', 'overall_loss', 'current_loss', 
                    'meta_loss', 'meta_batches', 'effective_batch_size', 'is_meta_step'
                ])
            
            for metrics in metrics_buffer:
                writer.writerow([
                    metrics['epoch'],
                    metrics['batch_num'],
                    f"{metrics['overall_loss']:.6f}",
                    f"{metrics['current_loss']:.6f}",
                    f"{metrics['meta_loss']:.6f}",
                    metrics['meta_batches'],
                    metrics['effective_batch_size'],
                    metrics['is_meta_step']
                ])
        
        print(f"\nLogged final {len(metrics_buffer)} metrics to {log_file}")
    
    # Save final checkpoint at end of epoch
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch': num_batches,
        'loss': total_loss / num_batches,
        'loss_params': loss_params,
        'accumulation_steps': accumulation_steps,
        'meta_batches': meta_batch_count
    }
    
    final_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_final.pt')
    torch.save(final_checkpoint, final_path)
    print(f"\nFinal epoch checkpoint saved: {final_path}")
    print(f"Complete training log saved: {log_file}")
    
    return total_loss / num_batches


def eval_epoch(
    model: nn.Module,
    dataloader,
    loss_params: Dict[str, float],
    device: str = 'cuda') -> Tuple[float, list[torch.Tensor]]:
    """
    Evaluate for one epoch
    """
    from tqdm import tqdm
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Evaluation step
            loss, predictions = eval_step(model, batch, loss_params)
            total_loss += loss
            all_predictions.append(predictions.cpu())
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches, all_predictions


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Dictionary containing model, optimizer, and training state
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, batch {checkpoint['batch']}")
    print(f"Loss at checkpoint: {checkpoint['loss']:.4f}")
    
    return checkpoint


if __name__ == "__main__":
    # Test basic functionality
    print("Training functions updated for new data format!")
    
    # Example of what the batch should look like:
    print("\nExpected batch format:")
    print("- embeddings_type1: (batch_size, seq_len1, input_dim_type1)")
    print("- embeddings_type2: (batch_size, seq_len2, input_dim_type2)")
    print("- mask: (batch_size, total_seq_len)")
    print("- type_indicators: (batch_size, total_seq_len)")
    print("- correct_mask: (batch_size, total_seq_len)")
    print("- is_original: (batch_size,)")