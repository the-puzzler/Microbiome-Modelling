#%%
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

class PerturbedMicrobiomeDataset(Dataset):
    def __init__(self, data_path, blacklist_path=None, M=5, otu_perturbation_rate=0.2, text_perturbation_rate=0.1, 
             min_otus=10, max_otus_per_sample=None, max_text_per_sample=None, 
             otu_topk_range=(1800, 10_000), text_topk_range=(50, 100)):
        """
        Args:
            data_path: Path to the pickle dataset file
            blacklist_path: Path to blacklist.txt containing sample IDs to exclude (one per line)
            M: Number of perturbed variations per sample
            otu_perturbation_rate: Probability of adding OTUs (additive only)
            text_perturbation_rate: Probability of adding text embeddings (additive only)
            min_otus: Minimum number of OTUs required per sample
            max_otus_per_sample: Maximum number of OTUs to include
            max_text_per_sample: Maximum number of text embeddings to include
            otu_topk: Number of most similar OTU embeddings to consider for adding
            text_topk: Number of most similar text embeddings to consider for adding
        """
        print("Loading integrated dataset...")
        
        # Load blacklist if provided
        blacklisted_sample_ids = set()
        if blacklist_path is not None:
            print(f"Loading blacklist from {blacklist_path}...")
            with open(blacklist_path, 'r') as f:
                blacklisted_sample_ids = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(blacklisted_sample_ids)} blacklisted sample IDs")
        
        # Load from pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract components
        self.otu_embedding_matrix = torch.FloatTensor(data['embedding_matrix'])
        self.text_embedding_matrix = torch.FloatTensor(data['text_embedding_matrix'])
        self.keyword_to_text_idx = data['keyword_to_text_idx']
        self.samples_data = data['samples_data']
        self.otu_to_matrix_idx = data['otu_to_matrix_idx']  # Extract the OTU mapping
        
        # Convert to numpy for similarity computations
        self.otu_embedding_matrix_np = self.otu_embedding_matrix.numpy()
        self.text_embedding_matrix_np = self.text_embedding_matrix.numpy()
        
        # Normalize embeddings for similarity computation
        self.otu_embedding_matrix_norm_np = self.otu_embedding_matrix_np / np.linalg.norm(
            self.otu_embedding_matrix_np, axis=1, keepdims=True)
        self.text_embedding_matrix_norm_np = self.text_embedding_matrix_np / np.linalg.norm(
            self.text_embedding_matrix_np, axis=1, keepdims=True)
        
        # Compute similarity matrices
        self._compute_otu_similarity_matrix()
        self._compute_text_similarity_matrix()
        
        # Filter samples by blacklist first, then by min_otus
        print("Filtering samples...")
        initial_count = len(self.samples_data)
        
        # Filter by blacklist
        if blacklisted_sample_ids:
            filtered_samples = []
            removed_count = 0
            
            for sample in self.samples_data:
                sample_name = sample['sample_name']  # This is the run ID
                
                if sample_name in blacklisted_sample_ids:
                    removed_count += 1
                else:
                    filtered_samples.append(sample)
            
            self.samples_data = filtered_samples
            print(f"Removed {removed_count} blacklisted samples")
        else:
            removed_count = 0
        
        # Filter by min_otus and apply max limits
        print(f"Filtering samples with >= {min_otus} OTUs...")
        filtered_samples = []
        
        for sample in self.samples_data:
            if len(sample['otu_matrix_indices']) >= min_otus:
                # Apply max filters if specified
                otu_indices = sample['otu_matrix_indices']
                text_indices = sample['text_indices']
                
                if max_otus_per_sample is not None:
                    otu_indices = otu_indices[:max_otus_per_sample]
                
                if max_text_per_sample is not None:
                    text_indices = text_indices[:max_text_per_sample]
                
                filtered_sample = {
                    'otu_matrix_indices': otu_indices,
                    'text_indices': text_indices,
                    'original_sample_idx': sample['original_sample_idx'],
                    'sample_name': sample['sample_name']
                }
                filtered_samples.append(filtered_sample)
        
        min_otu_filtered_count = len(self.samples_data) - len(filtered_samples)
        self.samples_data = filtered_samples
        
        print(f"Dataset filtering summary:")
        print(f"  Initial samples: {initial_count}")
        if blacklisted_sample_ids:
            print(f"  After blacklist filter: {initial_count - removed_count}")
        print(f"  After min OTU filter: {len(self.samples_data)}")
        print(f"  Total removed: {initial_count - len(self.samples_data)}")
        
        # Store parameters
        self.M = M
        self.otu_perturbation_rate = otu_perturbation_rate
        self.text_perturbation_rate = text_perturbation_rate
        self.min_otus = min_otus
        self.max_otus_per_sample = max_otus_per_sample
        self.max_text_per_sample = max_text_per_sample
        self.otu_topk_range = otu_topk_range
        self.text_topk_range = text_topk_range

        # Dataset properties
        self.otu_vocab_size = self.otu_embedding_matrix.shape[0]
        self.text_vocab_size = self.text_embedding_matrix.shape[0]
        self.otu_embedding_dim = self.otu_embedding_matrix.shape[1]
        self.text_embedding_dim = self.text_embedding_matrix.shape[1]
        
        print(f"Final dataset: {len(self.samples_data)} samples")
        print(f"OTU vocab: {self.otu_vocab_size}, embedding dim: {self.otu_embedding_dim}")
        print(f"Text vocab: {self.text_vocab_size}, embedding dim: {self.text_embedding_dim}")
    
    
    def _compute_otu_similarity_matrix(self):
        """Compute OTU similarity matrix"""
        vocab_size = self.otu_embedding_matrix_norm_np.shape[0]
        matrix_size_gb = (vocab_size ** 2 * 4) / (1024**3)
        
        print(f"Computing OTU similarity matrix ({vocab_size}x{vocab_size}, {matrix_size_gb:.1f}GB)...")
        
        self.otu_similarity_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        chunk_size = 2000
        with tqdm(total=vocab_size, desc="Computing OTU similarities", unit="rows") as pbar:
            for i in range(0, vocab_size, chunk_size):
                end_i = min(i + chunk_size, vocab_size)
                chunk_embeddings = self.otu_embedding_matrix_norm_np[i:end_i]
                similarities_chunk = np.dot(chunk_embeddings, self.otu_embedding_matrix_norm_np.T)
                self.otu_similarity_matrix[i:end_i, :] = similarities_chunk
                pbar.update(end_i - i)
        
        print("OTU similarity matrix computed!")
    
    def _compute_text_similarity_matrix(self):
        """Compute text similarity matrix"""
        vocab_size = self.text_embedding_matrix_norm_np.shape[0]
        matrix_size_gb = (vocab_size ** 2 * 4) / (1024**3)
        
        print(f"Computing text similarity matrix ({vocab_size}x{vocab_size}, {matrix_size_gb:.1f}GB)...")
        
        self.text_similarity_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        chunk_size = 2000
        with tqdm(total=vocab_size, desc="Computing text similarities", unit="rows") as pbar:
            for i in range(0, vocab_size, chunk_size):
                end_i = min(i + chunk_size, vocab_size)
                chunk_embeddings = self.text_embedding_matrix_norm_np[i:end_i]
                similarities_chunk = np.dot(chunk_embeddings, self.text_embedding_matrix_norm_np.T)
                self.text_similarity_matrix[i:end_i, :] = similarities_chunk
                pbar.update(end_i - i)
        
        print("Text similarity matrix computed!")
    
    def __len__(self):
        return len(self.samples_data)
    
    def get_otu_topk_candidates(self, present_indices):
        """Get top-k OTU candidates for addition"""
        # Sample k from range using exponential decay (8:1 ratio favoring lower values)
        min_k, max_k = self.otu_topk_range
        # Calculate alpha for 8:1 ratio between min and max
        alpha = np.log(8) / (max_k - min_k)
        alpha = 0 # flat for test purposes
        
        # Generate weights using exponential decay
        k_values = np.arange(min_k, max_k + 1)
        weights = np.exp(-alpha * (k_values - min_k))
        weights = weights / np.sum(weights)  # Normalize to probabilities
        
        # Sample k value using the weighted distribution
        otu_topk = np.random.choice(k_values, p=weights)
        
        present_set = set(present_indices)
        similarities_subset = self.otu_similarity_matrix[present_indices, :]
        max_similarities = np.max(similarities_subset, axis=0)
        
        # Zero out similarities for already present OTUs
        max_similarities[list(present_set)] = -1
        
        # Get candidates excluding already present ones
        valid_candidates = np.where(max_similarities > -1)[0]
        
        # Adjust k if we don't have enough candidates
        actual_k = min(otu_topk, len(valid_candidates))
        
        if actual_k > 0:
            topk_indices = np.argpartition(max_similarities[valid_candidates], -actual_k)[-actual_k:]
            return valid_candidates[topk_indices]
        else:
            return np.array([], dtype=np.int32)

    def get_text_topk_candidates(self, present_indices):
        """Get top-k text candidates for addition"""
        # Sample k from range using exponential decay (8:1 ratio favoring lower values)
        min_k, max_k = self.text_topk_range
        # Calculate alpha for 8:1 ratio between min and max
        alpha = np.log(8) / (max_k - min_k)
        alpha = 0 # flat for test purposes
        
        # Generate weights using exponential decay
        k_values = np.arange(min_k, max_k + 1)
        weights = np.exp(-alpha * (k_values - min_k))
        weights = weights / np.sum(weights)  # Normalize to probabilities
        
        # Sample k value using the weighted distribution
        text_topk = np.random.choice(k_values, p=weights)
        
        if len(present_indices) == 0:
            # If no text embeddings present, return random candidates
            return np.random.choice(self.text_vocab_size, size=min(text_topk, self.text_vocab_size), replace=False)
        
        present_set = set(present_indices)
        similarities_subset = self.text_similarity_matrix[present_indices, :]
        max_similarities = np.max(similarities_subset, axis=0)
        
        # Zero out similarities for already present text embeddings
        max_similarities[list(present_set)] = -1
        
        # Get candidates excluding already present ones
        valid_candidates = np.where(max_similarities > -1)[0]
        
        # Adjust k if we don't have enough candidates
        actual_k = min(text_topk, len(valid_candidates))
        
        if actual_k > 0:
            topk_indices = np.argpartition(max_similarities[valid_candidates], -actual_k)[-actual_k:]
            return valid_candidates[topk_indices]
        else:
            return np.array([], dtype=np.int32)
    
    def additive_perturbations(self, present_indices, topk_candidates, perturbation_rate, vocab_size):
        """
        Generate M additive perturbations (only adding, no dropping)
        """
        num_present = len(present_indices)
        expected_to_add = num_present * perturbation_rate
        
        all_variations = []
        all_added_masks = []
        
        if len(topk_candidates) > 0 and expected_to_add > 0:
            # Use Poisson distribution for number to add
            n_to_add_all = np.random.poisson(expected_to_add, size=self.M)
            
            # Clip to available candidates
            n_to_add_all = np.clip(n_to_add_all, 0, len(topk_candidates))
            
            # Generate random choices for each perturbation separately
            for m in range(self.M):
                # Start with original indices
                new_present = present_indices.copy()
                
                # Track which embeddings were added
                added_mask = np.zeros(len(present_indices), dtype=bool)
                
                # Add new indices
                n_to_add = n_to_add_all[m]
                if n_to_add > 0:
                    # Generate choices for this specific perturbation
                    added_indices = np.random.choice(
                        topk_candidates, 
                        size=n_to_add, 
                        replace=False
                    )
                    new_present = np.concatenate([new_present, added_indices])
                    
                    # Extend mask for newly added embeddings
                    added_mask = np.concatenate([added_mask, np.ones(n_to_add, dtype=bool)])
                
                all_variations.append(new_present)
                all_added_masks.append(added_mask)
        else:
            # No candidates to add or expected_to_add is 0
            for m in range(self.M):
                all_variations.append(present_indices.copy())
                all_added_masks.append(np.zeros(len(present_indices), dtype=bool))
        
        return all_variations, all_added_masks
    
    def additive_and_subtractive_perturbations(self, present_indices, topk_candidates, perturbation_rate, vocab_size):
        """
        Generate M perturbations with both additions and removals using single rate
        Approximately equal numbers of additions and deletions
        """
        num_present = len(present_indices)
        expected_to_add = num_present * perturbation_rate
        expected_to_remove = num_present * perturbation_rate  # Same rate for both
        
        all_variations = []
        all_added_masks = []
        
        for m in range(self.M):
            # Start with original indices
            current_present = present_indices.copy()
            
            # Step 1: Remove some existing embeddings
            n_to_remove = np.random.poisson(expected_to_remove)
            n_to_remove = min(n_to_remove, len(current_present) - 1)  # Keep at least 1
            
            if n_to_remove > 0:
                remove_indices = np.random.choice(len(current_present), size=n_to_remove, replace=False)
                current_present = np.delete(current_present, remove_indices)
            
            # Step 2: Add new embeddings
            added_mask = np.zeros(len(current_present), dtype=bool)
            
            if len(topk_candidates) > 0 and expected_to_add > 0:
                n_to_add = np.random.poisson(expected_to_add)
                n_to_add = min(n_to_add, len(topk_candidates))
                
                if n_to_add > 0:
                    added_indices = np.random.choice(topk_candidates, size=n_to_add, replace=False)
                    current_present = np.concatenate([current_present, added_indices])
                    added_mask = np.concatenate([added_mask, np.ones(n_to_add, dtype=bool)])
            
            all_variations.append(current_present)
            all_added_masks.append(added_mask)
        
        return all_variations, all_added_masks
    
    
    def __getitem__(self, idx):
        """
        Returns:
            otu_embeddings: tensor of shape (M+1, max_otus, otu_embedding_dim)
            otu_masks: tensor of shape (M+1, max_otus)
            otu_added_masks: tensor of shape (M+1, max_otus) - True for added embeddings
            text_embeddings: tensor of shape (M+1, max_text, text_embedding_dim) 
            text_masks: tensor of shape (M+1, max_text)
            text_added_masks: tensor of shape (M+1, max_text) - True for added embeddings
        """
        sample = self.samples_data[idx]
        original_otu_indices = sample['otu_matrix_indices']
        original_text_indices = sample['text_indices']
        
        # Get top-k candidates for both modalities
        otu_topk_candidates = self.get_otu_topk_candidates(original_otu_indices)
        
        # Generate OTU perturbations (additive only)
        otu_perturbations, otu_added_masks = self.additive_perturbations(
            original_otu_indices, otu_topk_candidates, 
            self.otu_perturbation_rate, self.otu_vocab_size
        )
        
        # Handle text perturbations only if text embeddings exist, no text perturbations with text conditoining mode
        if len(original_text_indices) > 0:
            # text_topk_candidates = self.get_text_topk_candidates(original_text_indices)
            # text_perturbations, text_added_masks = self.additive_perturbations(
            #     original_text_indices, text_topk_candidates,
            #     self.text_perturbation_rate, self.text_vocab_size
            # )
            
            # NEW: Keep original text for all M perturbations (no noise)
            text_perturbations = [original_text_indices.copy() for _ in range(self.M)]
            text_added_masks = [np.zeros(len(original_text_indices), dtype=bool) for _ in range(self.M)]
        else:
            # No text embeddings, create empty variations
            text_perturbations = [np.array([], dtype=np.int32) for _ in range(self.M)]
            text_added_masks = [np.array([], dtype=bool) for _ in range(self.M)]
        
        # Combine original + perturbations for both modalities
        all_otu_variations = [original_otu_indices] + otu_perturbations
        all_text_variations = [original_text_indices] + text_perturbations
        
        # Original has no added embeddings (all False)
        original_otu_added = np.zeros(len(original_otu_indices), dtype=bool)
        original_text_added = np.zeros(len(original_text_indices), dtype=bool)
        
        all_otu_added_masks = [original_otu_added] + otu_added_masks
        all_text_added_masks = [original_text_added] + text_added_masks
        
        # Convert to tensors and pad - OTUs
        max_otus_in_sample = max(len(var) for var in all_otu_variations)
        if self.max_otus_per_sample is not None:
            max_otus_in_sample = min(max_otus_in_sample, self.max_otus_per_sample)
        
        otu_embeddings = torch.zeros(self.M + 1, max_otus_in_sample, self.otu_embedding_dim)
        otu_masks = torch.zeros(self.M + 1, max_otus_in_sample, dtype=torch.bool)
        otu_added_masks_tensor = torch.zeros(self.M + 1, max_otus_in_sample, dtype=torch.bool)
        
        # Convert to tensors and pad - Text
        max_text_in_sample = max(len(var) for var in all_text_variations) if any(len(var) > 0 for var in all_text_variations) else 0
        if self.max_text_per_sample is not None and max_text_in_sample > 0:
            max_text_in_sample = min(max_text_in_sample, self.max_text_per_sample)
        
        text_embeddings = torch.zeros(self.M + 1, max_text_in_sample, self.text_embedding_dim)
        text_masks = torch.zeros(self.M + 1, max_text_in_sample, dtype=torch.bool)
        text_added_masks_tensor = torch.zeros(self.M + 1, max_text_in_sample, dtype=torch.bool)
        
        # Fill OTU tensors
        for m, (variation, added_mask) in enumerate(zip(all_otu_variations, all_otu_added_masks)):
            if self.max_otus_per_sample is not None:
                variation = variation[:self.max_otus_per_sample]
                added_mask = added_mask[:self.max_otus_per_sample]
            
            n_otus = len(variation)
            if n_otus > 0:
                variation_embeddings = self.otu_embedding_matrix[variation]
                otu_embeddings[m, :n_otus] = variation_embeddings
                otu_masks[m, :n_otus] = True
                otu_added_masks_tensor[m, :n_otus] = torch.from_numpy(added_mask)
        
        # Fill text tensors
        for m, (variation, added_mask) in enumerate(zip(all_text_variations, all_text_added_masks)):
            if self.max_text_per_sample is not None:
                variation = variation[:self.max_text_per_sample]
                added_mask = added_mask[:self.max_text_per_sample]
            
            n_text = len(variation)
            if n_text > 0:
                variation_embeddings = self.text_embedding_matrix[variation]
                text_embeddings[m, :n_text] = variation_embeddings
                text_masks[m, :n_text] = True
                text_added_masks_tensor[m, :n_text] = torch.from_numpy(added_mask)
        
        # return {
        #     'otu_embeddings': otu_embeddings,              # (M+1, max_otus, otu_embedding_dim)
        #     'otu_masks': otu_masks,                       # (M+1, max_otus)
        #     'otu_added_masks': otu_added_masks_tensor,    # (M+1, max_otus) - True for added
        #     'text_embeddings': text_embeddings,            # (M+1, max_text, text_embedding_dim)
        #     'text_masks': text_masks,                     # (M+1, max_text)
        #     'text_added_masks': text_added_masks_tensor,  # (M+1, max_text) - True for added
        #     'original_sample_idx': sample['original_sample_idx']
        # }
        # Hack: Only return perturbations (skip original at index 0) trying to make challange harder.
        return {
            'otu_embeddings': otu_embeddings[1:],              # Skip index 0
            'otu_masks': otu_masks[1:],                       
            'otu_added_masks': otu_added_masks_tensor[1:],    
            'text_embeddings': text_embeddings[1:],            
            'text_masks': text_masks[1:],                     
            'text_added_masks': text_added_masks_tensor[1:],  
            'original_sample_idx': sample['original_sample_idx']
        }


def collate_microbiome_batch(batch):
    """
    Collate function that flattens M+1 perturbations into batch dimension
    and creates correct_mask for noise discrimination
    """
    all_samples = []
    
    # Flatten M+1 variations into individual samples
    for sample in batch:
        M_plus_1 = sample['otu_embeddings'].shape[0]
        
        for m in range(M_plus_1):
            # Extract the m-th variation
            otu_emb = sample['otu_embeddings'][m]              # (max_otus, otu_dim)
            otu_mask = sample['otu_masks'][m]                  # (max_otus,)
            otu_added = sample['otu_added_masks'][m]           # (max_otus,) - True for added
            text_emb = sample['text_embeddings'][m]            # (max_text, text_dim)  
            text_mask = sample['text_masks'][m]                # (max_text,)
            text_added = sample['text_added_masks'][m]         # (max_text,) - True for added
            
            all_samples.append({
                'otu_embeddings': otu_emb,
                'otu_masks': otu_mask,
                'otu_added': otu_added,
                'text_embeddings': text_emb,
                'text_masks': text_mask,
                'text_added': text_added,
                'is_original': m == 0
            })
    
    # Now we have N×(M+1) individual samples
    effective_batch_size = len(all_samples)
    
    # Find max dimensions across all flattened samples
    max_otus = max(s['otu_embeddings'].shape[0] for s in all_samples)
    max_text = max(s['text_embeddings'].shape[0] for s in all_samples)
    
    max_text = max(max_text, 1)
    
    # Get embedding dimensions
    otu_dim = all_samples[0]['otu_embeddings'].shape[1]
    text_dim = 1536 #all_samples[0]['text_embeddings'].shape[1] if max_text > 0 else 0
    
    # Initialize tensors for model input format
    embeddings_type1 = torch.zeros(effective_batch_size, max_otus, otu_dim)
    embeddings_type2 = torch.zeros(effective_batch_size, max_text, text_dim)
    
    # Combined sequence length per sample
    total_seq_len = max_otus + max_text
    combined_mask = torch.zeros(effective_batch_size, total_seq_len, dtype=torch.bool)
    type_indicators = torch.zeros(effective_batch_size, total_seq_len, dtype=torch.long)
    correct_mask = torch.zeros(effective_batch_size, total_seq_len, dtype=torch.bool)  # NEW: correct vs noise
    
    # Fill tensors
    for i, sample in enumerate(all_samples):
        # OTU embeddings (type 1)
        n_otus = sample['otu_embeddings'].shape[0]
        if n_otus > 0:
            embeddings_type1[i, :n_otus] = sample['otu_embeddings']
            combined_mask[i, :n_otus] = sample['otu_masks']
            type_indicators[i, :n_otus] = 0  # Type 0 for OTU
            
            # Correct mask: True for original embeddings, False for added embeddings
            correct_mask[i, :n_otus] = ~sample['otu_added']  # Invert: False=added, True=original
        
        # Text embeddings (type 2)  
        n_text = sample['text_embeddings'].shape[0]
        if n_text > 0:
            embeddings_type2[i, :n_text] = sample['text_embeddings']
            combined_mask[i, max_otus:max_otus + n_text] = sample['text_masks']
            type_indicators[i, max_otus:max_otus + n_text] = 1  # Type 1 for text
            
            # Correct mask: True for original embeddings, False for added embeddings
            correct_mask[i, max_otus:max_otus + n_text] = ~sample['text_added']  # Invert: False=added, True=original
    
    # Create original/perturbation mask (per sample)
    is_original_mask = torch.tensor([s['is_original'] for s in all_samples], dtype=torch.bool)
    
    return {
        'embeddings_type1': embeddings_type1,          # (effective_batch, max_otus, otu_dim)
        'embeddings_type2': embeddings_type2,          # (effective_batch, max_text, text_dim)
        'mask': combined_mask,                          # (effective_batch, total_seq_len)
        'type_indicators': type_indicators,             # (effective_batch, total_seq_len)
        'correct_mask': correct_mask,                   # (effective_batch, total_seq_len) - True=original, False=added
        'is_original': is_original_mask                 # (effective_batch,) - True=original sample, False=perturbation
    }
   
   

# Example usage with filtering:
if __name__ == "__main__":
    # Create dataset with minimum 10 OTUs
    dataset = PerturbedMicrobiomeDataset(
        '/mnt/mnemo9/mpelus/matlas/EBM/individual-ebm/microbiome_dataset_simplified_with_names.pkl',
        blacklist_path='blacklist.txt'
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_microbiome_batch,
        num_workers=0
    )
    
   

#%%
