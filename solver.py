import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
import itertools
from tqdm import tqdm

from utils import compute_edge_similarity, safe_log_softmax

class JigsawSolver:
    """
    Implements the full ensemble + Hungarian + refinement pipeline.
    """
    def __init__(self, classifier_model, transformer_model, pairwise_model, weights, device):
        self.models = {
            'classifier': classifier_model.to(device).eval(),
            'transformer': transformer_model.to(device).eval(),
            'pairwise': pairwise_model.to(device).eval(),
        }
        self.weights = weights
        self.device = device
        self.num_tiles = 9
        self.grid_size = 3
        # Precompute grid neighbor relationships
        self.grid_neighbors = self._get_grid_neighbors()
        # Precompute relative position names (must match PairwiseModel training)
        # 0: top, 1: bottom, 2: left, 3: right (simplified for this example)
        # 4: not-adjacent
        # This mapping *must* match get_relative_position in data_loader.py
        # 0:TL, 1:T, 2:TR, 3:L, 4:R, 5:BL, 6:B, 7:BR, 8:NA
        self.rel_map = {
            'top': 1,
            'bottom': 6,
            'left': 3,
            'right': 4,
            # We also need to check the inverse relations
            'inv_top': 6, # j is bottom of i
            'inv_bottom': 1, # j is top of i
            'inv_left': 4, # j is right of i
            'inv_right': 3, # j is left of i
        }


    @torch.no_grad()
    def _get_model_logits(self, tiles_tensor_batch):
        """
        Gets logits from classifier and transformer.
        Assumes batch size B=1.
        """
        # tiles_tensor_batch: (1, 9, C, H, W)
        
        # 1. Classifier: (1*9, C, H, W) -> (9, 9)
        tiles_flat = tiles_tensor_batch.view(-1, *tiles_tensor_batch.shape[2:])
        logits_class = self.models['classifier'](tiles_flat) # (9, 9)
        
        # 2. Transformer: (1, 9, C, H, W) -> (1, 9, 9)
        logits_trans = self.models['transformer'](tiles_tensor_batch)
        logits_trans = logits_trans.squeeze(0) # (9, 9)
        
        return logits_class.cpu().numpy(), logits_trans.cpu().numpy()

    def _fuse_logits(self, logits_class, logits_trans):
        """
        Step 3: Calibrate & combining per-tile logits
        """
        logp_class = safe_log_softmax(logits_class, axis=-1)
        logp_trans = safe_log_softmax(logits_trans, axis=-1)
        
        w_c = self.weights['w_c']
        w_t = self.weights['w_t']
        
        # Combine in log-space
        combined_logp = w_c * logp_class + w_t * logp_trans
        
        # Cost matrix is negative log-probability
        cost_matrix = -combined_logp
        return cost_matrix, combined_logp

    @torch.no_grad()
    def _compute_pairwise_scores(self, tiles_tensor_batch):
        """
        Step 5: Compute pairwise compatibility matrix
        Precomputes P(i rel j) for all i, j and all relations.
        """
        tiles_tensor = tiles_tensor_batch.squeeze(0) # (9, C, H, W)
        
        # Store log-probabilities: P(i is [direction] of j)
        pair_scores = {
            'right': np.zeros((self.num_tiles, self.num_tiles)),
            'bottom': np.zeros((self.num_tiles, self.num_tiles))
        }
        
        for i, j in itertools.permutations(range(self.num_tiles), 2):
            ti = tiles_tensor[i].unsqueeze(0) # (1, C, H, W)
            tj = tiles_tensor[j].unsqueeze(0) # (1, C, H, W)
            
            # rel_logits: (1, num_relations)
            rel_logits = self.models['pairwise'](ti, tj)
            rel_probs = softmax(rel_logits.cpu().numpy().flatten())
            
            # P(i is left-of j) maps to rel_probs[rel_map['right']]
            pair_scores['right'][i, j] = rel_probs[self.rel_map['right']]
            # P(i is top-of j) maps to rel_probs[rel_map['bottom']]
            pair_scores['bottom'][i, j] = rel_probs[self.rel_map['bottom']]
            
        return pair_scores

    def _compute_edge_scores(self, tiles_list_np):
        """
        Step 5: Compute edge compatibility matrices
        """
        edge_scores = {
            'right': np.zeros((self.num_tiles, self.num_tiles)),
            'bottom': np.zeros((self.num_tiles, self.num_tiles))
        }
        
        for i, j in itertools.permutations(range(self.num_tiles), 2):
            tile_i = tiles_list_np[i]
            tile_j = tiles_list_np[j]
            
            # i is left-of j
            edge_scores['right'][i, j] = compute_edge_similarity(tile_i, tile_j, 'right')
            # i is top-of j
            edge_scores['bottom'][i, j] = compute_edge_similarity(tile_i, tile_j, 'bottom')
            
        return edge_scores
        
    def _get_grid_neighbors(self):
        """Helper to get (pos_p, pos_q, 'direction') tuples."""
        neighbors = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                p = r * self.grid_size + c
                # p is left-of q
                if c < self.grid_size - 1:
                    q = p + 1
                    neighbors.append((p, q, 'right'))
                # p is top-of q
                if r < self.grid_size - 1:
                    q = p + 3
                    neighbors.append((p, q, 'bottom'))
        return neighbors
        
    def _calculate_global_score(self, permutation, per_tile_logp, pairwise_scores, edge_scores):
        """
        Step 6: Calculate the GlobalScore(A) for a given permutation.
        """
        # We need the inverse: pos_to_tile[p] = i
        pos_to_tile = np.empty(self.num_tiles, dtype=int)
        for i_tile, p_pos in enumerate(permutation):
            pos_to_tile[p_pos] = i_tile
            
        # 1. Per-tile score (from fused logp)
        score_per_tile = 0.0
        for i_tile, p_pos in enumerate(permutation):
            score_per_tile += per_tile_logp[i_tile, p_pos]
            
        # 2. Pairwise and Edge scores
        score_pairwise = 0.0
        score_edge = 0.0
        
        for p, q, direction in self.grid_neighbors:
            # i = tile at pos p, j = tile at pos q
            i = pos_to_tile[p]
            j = pos_to_tile[q]
            
            # Add P(i is [direction]-of j)
            score_pairwise += pairwise_scores[direction][i, j]
            score_edge += edge_scores[direction][i, j]
            
        # 3. Combine with weights
        w = self.weights
        total_score = (w['alpha'] * score_per_tile + 
                       w['beta'] * score_pairwise + 
                       w['gamma'] * score_edge)
        return total_score

    def solve(self, tiles_tensor_batch, tiles_list_np, max_iter=100):
        """
        Runs the full solving pipeline for a single puzzle (batch size 1).
        
        Args:
            tiles_tensor_batch (torch.Tensor): (1, 9, C, H, W) tensor for models.
            tiles_list_np (list): List of 9 numpy images for edge scoring.
            max_iter (int): Max iterations (full passes) for local search.
        """
        
        # --- 1. Get Model Logits ---
        logits_class, logits_trans = self._get_model_logits(tiles_tensor_batch)
        
        # --- 2. Fuse Logits ---
        # (Step 3)
        cost_matrix, combined_logp = self._fuse_logits(logits_class, logits_trans)
        
        # --- 3. Run Hungarian ---
        # (Step 4)
        # tile_indices[i] = i_tile, pos_indices[i] = p_pos
        tile_indices, pos_indices = linear_sum_assignment(cost_matrix)
        
        # current_perm[i_tile] = p_pos
        current_perm = np.empty(self.num_tiles, dtype=int)
        current_perm[tile_indices] = pos_indices
        
        # --- 4. Precompute Scores for Refinement ---
        # (Step 5)
        print("Precomputing pairwise scores...")
        pairwise_scores = self._compute_pairwise_scores(tiles_tensor_batch)
        print("Precomputing edge scores...")
        edge_scores = self._compute_edge_scores(tiles_list_np)
        
        # --- 5. Local Search / Swap Refinement ---
        # (Step 6)
        print("Starting local search refinement...")
        current_score = self._calculate_global_score(
            current_perm, combined_logp, pairwise_scores, edge_scores
        )
        print(f"Initial (Hungarian) Score: {current_score:.4f}")
        
        for iteration in range(max_iter):
            swapped = False
            # Try swapping every pair of tiles (i, j)
            for i_tile, j_tile in itertools.combinations(range(self.num_tiles), 2):
                
                # Create a new trial permutation
                new_perm = current_perm.copy()
                
                # Swap the positions of tile i and tile j
                pos_i = new_perm[i_tile]
                pos_j = new_perm[j_tile]
                new_perm[i_tile] = pos_j
                new_perm[j_tile] = pos_i
                
                # Evaluate new score
                new_score = self._calculate_global_score(
                    new_perm, combined_logp, pairwise_scores, edge_scores
                )
                
                # Greedy hill-climbing
                if new_score > current_score:
                    current_perm = new_perm
                    current_score = new_score
                    swapped = True
                    # print(f"  Swap {i_tile} <-> {j_tile} | New Score: {current_score:.4f}")
                    break # Take first improvement
            
            if swapped:
                continue # Restart outer loop
            
            # If no swaps in a full pass, we are at a local optimum
            print(f"Local search finished after {iteration+1} iterations.")
            break
        
        print(f"Final Score: {current_score:.4f}")
        return current_perm
