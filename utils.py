import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def split_image_to_list(image_np, grid_size=3):
    """
    Splits a numpy image (H, W, C) into a list of 9 numpy images.
    This version is for processing in solver.py.
    """
    h, w, _ = image_np.shape
    tile_h, tile_w = h // grid_size, w // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile = image_np[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w, :]
            tiles.append(tile)
    return tiles

def reconstruct_image(tiles, permutation, grid_size=3):
    """
    Reconstructs an image from a list of tiles and a permutation.
    
    Args:
        tiles (list): List of 9 tile images (np arrays) in their *original scrambled order*.
        permutation (list or np.array): A list of 9 indices.
            permutation[i] = p means "tile i (from tiles list) goes to grid position p".
            Example: permutation = [6, 0, 1, 8, 5, 3, 7, 2, 4]
    """
    if len(tiles) != grid_size * grid_size:
        raise ValueError("Incorrect number of tiles for grid size.")
    
    tile_h, tile_w, c = tiles[0].shape
    img_h, img_w = tile_h * grid_size, tile_w * grid_size
    
    # Create an empty canvas
    canvas = np.zeros((img_h, img_w, c), dtype=tiles[0].dtype)
    
    num_tiles = grid_size * grid_size
    
    # Place each tile onto the canvas according to the permutation
    for tile_idx in range(num_tiles):
        # Get the tile
        tile = tiles[tile_idx]
        
        # Find its correct position in the 0-8 grid
        correct_pos = permutation[tile_idx]
        
        # Convert 1D grid position to 2D (row, col)
        r = correct_pos // grid_size
        c = correct_pos % grid_size
        
        # Calculate pixel coordinates
        y_start, y_end = r * tile_h, (r + 1) * tile_h
        x_start, x_end = c * tile_w, (c + 1) * tile_w
        
        # Place the tile
        canvas[y_start:y_end, x_start:x_end, :] = tile
        
    return canvas

def visualize_solution(scrambled_img, reconstructed_img, ground_truth_img=None):
    """
    Displays the scrambled, reconstructed, and optional ground truth images.
    """
    if ground_truth_img is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(scrambled_img)
        axes[0].set_title("Scrambled Input")
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_img)
        axes[1].set_title("Reconstructed Solution")
        axes[1].axis('off')
        
        axes[2].imshow(ground_truth_img)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(scrambled_img)
        axes[0].set_title("Scrambled Input")
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_img)
        axes[1].set_title("Reconstructed Solution")
        axes[1].axis('off')
        
    plt.tight_layout()
    plt.show()

def compute_edge_similarity(tile_i, tile_j, direction, border_width=1):
    """
    Computes edge similarity (1 / (1 + SSD)) for two tiles in a given direction.
    
    Args:
        tile_i (np.array): The "left" or "top" tile.
        tile_j (np.array): The "right" or "bottom" tile.
        direction (str): 'right' (i is left-of j) or 'bottom' (i is top-of j).
        border_width (int): How many pixels deep the border is.
    """
    try:
        if direction == 'right':
            # Compare right edge of i with left edge of j
            edge_i = tile_i[:, -border_width:, :].astype(np.float32)
            edge_j = tile_j[:, :border_width, :].astype(np.float32)
        elif direction == 'bottom':
            # Compare bottom edge of i with top edge of j
            edge_i = tile_i[-border_width:, :, :].astype(np.float32)
            edge_j = tile_j[:border_width, :, :].astype(np.float32)
        else:
            return 0

        # Sum of Squared Differences (SSD)
        ssd = np.sum((edge_i - edge_j) ** 2)
        
        # Normalize SSD (e.g., by number of pixels)
        ssd = ssd / (edge_i.size + 1e-6)
        
        # Convert distance to similarity
        similarity = 1.0 / (1.0 + ssd)
        return similarity

    except Exception as e:
        print(f"Error in edge similarity: {e}")
        return 0

def safe_log_softmax(logits, axis=-1):
    """
    Numerically stable log-softmax.
    """
    return logits - logsumexp(logits, axis=axis, keepdims=True)
