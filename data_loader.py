import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random

# Define standard transforms for tiles.
# NOTE: Use the *same* transform for all models.
TILE_SIZE = 224 # Example size, ResNet standard
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.RandomHorizontalFlip(), # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def split_image(image, grid_size=3):
    """Splits an image (H, W, C) into a list of 9 tile images."""
    h, w, _ = image.shape
    tile_h, tile_w = h // grid_size, w // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w, :]
            tiles.append(tile)
    return tiles

def get_relative_position(pos_i, pos_j, grid_size=3):
    """
    Calculates the relative position of j with respect to i.
    Returns an integer from 0-8, or 8 for 'not adjacent'.
    0: top-left, 1: top, 2: top-right,
    3: left, 4: right,
    5: bottom-left, 6: bottom, 7: bottom-right
    8: not-adjacent
    """
    r_i, c_i = pos_i // grid_size, pos_i % grid_size
    r_j, c_j = pos_j // grid_size, pos_j % grid_size
    
    dr, dc = r_j - r_i, c_j - c_i
    
    if abs(dr) > 1 or abs(dc) > 1:
        return 8 # Not adjacent
    
    if dr == -1 and dc == -1: return 0
    if dr == -1 and dc ==  0: return 1
    if dr == -1 and dc ==  1: return 2
    if dr ==  0 and dc == -1: return 3
    if dr ==  0 and dc ==  1: return 4
    if dr ==  1 and dc == -1: return 5
    if dr ==  1 and dc ==  0: return 6
    if dr ==  1 and dc ==  1: return 7
    
    return 8 # Should not happen if i != j


class JigsawDataset(Dataset):
    """
    Base dataset for loading images and parsing labels.
    The label "601853724" means:
    - tile 0 (at scrambled pos 0) belongs at correct pos 6
    - tile 1 (at scrambled pos 1) belongs at correct pos 0
    - ...
    """
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.grid_size = 3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_str = self.df.iloc[idx, 1]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Split image into tiles (list of 9 np arrays)
        tiles_np = split_image(image, self.grid_size)
        
        # 2. Parse the label string
        # label[i] = p means tile `i` (from scrambled grid) goes to correct position `p`
        label = torch.tensor([int(c) for c in str(label_str)], dtype=torch.long)
        
        # 3. Apply transforms
        tiles_tensor = torch.stack([self.transform(tile) for tile in tiles_np])
        
        return tiles_tensor, label

class JigsawTileDataset(Dataset):
    """
    Dataset for the TileClassifier.
    Returns: (one_tile, correct_position_label)
    """
    def __init__(self, csv_file, image_dir, transform=None):
        self.base_dataset = JigsawDataset(csv_file, image_dir, transform)
        self.num_tiles = self.base_dataset.grid_size ** 2

    def __len__(self):
        return len(self.base_dataset) * self.num_tiles

    def __getitem__(self, idx):
        # Figure out which image and which tile
        img_idx = idx // self.num_tiles
        tile_idx = idx % self.num_tiles
        
        # Get all tiles and labels for that image
        tiles_tensor, labels_tensor = self.base_dataset[img_idx]
        
        # Pick the specific tile and its corresponding correct position
        tile = tiles_tensor[tile_idx]
        label = labels_tensor[tile_idx] # This is the correct position
        
        return tile, label

class JigsawPairwiseDataset(Dataset):
    """
    Dataset for the PairwiseModel.
    Returns: (tile_i, tile_j, relative_position_label)
    """
    def __init__(self, csv_file, image_dir, transform=None):
        self.base_dataset = JigsawDataset(csv_file, image_dir, transform)
        self.num_tiles = self.base_dataset.grid_size ** 2

    def __len__(self):
        # Can be as long as you want, just samples pairs
        return len(self.base_dataset) * (self.num_tiles * (self.num_tiles - 1) // 2)

    def __getitem__(self, idx):
        # Get a random image
        img_idx = random.randint(0, len(self.base_dataset) - 1)
        tiles_tensor, labels_tensor = self.base_dataset[img_idx]
        
        # Pick two different random tile indices from the scrambled image
        i, j = random.sample(range(self.num_tiles), 2)
        
        tile_i = tiles_tensor[i]
        tile_j = tiles_tensor[j]
        
        # Find their *true* positions
        pos_i = labels_tensor[i].item()
        pos_j = labels_tensor[j].item()
        
        # Get the relative label
        relative_label = get_relative_position(pos_i, pos_j)
        
        return tile_i, tile_j, torch.tensor(relative_label, dtype=torch.long)

# Example usage:
if __name__ == "__main__":
    # Create a dummy CSV for testing
    if not os.path.exists('data'):
        os.makedirs('data/train_images')
    with open('data/train.csv', 'w') as f:
        f.write('image,label\n')
        f.write('dummy.jpg,601853724\n')
    
    # Create a dummy image
    # dummy_img = (np.random.rand(240, 240, 3) * 255).astype(np.uint8)
    # cv2.imwrite('data/train_images/dummy.jpg', dummy_img)
    # dummy_img = cv2.read('data/sample_test/1.jpg')
    
    print("--- Testing JigsawTileDataset (for Classifier) ---")
    tile_ds = JigsawTileDataset('data/train.csv', 'data/train_images', val_transform)
    tile, label = tile_ds[0] # Get tile 0 from image 0
    print(f"Tile shape: {tile.shape}, Label: {label}") # Label should be 6
    tile, label = tile_ds[1] # Get tile 1 from image 0
    print(f"Tile shape: {tile.shape}, Label: {label}") # Label should be 0

    print("\n--- Testing JigsawDataset (for Transformer) ---")
    trans_ds = JigsawDataset('data/train.csv', 'data/train_images', val_transform)
    tiles, labels = trans_ds[0]
    print(f"Tiles shape: {tiles.shape}, Labels: {labels}") # 9, C, H, W and 9,
    
    print("\n--- Testing JigsawPairwiseDataset (for Pairwise) ---")
    pair_ds = JigsawPairwiseDataset('data/train.csv', 'data/train_images', val_transform)
    ti, tj, rel_label = pair_ds[0]
    print(f"Tile i shape: {ti.shape}, Tile j shape: {tj.shape}, Rel Label: {rel_label}")
