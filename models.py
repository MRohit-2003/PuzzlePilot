import torch
import torch.nn as nn
import torchvision.models as tv_models

# Step 2: Define Model Architectures

class TileClassifier(nn.Module):
    """
    Classifier (per-tile):
    Input: single tile image
    Output: 9 logits (probability of absolute position)
    """
    def __init__(self, backbone_name='resnet18', num_positions=9, pretrained=True):
        super().__init__()
        # Load a pretrained backbone
        self.backbone = tv_models.get_model(backbone_name, weights='DEFAULT' if pretrained else None)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_positions)
        
    def forward(self, x):
        """ x: (B, C, H, W) - a batch of single tiles """
        return self.backbone(x) # (B, 9)

class PairwiseModel(nn.Module):
    """
    Pairwise model (relative relationships):
    Input: pair of tiles (i, j)
    Output: logits for relative relation (e.g., 9 classes: 8 neighbors + 'not-adjacent')
    """
    def __init__(self, backbone_name='resnet18', embedding_dim=512, num_relations=9, pretrained=True):
        super().__init__()
        # Siamese backbone
        self.backbone = tv_models.get_model(backbone_name, weights='DEFAULT' if pretrained else None)
        in_features = self.backbone.fc.in_features
        # Replace fc layer with an embedding projection
        self.backbone.fc = nn.Linear(in_features, embedding_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_relations)
        )
        
    def forward(self, tile_i, tile_j):
        """
        tile_i: (B, C, H, W)
        tile_j: (B, C, H, W)
        """
        # Get embeddings for each tile
        emb_i = self.backbone(tile_i) # (B, embedding_dim)
        emb_j = self.backbone(tile_j) # (B, embedding_dim)
        
        # Concatenate embeddings
        combined_embedding = torch.cat([emb_i, emb_j], dim=1) # (B, embedding_dim * 2)
        
        # Classify the relationship
        logits = self.classifier(combined_embedding) # (B, num_relations)
        return logits

class TransformerSolver(nn.Module):
    """
    Transformer / permutation model (global):
    Input: all 9 tiles (sequence)
    Output: per-tile position logits (9x9)
    """
    def __init__(self, backbone_name='resnet18', embedding_dim=512, nhead=8, num_layers=6, num_positions=9, pretrained=True):
        super().__init__()
        self.num_positions = num_positions
        
        # 1. Tile embedding backbone
        self.backbone = tv_models.get_model(backbone_name, weights='DEFAULT' if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)
        
        # 2. Learnable positional embedding for the 9 *slots*
        # (CLS token is not strictly necessary here, but can be added)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, embedding_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead,
            batch_first=True # Expects (B, N, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output head
        # Predicts the position (0-8) for each of the 9 input tiles
        self.fc_out = nn.Linear(embedding_dim, num_positions)
        
    def forward(self, x):
        """ x: (B, N, C, H, W) - Batch of puzzles, N=9 tiles """
        B, N, C, H, W = x.shape
        if N != self.num_positions:
            raise ValueError(f"Input must have {self.num_positions} tiles, but got {N}")
            
        # 1. Get tile embeddings: (B, N, C, H, W) -> (B*N, C, H, W)
        x = x.view(B * N, C, H, W)
        embeddings = self.backbone(x) # (B*N, embedding_dim)
        
        # Reshape to (B, N, embedding_dim)
        embeddings = embeddings.view(B, N, -1)
        
        # 2. Add positional embeddings
        embeddings = embeddings + self.pos_embedding
        
        # 3. Pass through Transformer
        transformer_out = self.transformer_encoder(embeddings) # (B, N, embedding_dim)
        
        # 4. Output logits
        logits = self.fc_out(transformer_out) # (B, N, num_positions) -> (B, 9 tiles, 9 positions)
        return logits

# Example usage:
if __name__ == "__main__":
    # 1. Classifier
    print("--- Testing TileClassifier ---")
    dummy_tile = torch.randn(4, 3, 224, 224) # 4 tiles
    classifier = TileClassifier()
    out = classifier(dummy_tile)
    print(f"Output shape: {out.shape}") # Should be (4, 9)

    # 2. Pairwise
    print("\n--- Testing PairwiseModel ---")
    dummy_tile_i = torch.randn(4, 3, 224, 224) # 4 pairs
    dummy_tile_j = torch.randn(4, 3, 224, 224)
    pairwise = PairwiseModel()
    out = pairwise(dummy_tile_i, dummy_tile_j)
    print(f"Output shape: {out.shape}") # Should be (4, 9)
    
    # 3. Transformer
    print("\n--- Testing TransformerSolver ---")
    dummy_puzzle = torch.randn(2, 9, 3, 224, 224) # 2 puzzles, 9 tiles each
    transformer = TransformerSolver()
    out = transformer(dummy_puzzle)
    print(f"Output shape: {out.shape}") # Should be (2, 9, 9)
