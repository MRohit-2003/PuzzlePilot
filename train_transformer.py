import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data_loader import JigsawDataset, train_transform, val_transform
from models import TransformerSolver

# --- Config ---
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "sample_train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "sample_train")
WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "transformer.pth")

BATCH_SIZE = 32 # Smaller batch for transformer
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4
NUM_POSITIONS = 9
# --------------

def main():
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data ---
    # This dataset returns (B, 9, C, H, W) tiles and (B, 9) labels
    train_dataset = JigsawDataset(CSV_FILE, IMAGE_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    val_dataset = JigsawDataset(CSV_FILE, IMAGE_DIR, transform=val_transform) # Using same data for demo
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Model ---
    model = TransformerSolver(
        backbone_name='resnet18', 
        num_positions=NUM_POSITIONS
    ).to(device)
    
    # Loss: CrossEntropy on (B*9, 9) logits and (B*9) labels
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_acc = 0.0

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        model.train()
        train_loss = 0.0
        
        for tiles, labels in tqdm(train_loader, desc="Training"):
            # tiles: (B, 9, C, H, W), labels: (B, 9)
            tiles, labels = tiles.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # logits: (B, 9_tiles, 9_positions)
            logits = model(tiles)
            
            # Reshape for CrossEntropy
            # Input: (B*9, 9_positions), Target: (B*9)
            loss = criterion(logits.view(-1, NUM_POSITIONS), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for tiles, labels in tqdm(val_loader, desc="Validation"):
                tiles, labels = tiles.to(device), labels.to(device)
                
                logits = model(tiles)
                loss = criterion(logits.view(-1, NUM_POSITIONS), labels.view(-1))
                val_loss += loss.item()
                
                # Check accuracy
                # (B*9, 9) -> (B*9)
                _, predicted = torch.max(logits.view(-1, NUM_POSITIONS).data, 1)
                total += labels.view(-1).size(0)
                correct += (predicted == labels.view(-1)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val Acc (per-tile): {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving best model to {MODEL_PATH} (Acc: {val_acc:.2f}%)")
            torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()
