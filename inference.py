import torch
import cv2
import argparse
import numpy as np
import os

from models import TileClassifier, TransformerSolver, PairwiseModel
from solver import JigsawSolver
from utils import split_image_to_list, reconstruct_image, visualize_solution
from data_loader import val_transform # Use validation transform

def main():
    parser = argparse.ArgumentParser(description="Solve a 3x3 jigsaw puzzle")
    parser.add_argument('--image', type=str, required=True, help="Path to the scrambled puzzle image.")
    parser.add_argument('--weights_dir', type=str, default='weights', help="Directory with trained model weights.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Models ---
    print("Loading models...")
    try:
        classifier = TileClassifier()
        classifier.load_state_dict(torch.load(os.path.join(args.weights_dir, 'classifier.pth'), map_location=device))
        
        transformer = TransformerSolver()
        transformer.load_state_dict(torch.load(os.path.join(args.weights_dir, 'transformer.pth'), map_location=device))
        
        pairwise = PairwiseModel()
        pairwise.load_state_dict(torch.load(os.path.join(args.weights_dir, 'pairwise.pth'), map_location=device))
    except FileNotFoundError as e:
        print(f"Error: Could not load model weights. {e}")
        print("Please run the training scripts first.")
        return

    # --- 2. Load and Preprocess Image ---
    print(f"Loading image: {args.image}")
    img_np = cv2.imread(args.image)
    if img_np is None:
        print(f"Error: Could not read image file {args.image}")
        return
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # Split into 9 numpy tiles (for edge scoring and reconstruction)
    tiles_list_np = split_image_to_list(img_np)
    
    # Convert to tensor (for model input)
    tiles_tensor_list = [val_transform(tile) for tile in tiles_list_np]
    tiles_tensor = torch.stack(tiles_tensor_list).unsqueeze(0).to(device) # (1, 9, C, H, W)

    # --- 3. Init Solver ---
    # These weights are hyperparameters you must tune on a validation set
    solver_weights = {
        'w_c': 0.6,     # Classifier logp weight
        'w_t': 0.4,     # Transformer logp weight
        'alpha': 1.0,   # Weight for per-tile score (fused)
        'beta': 0.5,    # Weight for pairwise score
        'gamma': 0.5    # Weight for edge score
    }
    
    solver = JigsawSolver(classifier, transformer, pairwise, solver_weights, device)

    # --- 4. Solve ---
    print("Solving puzzle...")
    final_permutation = solver.solve(tiles_tensor, tiles_list_np)
    print(f"Solved permutation: {final_permutation}")

    # --- 5. Reconstruct and Visualize ---
    print("Reconstructing final image...")
    reconstructed_img = reconstruct_image(tiles_list_np, final_permutation)
    
    # Show the result
    visualize_solution(img_np, reconstructed_img)

if __name__ == "__main__":
    # Create dummy weights for testing if they don't exist
    # In a real scenario, these come from training.
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if not os.path.exists('weights/classifier.pth'):
        print("Warning: Creating dummy weights for testing.")
        torch.save(TileClassifier().state_dict(), 'weights/classifier.pth')
        torch.save(TransformerSolver().state_dict(), 'weights/transformer.pth')
        torch.save(PairwiseModel().state_dict(), 'weights/pairwise.pth')
    
    # Create dummy data for testing
    if not os.path.exists('data/sample_train'):
        os.makedirs('data/sample_train')
    if not os.path.exists('data/train.csv'):
        with open('data/train.csv', 'w') as f:
            f.write('image,label\n')
            f.write('dummy.jpg,601853724\n')
        dummy_img = (np.random.rand(240, 240, 3) * 255).astype(np.uint8)
        cv2.imwrite('data/sample_train/dummy.jpg', dummy_img)
    
    print("\nTo run inference, use the following command:")
    print("python inference.py --image data/sample_train/dummy.jpg\n")
    
    # Call main() so argparse reads the actual command-line args passed by the user.
    main()
