import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import prepare_dataloaders
from src.model import build_model
from src.train import train

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BASE=r'D:\DR_Project\data\raw'

CSV_PATH=os.path.join(BASE, 'train.csv')
IMG_DIR=os.path.join(BASE, 'train_images')
SAVE_DIR=r'D:\DR_Project\models'

print(f"Device : {DEVICE}")
print(f"CSV    : {CSV_PATH}")
print(f"Images : {IMG_DIR}")

loaders, train_df, val_df, test_df=prepare_dataloaders(CSV_PATH, IMG_DIR, batch_size=4, img_size=300)
model=build_model(num_classes=5, device=DEVICE)
history=train(model, loaders, train_df, save_dir=SAVE_DIR, device=DEVICE, epochs=10)

print("Done, Model saved to", SAVE_DIR)