import os, sys

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, accuracy_score,
                            precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import autocast

DR_CLASSES=['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
LESION_TYPES={
    'MA':  'Microaneurysms',
    'HE':  'Hemorrhages',
    'EX':  'Hard Exudates',
    'SE':  'Soft Exudates',
    'OD':  'Optic Disc',
}
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images=images.to(device)
        with autocast():
            out=model(images)
            probs=torch.softmax(out, dim=1)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def comput_metrics(labels, preds, probs):
    acc=accuracy_score(labels, preds)
    p, r, f1, _=precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    try:
        lb=label_binarize(labels, classes=list(range(5)))
        auc=roc_auc_score(lb, probs, multi_class='ovr', average='weighted')
    except Exception:
        auc=None
    return{
        'Accuracy':  round(float(acc), 4),
        'Precision': round(float(p),   4),
        'Recall':    round(float(r),   4),
        'F1-Score':  round(float(f1),  4),
        'AUC-ROC':   round(float(auc), 4) if auc else 'N/A',
    }

def plot_confusion_matrix(labels, preds, save_path=None):
    cm=confusion_matrix(labels, preds)
    cm_norm=cm.astype(float)/cm.sum(axis=1, keepdims=True)
    fig, axes=plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Counts', 'Normalized'],
        ['d', '.2f']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=DR_CLASSES, yticklabels=DR_CLASSES, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion matrix ({title})', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_roc_curves(labels, probs, save_path=None):
    from sklearn.metrics import roc_curve
    lb=label_binarize(labels, classes=list(range(5)))
    fig, ax=plt.subplots(figsize=(7, 6))
    colors=['#3498db','#2ecc71','#e67e22','#e74c3c','#9b59b6']
    for i, (cls, col) in enumerate(zip(DR_CLASSES, colors)):
        fpr, tpr, _=roc_curve(lb[:, i], probs[:, i])
        auc=roc_auc_score(lb[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f'{cls} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curves - per DR stage')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def validate_xai_with_idrid_masks(model, image_tensor, image_id,
                                   masks_dir, device='cuda'):
    """
    IDRiD-specific: compare Grad-CAM activation against each lesion type mask.
    Returns dict of {lesion_type: IoU_score}.
    """
    from src.gradcam import generate_gradcam, overlay_lesion_mask

    overlay, gradcam_map, pred_class, probs = generate_gradcam(
        model, image_tensor, device=device)

    results = {}
    for code, name in LESION_TYPES.items():
        pattern = os.path.join(masks_dir, code,
                               f'{image_id}_{code}.tif')
        mask_files = glob.glob(pattern)
        if not mask_files:
            continue

        mask_raw = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            continue

        h, w = gradcam_map.shape
        mask_resized = cv2.resize(mask_raw, (w, h))
        mask_binary  = (mask_resized > 127).astype(np.uint8)

        iou, _ = overlay_lesion_mask(gradcam_map, mask_binary)
        results[name] = round(iou, 4)
        print(f"  {name:25s}: IoU = {iou:.4f}")

    return results

def full_evaluation(model, loaders, device, save_dir='results/'):
    os.makedirs(save_dir, exist_ok=True)
    preds, labels, probs=get_predictions(model, loaders['test'], device)
    metrics=comput_metrics(labels, preds, probs)
    print("\nTEST SET METRICS")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print("\nPER-CLASS REPORT")
    print(classification_report(labels, preds, target_names=DR_CLASSES, zero_division=0))
    plot_confusion_matrix(labels, preds, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    plot_roc_curves(probs=probs, labels=labels, save_path=os.path.join(save_dir, 'roc_curves.png'))
    return metrics, preds, labels, probs


if __name__=='__main__':
    from src.model import build_model
    from src.dataset import prepare_dataloaders
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=5, device=DEVICE)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    loaders, _, _, _=prepare_dataloaders(
        csv_path='data/raw/train.csv',
        image_dir='data/raw/train_images',
        batch_size=32
    )

    metrics, _, _, _=full_evaluation(model, loaders, device, save_dir='results/')
    print("\n=== Final Accuracy: {}% ===".format(metrics['Accuracy'] * 100))