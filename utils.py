import numpy as np
import torch
import random
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.cluster import KMeans
from datetime import datetime
import logging
import sys

def set_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_logging(log_file=None):
    """Configure logging."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # avoid duplicate init
    )

    logger = logging.getLogger(__name__)
    logger.info("Logger initialized. All logs will be flushed line by line.")
    return logger


def hungarian_alignment(y_true, y_pred):
    """Align predicted and true labels via Hungarian algorithm."""
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred, known_labels):
    """Compute clustering accuracy (overall/known/novel and H-Score)."""
    ind, w = hungarian_alignment(y_true, y_pred)
    
    # overall accuracy
    acc_all = sum([w[i, j] for i, j in ind]) / y_pred.size
    
    # label map
    ind_map = {j: i for i, j in ind}
    
    # known-class accuracy
    old_acc = 0
    total_old_instances = 0
    for i in known_labels:
        if i in ind_map:
            old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    
    old_acc = old_acc / total_old_instances if total_old_instances > 0 else 0
    
    # novel-class accuracy
    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_labels:
            if i in ind_map:
                new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    
    new_acc = new_acc / total_new_instances if total_new_instances > 0 else 0
    
    # harmonic mean (H-Score)
    h_score = 2 * old_acc * new_acc / (old_acc + new_acc) if (old_acc + new_acc) > 0 else 0
    
    return (
        round(acc_all * 100, 2), 
        round(old_acc * 100, 2), 
        round(new_acc * 100, 2), 
        round(h_score * 100, 2)
    )

def clustering_score(y_true, y_pred, known_labels):
    """Compute clustering metrics."""
    acc_all, acc_known, acc_novel, h_score = clustering_accuracy_score(y_true, y_pred, known_labels)
    
    return {
        'all-acc': acc_all,
        'k-acc': acc_known,
        'n-acc': acc_novel,
        'H-Score': h_score,
        'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2)
    }


def create_label_template(label):
    """Create label template."""
    return f"the category of the text is: {label}"

def extract_features_and_labels(model, dataloader, device):
    """Extract features and labels from a dataloader."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels = [
                t.to(device) for t in batch
            ]

            # encode text
            text_embeddings = model.encode_text(
                input_ids, attention_mask, token_type_ids
            )

            all_features.append(text_embeddings.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    features = np.vstack(all_features)
    labels = np.array(all_labels)

    return features, labels
