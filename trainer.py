import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from models import create_tlsa_model
from losses import create_loss_function
from clustering import create_cluster_manager
from utils import clustering_score, extract_features_and_labels
from dataloader import Phase2Dataset, UniqueLabelBatchSampler

logger = logging.getLogger(__name__)

class TLSATrainer:
    """TLSA trainer."""

    def __init__(self, args, data_manager, device):
        self.args = args
        self.data_manager = data_manager
        self.device = device
        self.model = create_tlsa_model(
            args, num_known_classes=len(data_manager.known_labels)
        ).to(device)

        self.alignment_loss_fn = create_loss_function(args)

        self.cluster_manager = create_cluster_manager(args)

        # training history
        self.metrics_history = []
        self.current_epoch = 0

        # prepare save directories
        self._setup_save_directories()

    def _setup_save_directories(self):
        """Prepare save directory structure."""
        self.base_save_dir = self.args.save_results_path

        # sub-directories
        self.models_dir = os.path.join(self.base_save_dir, "models")

        # ensure directories exist
        for directory in [self.base_save_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info("Save directories created:")
        logger.info(f"   Base: {self.base_save_dir}")
        logger.info(f"   Models: {self.models_dir}")

    def train(self):
        """Full training pipeline."""
        logger.info("Starting training...")
        _train_start = time.time()
        

        # Phase A: supervised warm-up
        logger.info("Phase A: Supervised Warm-up")
        self.supervised_warmup()

        # Load Phase-1 best for Phase-2
        logger.info("Loading best Phase 1 on Dev Known model for Phase 2 training...")
        self._load_best_phase1_model()

        # Phase B: semi-supervised EM
        logger.info("Phase B: Semi-supervised EM Training")
        self.semi_supervised_training()

        # Final test evaluation (use current model)
        self.current_epoch = "Final"
        final_results = self.evaluate_results(self.data_manager.test_dataloader)

        # Save summary then final model
        training_time = time.time() - _train_start
        summary_path = self._save_training_summary(final_results, training_time=training_time)
        self._save_final_model(final_results, summary_path=summary_path)

        # print training summary
        self._print_training_summary()

        logger.info("Training completed!")
        return final_results

    def supervised_warmup(self):
        """Phase-1 warm-up with alignment loss (text/label encoders trained jointly)."""
        # load existing Phase-1 best if present
        warmup_model_path = self._get_warmup_model_path()

        if os.path.exists(warmup_model_path):
            logger.info(f"Loading warmup model from {warmup_model_path}")
            self._load_warmup_model(warmup_model_path)
            return

        logger.info("No warmup model found. Starting warm-up...")

        # enable training mode (both text/label encoders)
        self.model.set_training_mode('supervised')

        # optimizer
        optimizer = self._create_optimizer(self.model, phase='warmup')

        # build text-label pair dataloader
        phase1_dataloader = self._create_phase1_dataloader()

        best_performance = 0    # use dev k-acc as early-stop metric
        patience = 20           # early-stop patience
        patience_counter = 0

        for epoch in range(self.args.num_warmup_epochs):
            # fixed label prototypes for CE (if enabled)
            if hasattr(self.alignment_loss_fn, 'set_label_prototypes'):
                try:
                    protos = self._encode_label_prototypes(self.data_manager.known_labels)
                    self.alignment_loss_fn.set_label_prototypes(protos)
                except Exception as _e:
                    pass

            # train one epoch (alignment loss)
            train_loss = self._train_phase1_epoch(optimizer, phase1_dataloader)
            # evaluate on dev
            self.current_epoch = epoch + 1
            # dev known-class zero-shot accuracy (k-acc)
            eval_results = self.evaluate_dev_known_acc()
            # record metrics history
            self.metrics_history.append(eval_results.copy())
            # use dev k-acc as early-stop and best checkpoint criterion
            performance = eval_results.get('k-acc', 0)

            logger.info(f"Warmup Epoch {epoch + 1}/{self.args.num_warmup_epochs} - Loss: {train_loss:.4f}, Acc: {performance:.4f}")

            # save best (Phase-1)
            if performance > best_performance:
                best_performance = performance
                patience_counter = 0
                self._save_warmup_model(warmup_model_path, optimizer, epoch, best_performance)
                logger.info(f"New best performance: {best_performance:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

            # check early-stop
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                logger.info(f"Best performance achieved: {best_performance:.4f} at epoch {epoch + 1 - patience}")
                break

        logger.info(f"Warmup completed. Best performance: {best_performance:.4f}")

    def _get_warmup_model_path(self):
        """Path of Phase-1 best checkpoint."""
        return os.path.join(self.models_dir, "best_phase1_model.pth")

    def _save_warmup_model(self, save_path, optimizer, epoch, loss):
        """Save minimal warmup weights (only model_state_dict)."""
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)

    def _load_warmup_model(self, model_path):
        """Load warmup weights (supports minimal checkpoints)."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # Support either {'model_state_dict': ...} or raw state_dict
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded warmup weights from {model_path}")

    def _load_best_phase1_model(self):
        """Load Phase-1 best checkpoint if available."""
        warmup_model_path = self._get_warmup_model_path()
        if os.path.exists(warmup_model_path):
            logger.info(f"Loading best Phase 1 model from {warmup_model_path}")
            self._load_warmup_model(warmup_model_path)
        else:
            logger.warning("No Phase 1 best model found, continuing with current model state")

    def _save_specific_epoch_model(self, epoch):
        """Save minimal checkpoint at a specific epoch (only model_state_dict)."""
        save_path = os.path.join(self.models_dir, f"phase2_model_epoch_{epoch}.pth")
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        logger.info(f"Saved specific epoch {epoch} model to {save_path}")

    def _get_final_model_path(self):
        """Path of final checkpoint."""
        return os.path.join(self.models_dir, "final_model.pth")

    def _save_final_model(self, final_results=None, summary_path: Optional[str] = None):
        """Save final minimal checkpoint (only model_state_dict)."""
        save_path = self._get_final_model_path()
        payload = {'model_state_dict': self.model.state_dict()}
        try:
            torch.save(payload, save_path)
            logger.info(f"Final model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save final model to {save_path}: {e}")


    def _create_phase1_dataloader(self):
        """Build text/label pair dataloader (Phase-1)."""
        # build text-label pairs from labeled data
        pair_data = []

        for example in self.data_manager.train_labeled_examples:
            text = example.text_a
            label = example.label
            label_template = f"the category of the text is: {label}"

            # encode text
            text_encoding = self.data_manager.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.args.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # encode label template
            label_encoding = self.data_manager.tokenizer(
                label_template,
                add_special_tokens=True,
                max_length=self.args.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            pair_data.append({
                'text_input_ids': text_encoding['input_ids'].squeeze(0),
                'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
                'text_token_type_ids': text_encoding['token_type_ids'].squeeze(0),
                'label_input_ids': label_encoding['input_ids'].squeeze(0),
                'label_attention_mask': label_encoding['attention_mask'].squeeze(0),
                'label_token_type_ids': label_encoding['token_type_ids'].squeeze(0),
                'label': label  # keep original label for dedup
            })

        # dataset for pairs
        class TextLabelPairDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                return (
                    item['text_input_ids'],
                    item['text_attention_mask'],
                    item['text_token_type_ids'],
                    item['label_input_ids'],
                    item['label_attention_mask'],
                    item['label_token_type_ids'],
                    item['label']
                )

        dataset = TextLabelPairDataset(pair_data)

        # label -> idx (known classes only in Phase-1)
        label_to_idx = {lab: i for i, lab in enumerate(self.data_manager.known_labels)}

        # collate: remove duplicate labels in a batch
        def remove_duplicate_labels_collate(batch):
            # deduplicate by label (keep first occurrence)
            seen_labels = set()
            filtered_batch = []

            for item in batch:
                label = item[6]
                if label not in seen_labels:
                    seen_labels.add(label)
                    filtered_batch.append(item)

            if len(filtered_batch) == 0:
                # fallback: keep one sample if all dup
                filtered_batch = [batch[0]]

            # re-pack tensors
            text_input_ids = torch.stack([item[0] for item in filtered_batch])
            text_attention_mask = torch.stack([item[1] for item in filtered_batch])
            text_token_type_ids = torch.stack([item[2] for item in filtered_batch])
            label_input_ids = torch.stack([item[3] for item in filtered_batch])
            label_attention_mask = torch.stack([item[4] for item in filtered_batch])
            label_token_type_ids = torch.stack([item[5] for item in filtered_batch])
            targets = torch.tensor([label_to_idx[item[6]] for item in filtered_batch], dtype=torch.long)

            return (text_input_ids, text_attention_mask, text_token_type_ids,
                    label_input_ids, label_attention_mask, label_token_type_ids,
                    targets)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=remove_duplicate_labels_collate,
            drop_last=False  # allow variable batch sizes
        )

        # label distribution stats
        label_counts = {}
        for item in pair_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        unique_labels = len(label_counts)
        total_samples = len(pair_data)
        logger.info(f"Phase 1 DataLoader: {total_samples} samples, {unique_labels} labels")
        return dataloader

    def _train_phase1_epoch(self, optimizer, phase1_dataloader):
        """Train one epoch with alignment loss."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(phase1_dataloader, desc="Phase 1 alignment training")

        for batch in progress_bar:
            # unpack; supports optional targets
            if isinstance(batch, (list, tuple)) and len(batch) == 7:
                (text_input_ids, text_attention_mask, text_token_type_ids,
                 label_input_ids, label_attention_mask, label_token_type_ids,
                 targets) = batch
                targets = targets.to(self.device)
            else:
                (text_input_ids, text_attention_mask, text_token_type_ids,
                 label_input_ids, label_attention_mask, label_token_type_ids) = batch
                targets = None
            # move to device
            text_input_ids = text_input_ids.to(self.device)
            text_attention_mask = text_attention_mask.to(self.device)
            text_token_type_ids = text_token_type_ids.to(self.device)
            label_input_ids = label_input_ids.to(self.device)
            label_attention_mask = label_attention_mask.to(self.device)
            label_token_type_ids = label_token_type_ids.to(self.device)

            # current batch size (for logs)
            current_batch_size = text_input_ids.size(0)
            # encode text/labels
            text_embeddings = self.model.encode_text(
                text_input_ids, text_attention_mask, text_token_type_ids
            )
            label_embeddings = self.model.encode_labels(
                label_input_ids, label_attention_mask, label_token_type_ids
            )

            # compute loss (supports cached CE)
            if hasattr(self.alignment_loss_fn, 'set_targets') and hasattr(self.alignment_loss_fn, 'label_prototypes') and targets is not None:
                self.alignment_loss_fn.set_targets(targets)
                loss = self.alignment_loss_fn(text_embeddings, None)
            else:
                loss = self.alignment_loss_fn(text_embeddings, label_embeddings)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_size': current_batch_size
            })

        return total_loss / max(1, num_batches)

    def semi_supervised_training(self, start_epoch: int = 1):
        """Phase-2 semi-supervised training loop (no dev eval/best saving)."""
        self.model.set_training_mode('semi_supervised')
        start_epoch = max(1, int(start_epoch))
        for epoch in range(start_epoch - 1, self.args.num_train_epochs):
            logger.info(f"Phase 2 Epoch {epoch + 1}/{self.args.num_train_epochs}")
            epoch_start_time = time.time()

            # linear schedule high_conf_ratio: args.high_conf_ratio -> args.high_conf_ratio_end
            t = epoch / max(1, (self.args.num_train_epochs - 1))
            scheduled_high = (1 - t) * getattr(self.args, 'high_conf_ratio', 0.2) + t * getattr(self.args, 'high_conf_ratio_end', 0.4)
            # clamp
            scheduled_high = max(0.0, min(0.99, scheduled_high))
            self.current_high_conf_ratio = scheduled_high
            logger.info(f"[SCHEDULE] high_conf_ratio={scheduled_high:.3f} (epoch {epoch+1})")

            # Step 1: discover novel clusters via Hungarian matching; return details
            novel_clusters_info, known_clusters_info, all_cluster_centers = self._discover_novel_clusters_hungarian()
            logger.info(f"Found {len(novel_clusters_info)} novel clusters")
            logger.info(f"Found {len(known_clusters_info)} known clusters")

            # Step 2: generate labels for novel clusters (LLM)
            novel_labels = self._generate_labels_for_novel_clusters(novel_clusters_info)
            logger.info(f"Generated {len(novel_labels)} novel labels")
            # optional: check duplicates/overlap
            try:
                from collections import Counter
                counts = Counter(novel_labels)
                dup_novel = [lab for lab, c in counts.items() if c > 1]
                if dup_novel:
                    logger.warning(f"[CHECK] Duplicated novel labels (within novel set): {dup_novel}")
                # overlap with known labels (should be avoided by LLM)
                overlap = sorted(set(novel_labels) & set(self.data_manager.known_labels))
                if overlap:
                    logger.warning(f"[CHECK] Novel labels overlapping known labels: {overlap}")
            except Exception as e:
                logger.warning(f"[CHECK] Failed to check duplicate labels: {e}")

            # Step 2.5: label refinement (post-hoc) after a given start epoch
            refine_start_epoch = max(1, int(getattr(self.args, 'label_refine_start_epoch', 1)))
            current_epoch_idx = epoch + 1  # 1-based

            if current_epoch_idx >= refine_start_epoch:
                try:
                    # debug dump only at the first refine epoch
                    debug_log_first = (current_epoch_idx == refine_start_epoch) and (not getattr(self, '_label_refine_debug_done', False))

                    info = self._label_refine_posthoc(
                        known_labels=self.data_manager.known_labels,
                        novel_clusters_info=novel_clusters_info,
                        known_clusters_info=known_clusters_info,
                        all_cluster_centers=all_cluster_centers,
                        debug_log_first_epoch=debug_log_first
                    )
                    # rebuild novel_labels after refinement (dedup)
                    novel_labels = sorted(set([
                        v.get('assigned_label') for _cid, v in novel_clusters_info.items() if v.get('assigned_label')
                    ]))
                    logger.info(
                        f"[LABEL-REFINE-SUMMARY] components={info.get('num_components', 0)}, processed={info.get('num_processed_components', 0)}, renamed_novel={info.get('num_renamed_novel', 0)}"
                    )
                except Exception as e:
                    logger.warning(f"[LABEL-REFINE] post-hoc label refinement failed: {e}")
                finally:
                    if current_epoch_idx == refine_start_epoch:
                        try:
                            setattr(self, '_label_refine_debug_done', True)
                        except Exception:
                            pass

            # Step 3: assemble all labels (known + novel)
            all_labels = self.data_manager.known_labels + novel_labels
            logger.info(f"Total labels: {len(all_labels)} (Known: {len(self.data_manager.known_labels)}, Novel: {len(novel_labels)})")

            if hasattr(self.alignment_loss_fn, 'set_label_prototypes'):
                try:
                    protos = self._encode_label_prototypes(all_labels)
                    self.alignment_loss_fn.set_label_prototypes(protos)
                    logger.info(f"Updated label_prototypes with {len(all_labels)} labels after potential renaming")
                except Exception as _e:
                    logger.warning(f"Failed to update label_prototypes: {_e}")

            train_loss = self._train_phase2_epoch_clip(all_labels, novel_clusters_info, known_clusters_info)
            self.current_epoch = epoch + 1
            save_at_epochs = getattr(self.args, 'save_phase2_model_at_epochs', [])
            if (epoch + 1) in save_at_epochs:
                self._save_specific_epoch_model(epoch + 1)
                
            epoch_time = time.time() - epoch_start_time
            logger.info(f"   Epoch {epoch + 1} completed in {epoch_time:.1f}s - Loss: {train_loss:.4f}")

        logger.info("Phase 2 completed.")

    def _discover_novel_clusters_hungarian(self):
        """Discover novel clusters via Hungarian matching and return details."""
        # extract features/texts for unlabeled data
        all_unlabeled_features, all_unlabeled_texts = self._extract_unlabeled_features_and_texts()
        # extract features/texts/true labels for labeled data
        labeled_features, _, labeled_true_labels = self._extract_labeled_features_and_texts()
        # cluster all unlabeled into total number of classes
        total_classes = len(self.data_manager.all_labels)
        all_cluster_labels, all_cluster_centers = self.cluster_manager.perform_clustering(
            all_unlabeled_features, num_clusters=total_classes
        )
        # compute per-known-class centroids
        label_to_indices = {}
        for idx, lab in enumerate(labeled_true_labels):
            label_to_indices.setdefault(lab, []).append(idx)

        known_label_centers = []
        label_names_for_centers = []
        for lab in self.data_manager.known_labels:
            idxs = label_to_indices.get(lab, [])
            if len(idxs) > 0:
                center = labeled_features[idxs].mean(axis=0)
                known_label_centers.append(center)
                label_names_for_centers.append(lab)
        labeled_centers_tensor = torch.tensor(np.vstack(known_label_centers)).to(self.device)

        # similarity between cluster centers and known label centroids
        all_centers_tensor = torch.tensor(all_cluster_centers).to(self.device)
        all_centers_norm = torch.nn.functional.normalize(all_centers_tensor, dim=-1)
        labeled_centers_norm = torch.nn.functional.normalize(labeled_centers_tensor, dim=-1)
        similarity_matrix = torch.mm(all_centers_norm, labeled_centers_norm.t()).cpu().numpy()
        # Hungarian matching: associate clusters to known labels
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
        # map cluster_id -> known label
        cluster_to_known_label_mapping = {}
        for cluster_id, col_idx in zip(row_indices, col_indices):
            if 0 <= col_idx < len(label_names_for_centers):
                cluster_to_known_label_mapping[cluster_id] = label_names_for_centers[col_idx]
        # unmatched clusters are novel
        matched_cluster_ids = set(row_indices)
        all_cluster_ids = set(range(total_classes))
        novel_cluster_ids = list(all_cluster_ids - matched_cluster_ids)

        logger.info("Matching results:")
        logger.info(f"   Found {len(novel_cluster_ids)} novel clusters: {novel_cluster_ids}")
        logger.info(f"   Matched {len(matched_cluster_ids)} clusters to known classes: {sorted(matched_cluster_ids)}")

        # show assignment similarities
        assigned_similarities = [
            similarity_matrix[i, j]
            for i, j in zip(row_indices, col_indices)
            if 0 <= j < similarity_matrix.shape[1]
        ]
        if assigned_similarities:
            logger.info(f"   Assigned pair similarities: avg={np.mean(assigned_similarities):.3f}, min={np.min(assigned_similarities):.3f}, max={np.max(assigned_similarities):.3f}")

        # novel cluster details
        novel_clusters_detailed = {}
        for cluster_id in novel_cluster_ids:
            cluster_mask = (all_cluster_labels == cluster_id)
            cluster_texts = [all_unlabeled_texts[i] for i in range(len(all_unlabeled_texts)) if cluster_mask[i]]
            cluster_features = all_unlabeled_features[cluster_mask]
            center = all_cluster_centers[cluster_id]
            # distances to center (for logs)
            dists = np.linalg.norm(cluster_features - center, axis=1)
            # confidence ranking via normalized margin
            diff = cluster_features[:, None, :] - all_cluster_centers[None, :, :]
            dmat = np.linalg.norm(diff, axis=2)  # [n_k, K]
            
            # distance to own center (should be min)
            dist_to_own = dmat[:, cluster_id]  # [n_k]
            
            # min distance to other centers (mask own center)
            dmat_others = dmat.copy()
            dmat_others[:, cluster_id] = np.inf
            nearest_other_idx = np.argmin(dmat_others, axis=1)  # [n_k]
            dist_to_nearest_other = dmat_others[np.arange(len(cluster_features)), nearest_other_idx]  # [n_k]
            
            # inter-center distances
            center_to_center_dists = np.linalg.norm(
                all_cluster_centers[nearest_other_idx] - center[None, :], axis=1
            )  # [n_k]
            
            # normalized margin: (dist_other - dist_own) / dist_between_centers
            normalized_margins = np.where(
                center_to_center_dists > 1e-8,
                (dist_to_nearest_other - dist_to_own) / center_to_center_dists,
                0.0
            )
            order = np.argsort(-normalized_margins)  # desc: larger => higher confidence

            novel_clusters_detailed[cluster_id] = {
                'texts': cluster_texts,
                'features': cluster_features,
                'center': center,
                'dists': dists,
                'sorted_indices_by_conf': order,
                'all_indices': np.where(cluster_mask)[0]
            }

        # known cluster details (matched from unlabeled)
        known_clusters_detailed = {}
        for cluster_id in matched_cluster_ids:
            cluster_mask = (all_cluster_labels == cluster_id)
            cluster_texts = [all_unlabeled_texts[j] for j in range(len(all_unlabeled_texts)) if cluster_mask[j]]
            cluster_features = all_unlabeled_features[cluster_mask]
            center = all_cluster_centers[cluster_id]
            known_label = cluster_to_known_label_mapping.get(cluster_id, None)
            # distances to center (for logs)
            dists = np.linalg.norm(cluster_features - center, axis=1)
            # confidence via normalized margin
            diff = cluster_features[:, None, :] - all_cluster_centers[None, :, :]
            dmat = np.linalg.norm(diff, axis=2)  # [n_k, K]
            
            dist_to_own = dmat[:, cluster_id]  # [n_k]
            
            dmat_others = dmat.copy()
            dmat_others[:, cluster_id] = np.inf
            nearest_other_idx = np.argmin(dmat_others, axis=1)  # [n_k]
            dist_to_nearest_other = dmat_others[np.arange(len(cluster_features)), nearest_other_idx]  # [n_k]
            
            center_to_center_dists = np.linalg.norm(
                all_cluster_centers[nearest_other_idx] - center[None, :], axis=1
            )  # [n_k]
            
            normalized_margins = np.where(
                center_to_center_dists > 1e-8,
                (dist_to_nearest_other - dist_to_own) / center_to_center_dists,
                0.0
            )
            order = np.argsort(-normalized_margins)

            known_clusters_detailed[cluster_id] = {
                'texts': cluster_texts,
                'features': cluster_features,
                'center': center,
                'dists': dists,
                'sorted_indices_by_conf': order,
                'all_indices': np.where(cluster_mask)[0],
                'assigned_label': known_label
            }

        logger.info(f"Prepared detailed info for {len(novel_clusters_detailed)} novel clusters")
        logger.info(f"Prepared detailed info for {len(known_clusters_detailed)} known clusters")

        # sample counts
        total_novel_samples = sum(len(info['texts']) for info in novel_clusters_detailed.values())
        total_known_samples = sum(len(info['texts']) for info in known_clusters_detailed.values())
        logger.info(f"Total novel samples: {total_novel_samples}")
        logger.info(f"Total known unlabeled samples: {total_known_samples}")

        return novel_clusters_detailed, known_clusters_detailed, all_cluster_centers

    def _generate_labels_for_novel_clusters(self, novel_clusters_info):
        """Generate labels for novel clusters (sample high-confidence texts)."""
        novel_labels = []
        # init LLM label generator
        from llm_integration import create_label_generator
        llm_generator = create_label_generator()
        for cluster_id, cluster_info in novel_clusters_info.items():
            texts = cluster_info['texts']
            order = cluster_info.get('sorted_indices_by_conf')
            if order is None:
                order = np.arange(len(texts))
            texts_sorted = [texts[i] for i in order]

            n = len(texts_sorted)
            if n == 0:
                logger.warning(f"   No texts in novel cluster {cluster_id}, skipping...")
                continue

            # bucket split (high/mid/low)
            high_ratio = getattr(self, 'current_high_conf_ratio', getattr(self.args, 'high_conf_ratio', 0.4))
            low_ratio = getattr(self.args, 'low_conf_ratio', 0.2)
            h = int(round(n * high_ratio))
            l = int(round(n * low_ratio))
            h = max(0, min(h, n))
            l = max(0, min(l, n - h))

            high_conf_texts = texts_sorted[:h]
            # sample up to 60 from high-confidence bucket
            if len(high_conf_texts) > 0:
                k = min(60, len(high_conf_texts))
                if len(high_conf_texts) <= k:
                    sample_texts = list(high_conf_texts)
                else:
                    idxs = np.random.choice(len(high_conf_texts), size=k, replace=False)
                    sample_texts = [high_conf_texts[i] for i in idxs]
            else:
                # fallback: top-60 overall
                k = min(60, n)
                sample_texts = texts_sorted[:k]
            
            # context: only known labels to avoid contamination
            current_known_labels = list(self.data_manager.known_labels)
            try:
                generated_label = llm_generator.generate_label_for_texts(
                    sample_texts,
                    current_known_labels,
                    cluster_id
                )
                if generated_label in current_known_labels:
                    logger.info(
                        f"   Duplicate label '{generated_label}' detected for cluster {cluster_id}. "
                        "Keeping original result; downstream force-renaming will handle conflicts."
                    )
            except Exception as e:
                logger.warning(f"   Exception in label generation for cluster {cluster_id}: {e}")
                raise Exception(f"Failed to generate label for cluster {cluster_id}.")
            # write back result (label only)
            cluster_info['assigned_label'] = generated_label
            # debug output
            try:
                logger.info(f"[DEBUG] Cluster {cluster_id} -> label: {generated_label}, high_n={len(high_conf_texts)}")
                for pt in sample_texts[:3]:
                    _pt = pt[:120].replace('\n', ' ')
                    logger.info(f"[DEBUG]  sample: {_pt}")
            except Exception:
                pass

            novel_labels.append(generated_label)

        return novel_labels

    def _encode_label_prototypes(self, labels_list: List[str]):
        """Encode label templates to normalized prototypes [K, D]."""
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            templates = [f"the category of the text is: {lab}" for lab in labels_list]
            enc = self.data_manager.tokenizer(
                templates,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=self.args.max_seq_length,
                return_tensors='pt'
            )
            label_input_ids = enc['input_ids'].to(self.device)
            label_attention_mask = enc['attention_mask'].to(self.device)
            label_token_type_ids = enc.get('token_type_ids', torch.zeros_like(label_input_ids)).to(self.device)
            proto = self.model.encode_labels(label_input_ids, label_attention_mask, label_token_type_ids)
            proto = F.normalize(proto, dim=-1)
        self.model.train()
        return proto

    def _label_refine_posthoc(self, known_labels: List[str], novel_clusters_info: Dict[int, Dict], known_clusters_info: Dict[int, Dict], all_cluster_centers: np.ndarray, debug_log_first_epoch: bool = False) -> Dict[str, int]:
        """Post-hoc label refinement by connected components on label-prototype similarity.
        Applies per-component renaming for novel clusters (simple implementation)."""
        # build nodes: known labels + novel clusters (no dedup)
        nodes_texts: List[str] = []
        nodes_meta: List[Dict] = []

        # known label nodes
        for lab in known_labels:
            nodes_texts.append(lab)
            nodes_meta.append({'type': 'known', 'label': lab})

        # novel cluster nodes (per cluster)
        for cid, info in novel_clusters_info.items():
            lab = info.get('assigned_label', None)
            if not lab:
                continue
            nodes_texts.append(lab)
            nodes_meta.append({'type': 'novel', 'cid': cid, 'label': lab})

        K_all = len(nodes_texts)
        if K_all <= 1:
            return {'num_components': K_all, 'num_processed_components': 0, 'num_renamed_novel': 0}

        # 2) encode label prototypes and build similarity matrix
        try:
            proto = self._encode_label_prototypes(nodes_texts)  # [K, D], normalized
        except Exception as e:
            logger.warning(f"[LABEL-REFINE] encoding failed, skip label refinement: {e}")
            return {'num_components': 0, 'num_processed_components': 0, 'num_renamed_novel': 0}

        with torch.no_grad():
            sim = torch.clamp(torch.matmul(proto, proto.t()), -1.0, 1.0).cpu().numpy()

        # 3) build connected components by similarity threshold
        SIM_THR = 0.96  # fixed threshold
        parent = list(range(K_all))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(K_all):
            order = np.argsort(-sim[i])
            for j in order:
                if j == i:
                    continue
                if sim[i, j] < SIM_THR:
                    break
                union(i, j)

        comp_map: Dict[int, List[int]] = {}
        for idx in range(K_all):
            r = find(idx)
            comp_map.setdefault(r, []).append(idx)
        components = list(comp_map.values())
        num_components = len(components)

        # component-level stats (concise)
        try:
            num_will = 0
            num_mixed = 0
            num_pure = 0
            for _root_idx, _members in comp_map.items():
                _novel_cnt = sum(1 for _i in _members if nodes_meta[_i].get('type') == 'novel')
                _known_cnt = sum(1 for _i in _members if nodes_meta[_i].get('type') == 'known')
                if len(_members) > 1 and _novel_cnt > 0:
                    num_will += 1
                    if _known_cnt > 0:
                        num_mixed += 1
                    else:
                        num_pure += 1
            logger.info(f"[LABEL-REFINE] components={num_components}, will_process={num_will}, mixed={num_mixed}, pure={num_pure}")
        except Exception:
            pass

        # 4) LLM generator and sampling helpers
        from llm_integration import create_label_generator
        llm = create_label_generator()

        MAX_TEXTS_PER_GROUP = 25

        def _collect_group_texts(_info: Dict) -> List[str]:
            """Collect representative texts from top-confidence subset (cap per group)."""
            texts = _info.get('texts', []) or []
            order = _info.get('sorted_indices_by_conf', None)
            if order is None:
                order = np.arange(len(texts))
            texts_sorted = [texts[i] for i in order]

            n = len(texts_sorted)
            if n == 0:
                return []

            # take top-35% high-confidence
            high_ratio = 0.35
            n_high = max(1, int(round(n * high_ratio)))  # at least 1
            n_high = min(n_high, n)
            
            # top-conf subset
            high_conf_texts = texts_sorted[:n_high]
            
            # sample up to MAX_TEXTS_PER_GROUP
            K = min(MAX_TEXTS_PER_GROUP, len(high_conf_texts))
            if len(high_conf_texts) <= K:
                picked = high_conf_texts
            else:
                # random sample K
                idxs = np.random.choice(len(high_conf_texts), size=K, replace=False)
                picked = [high_conf_texts[i] for i in sorted(idxs)]
            
            return picked

        def _collect_known_texts_with_local_confidence(_label: str, assigned_cluster_id: int, all_cids_in_comp: List[int], local_centers_array: np.ndarray, max_k: int = 25) -> List[str]:
            #Collect texts for a known label from labeled set, filtered by local confidence.
            try:
                texts = []
                features_list = []
                
                labeled_examples = getattr(self.data_manager, 'train_labeled_examples', []) or []
                labeled_dataset = getattr(self.data_manager, 'train_labeled_dataset', []) or []
                
                for i, ex in enumerate(labeled_examples):
                    if getattr(ex, 'label', None) == _label:
                        texts.append(getattr(ex, 'text_a', ''))
                        # Extract features for this example
                        if i < len(labeled_dataset):
                            feature = labeled_dataset[i]
                            input_ids = feature[0].unsqueeze(0).to(self.device)
                            attention_mask = feature[1].unsqueeze(0).to(self.device)
                            token_type_ids = feature[2].unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                text_embedding = self.model.encode_text(
                                    input_ids, attention_mask, token_type_ids
                                )
                                features_list.append(text_embedding.cpu().numpy())
                
                if len(texts) == 0 or len(features_list) == 0:
                    return []
                
                # Convert to array
                labeled_features = np.vstack(features_list)  # [n_labeled, D]
                
                # Compute local confidence for each labeled sample
                # Use the ASSIGNED cluster as "own" cluster (not the nearest cluster)
                diff = labeled_features[:, None, :] - local_centers_array[None, :, :]  # [n_labeled, K_local, D]
                dmat = np.linalg.norm(diff, axis=2)  # [n_labeled, K_local]
                
                # Find the index of assigned cluster in local_centers_array
                try:
                    assigned_local_idx = all_cids_in_comp.index(assigned_cluster_id)
                except ValueError:
                    # Assigned cluster not in component (shouldn't happen)
                    logger.warning(f"[LABEL-REFINE] Assigned cluster {assigned_cluster_id} not in component for label '{_label}'")
                    return []
                
                if dmat.shape[1] >= 2:
                    # Distance to assigned (own) cluster
                    dist_to_own = dmat[:, assigned_local_idx]  # [n_labeled]
                    
                    # Find nearest OTHER cluster (mask assigned cluster)
                    dmat_others = dmat.copy()
                    dmat_others[:, assigned_local_idx] = np.inf
                    nearest_other_idx = np.argmin(dmat_others, axis=1)  # [n_labeled]
                    dist_to_nearest_other = dmat_others[np.arange(len(labeled_features)), nearest_other_idx]  # [n_labeled]
                    
                    # Compute inter-center distances (assigned center to nearest other center)
                    center_to_center_dists = np.linalg.norm(
                        local_centers_array[nearest_other_idx] - local_centers_array[assigned_local_idx][None, :], axis=1
                    )  # [n_labeled]
                    
                    # Normalized margin: (dist_to_nearest_other - dist_to_own) / dist_between_centers
                    normalized_margins = np.where(
                        center_to_center_dists > 1e-8,
                        (dist_to_nearest_other - dist_to_own) / center_to_center_dists,
                        0.0
                    )
                    
                    # Sort by confidence (higher margin = higher confidence)
                    local_order = np.argsort(-normalized_margins)  # descending
                else:
                    # If only one local cluster, sort by distance to that cluster
                    dists = dmat[:, 0]
                    local_order = np.argsort(dists)  # ascending
                
                # Take top 35% high-confidence samples
                n = len(texts)
                high_ratio = 0.35
                n_high = max(1, int(round(n * high_ratio)))
                n_high = min(n_high, n)
                
                # Get high-confidence indices
                high_conf_indices = local_order[:n_high]
                
                # Sample up to max_k from high-confidence subset
                K = min(max_k, len(high_conf_indices))
                if len(high_conf_indices) <= K:
                    picked_indices = high_conf_indices
                else:
                    # Random sample K
                    sampled_positions = np.random.choice(len(high_conf_indices), size=K, replace=False)
                    picked_indices = high_conf_indices[sampled_positions]
                
                # Return the selected texts
                return [texts[i] for i in sorted(picked_indices)]
                
            except Exception as e:
                logger.warning(f"[LABEL-REFINE] Failed to collect known texts with local confidence for '{_label}': {e}")
                return []

        num_processed_components = 0
        num_renamed_novel = 0

        # 5) iterate components, decide trigger, call LLM
        for root, members in comp_map.items():
            # collect novel clusters and known labels in this component
            novel_cids: List[int] = []
            known_in_comp: List[str] = []
            for idx in members:
                meta = nodes_meta[idx]
                if meta.get('type') == 'novel':
                    novel_cids.append(meta['cid'])
                else:
                    known_in_comp.append(meta['label'])

            sim_stats = None
            if len(members) > 1:
                try:
                    local_sim = sim[np.ix_(members, members)]
                    mask = ~np.eye(len(members), dtype=bool)
                    sim_vals = local_sim[mask]
                    if sim_vals.size > 0:
                        sim_stats = {
                            'min': float(np.min(sim_vals)),
                            'avg': float(np.mean(sim_vals)),
                            'max': float(np.max(sim_vals)),
                        }
                except Exception:
                    sim_stats = None

            total_nodes = len(members)
            if total_nodes <= 1:
                continue
            if len(novel_cids) == 0:
                continue

            num_processed_components += 1

            # recompute local confidence for this component (using only local clusters)
            # collect cluster IDs (known + novel) in this component
            all_cids_in_comp = []
            for idx in members:
                meta = nodes_meta[idx]
                if meta.get('type') == 'novel':
                    all_cids_in_comp.append(meta['cid'])
                elif meta.get('type') == 'known':
                    # map known label back to cluster_id via matching
                    known_label = meta['label']
                    for kcid, kinfo in known_clusters_info.items():
                        if kinfo.get('assigned_label') == known_label:
                            all_cids_in_comp.append(kcid)
                            break
            
            # extract local centers
            local_centers = all_cluster_centers[all_cids_in_comp]  # [K_local, D]
            
            # logs: local confidence recompute
            try:
                logger.info(
                    f"[LABEL-REFINE] component {root}: recomputing local confidence for {len(novel_cids)} novel clusters "
                    f"based on {len(all_cids_in_comp)} local clusters (global K={len(all_cluster_centers)})"
                )
            except Exception:
                pass
            
            # recompute local confidence for each novel cluster
            for cid in novel_cids:
                cluster_info = novel_clusters_info[cid]
                cluster_features = cluster_info['features']
                center = cluster_info['center']
                
                local_cluster_idx = all_cids_in_comp.index(cid)
                diff = cluster_features[:, None, :] - local_centers[None, :, :]
                dmat = np.linalg.norm(diff, axis=2)  # [n_k, K_local]
                
                if dmat.shape[1] >= 2:
                    dist_to_own = dmat[:, local_cluster_idx]
                    
                    dmat_others = dmat.copy()
                    dmat_others[:, local_cluster_idx] = np.inf
                    nearest_other_idx = np.argmin(dmat_others, axis=1)
                    dist_to_nearest_other = dmat_others[np.arange(len(cluster_features)), nearest_other_idx]
                    
                    center_to_center_dists = np.linalg.norm(
                        local_centers[nearest_other_idx] - center[None, :], axis=1
                    )
                    
                    normalized_margins = np.where(
                        center_to_center_dists > 1e-8,
                        (dist_to_nearest_other - dist_to_own) / center_to_center_dists,
                        0.0
                    )
                    local_order = np.argsort(-normalized_margins)
                else:
                    dists = cluster_info['dists']
                    local_order = np.argsort(dists)
                
                cluster_info['sorted_indices_by_conf_local'] = local_order
            
            # sample using local confidence
            cids_sorted = sorted(novel_cids)
            groups_texts = []
            for cid in cids_sorted:
                cluster_info = novel_clusters_info[cid]
                # temporarily replace with local confidence order
                original_order = cluster_info.get('sorted_indices_by_conf')
                cluster_info['sorted_indices_by_conf'] = cluster_info.get('sorted_indices_by_conf_local', original_order)
                texts = _collect_group_texts(cluster_info)
                # restore global order
                cluster_info['sorted_indices_by_conf'] = original_order
                groups_texts.append(texts)

            new_labels = None
            try:
                sim_text = ""
                if sim_stats:
                    sim_text = "; sim[min={min:.3f}, avg={avg:.3f}, max={max:.3f}]".format(**sim_stats)
                if len(known_in_comp) > 0:
                    # mixed: rename only novel; provide local forbidden list and sample texts with local confidence
                    try:
                        logger.info(
                            f"[LABEL-REFINE] component {root}: type=mixed, novel={len(cids_sorted)}, known={len(known_in_comp)}; "
                            f"calling LLM(mixed) with local-confidence-filtered known context{sim_text}"
                        )
                    except Exception:
                        pass
                    known_context = []
                    for klab in known_in_comp:
                        # Find the assigned cluster ID for this known label
                        assigned_cid = None
                        for kcid, kinfo in known_clusters_info.items():
                            if kinfo.get('assigned_label') == klab:
                                assigned_cid = kcid
                                break
                        
                        if assigned_cid is not None:
                            # Use local confidence filtering for known labeled data
                            ktexts = _collect_known_texts_with_local_confidence(
                                klab, assigned_cid, all_cids_in_comp, local_centers, max_k=25
                            )
                            known_context.append({'label': klab, 'sample_texts': ktexts})
                        else:
                            logger.warning(f"[LABEL-REFINE] Could not find assigned cluster for known label '{klab}'")
                            known_context.append({'label': klab, 'sample_texts': []})

                    new_labels = llm.generate_labels_for_groups_mixed(
                        groups_sample_texts=groups_texts,
                        forbidden_labels_local=known_in_comp,
                        global_forbidden_labels=list(known_labels),
                        known_labels_context=known_context,
                        debug_log=debug_log_first_epoch
                    )
                else:
                    # pure novel: rename all novel in this component; avoid global known conflicts
                    try:
                        logger.info(
                            f"[LABEL-REFINE] component {root}: type=pure, novel={len(cids_sorted)}; "
                            f"calling LLM(pure){sim_text}"
                        )
                    except Exception:
                        pass
                    new_labels = llm.generate_labels_for_groups_pure(
                        groups_sample_texts=groups_texts,
                        global_forbidden_labels=list(known_labels),
                        debug_log=debug_log_first_epoch
                    )
            except Exception as e:
                logger.warning(f"[LABEL-REFINE] LLM labeling failed for component {root}: {e}")
                new_labels = None

            # strict length check; skip component if invalid
            if not new_labels or len(new_labels) != len(cids_sorted):
                logger.warning(f"[LABEL-REFINE] component {root}: invalid labels returned (expected {len(cids_sorted)}, got {0 if not new_labels else len(new_labels)}), skip this component")
                continue

            # write back (no post-processing)
            rename_records = []
            for pos, cid in enumerate(cids_sorted):
                old = novel_clusters_info[cid].get('assigned_label', None)
                new = new_labels[pos]
                if not isinstance(new, str):
                    new = "" if new is None else str(new)
                new = new.strip()
                if not new:
                    logger.warning(f"[LABEL-REFINE] component {root}, cluster {cid}: empty label returned, keeping old='{old}'")
                    rename_records.append((cid, old, old, False))
                    continue
                novel_clusters_info[cid]['assigned_label'] = new
                changed = new != old
                if changed:
                    num_renamed_novel += 1
                rename_records.append((cid, old, new, changed))

            try:
                preview = [(cid, old, new) for cid, old, new, _ in rename_records[:5]]
                comp_type = 'mixed' if len(known_in_comp) > 0 else 'pure'
                logger.info(
                    f"[LABEL-REFINE] component {root} ({comp_type}): renamed K={len(rename_records)} -> "
                    f"{[(cid, old, new) for cid, old, new, _ in rename_records] if len(rename_records) <= 5 else preview}"
                )
            except Exception:
                pass

        return {
            'num_components': num_components,
            'num_processed_components': num_processed_components,
            'num_renamed_novel': num_renamed_novel,
        }

    def _train_phase2_epoch_clip(self, all_labels, novel_clusters_info, known_clusters_info):
        """Train one Phase-2 epoch with alignment loss."""
        optimizer = self._create_optimizer(self.model, phase='semi_supervised')
        dataloader = self._create_phase2_dataloader(all_labels, novel_clusters_info, known_clusters_info)

        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Phase 2 alignment training")

        for batch in progress_bar:
            # unpack, support optional targets at the end
            if isinstance(batch, (list, tuple)) and len(batch) == 7:
                (text_input_ids, text_attention_mask, text_token_type_ids,
                 label_input_ids, label_attention_mask, label_token_type_ids,
                 targets) = batch
                targets = targets.to(self.device)
            else:
                (text_input_ids, text_attention_mask, text_token_type_ids,
                 label_input_ids, label_attention_mask, label_token_type_ids) = batch
                targets = None
            # move tensors to device
            text_input_ids = text_input_ids.to(self.device)
            text_attention_mask = text_attention_mask.to(self.device)
            text_token_type_ids = text_token_type_ids.to(self.device)
            label_input_ids = label_input_ids.to(self.device)
            label_attention_mask = label_attention_mask.to(self.device)
            label_token_type_ids = label_token_type_ids.to(self.device)

            text_embeddings = self.model.encode_text(
                text_input_ids, text_attention_mask, text_token_type_ids
            )
            label_embeddings = self.model.encode_labels(
                label_input_ids, label_attention_mask, label_token_type_ids
            )

            # compute loss
            if hasattr(self.alignment_loss_fn, 'set_targets') and hasattr(self.alignment_loss_fn, 'label_prototypes') and targets is not None:
                self.alignment_loss_fn.set_targets(targets)
                loss = self.alignment_loss_fn(text_embeddings, None)
            else:
                loss = self.alignment_loss_fn(text_embeddings, label_embeddings)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                getattr(self.args, 'max_grad_norm', 1.0)
            )
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def _create_phase2_dataloader(self, all_labels, novel_clusters_info, known_clusters_info):
        """Build Phase-2 dataloader with unique labels per batch."""
        text_label_pairs = []
        def _collect_train_samples(
            texts: List[str],
            order: Optional[np.ndarray],
            target_label: str,
            known_labels_for_style: List[str],
            llm_generator,
            header: str
        ) -> List[str]:
            if texts is None:
                return []
            if order is None:
                order_local = np.arange(len(texts))
            else:
                order_local = order
            texts_sorted_local = [texts[i] for i in order_local]

            n_local = len(texts_sorted_local)
            if n_local == 0:
                return []

            high_ratio_local = getattr(self, 'current_high_conf_ratio', getattr(self.args, 'high_conf_ratio', 0.4))
            low_ratio_local = getattr(self.args, 'low_conf_ratio', 0.2)
            h_local = int(round(n_local * high_ratio_local))
            l_local = int(round(n_local * low_ratio_local))
            h_local = max(0, min(h_local, n_local))
            l_local = max(0, min(l_local, n_local - h_local))

            high_conf_texts_local = texts_sorted_local[:h_local]
            mid_conf_texts_local = texts_sorted_local[h_local:n_local - l_local] if n_local - l_local > h_local else []
            low_conf_texts_local = texts_sorted_local[n_local - l_local:] if l_local > 0 else []

            # fixed-bucket sampling
            num_high = max(0, min(getattr(self.args, 'num_high_sample', 0), len(high_conf_texts_local)))
            num_mid = max(0, min(getattr(self.args, 'num_mid_sample', 0), len(mid_conf_texts_local)))
            num_low = max(0, min(getattr(self.args, 'num_low_sample', 0), len(low_conf_texts_local)))

            def _rand_pick(arr, k):
                if k <= 0 or not arr:
                    return []
                if k >= len(arr):
                    return list(arr)
                idxs = np.random.choice(len(arr), size=k, replace=False)
                return [arr[i] for i in idxs]

            picked_high = _rand_pick(high_conf_texts_local, num_high)
            picked_mid = _rand_pick(mid_conf_texts_local, num_mid)
            picked_low = _rand_pick(low_conf_texts_local, num_low)
            candidate_texts = picked_high + picked_mid + picked_low

            selected_idxs_local = []
            if llm_generator and candidate_texts:
                try:
                    selected_idxs_local = llm_generator.select_indices_for_known_label(
                        candidate_texts,
                        target_label,
                        known_labels_for_style
                    )
                except Exception:
                    selected_idxs_local = []

            filtered_texts_local = []
            for i_local in (selected_idxs_local or []):
                if isinstance(i_local, int) and 1 <= i_local <= len(candidate_texts):
                    filtered_texts_local.append(candidate_texts[i_local - 1])

            try:
                logger.info(f"{header} -> label: {target_label}, bucket_sampling: high_s={len(picked_high)}, mid_s={len(picked_mid)}, low_s={len(picked_low)}, selected_k={len(filtered_texts_local)}")
                for pt in candidate_texts[:2]:
                    _pt = pt[:120].replace('\n', ' ')
                    logger.info(f"[DEBUG]  sample: {_pt}")
            except Exception:
                pass

            return filtered_texts_local

        # add known-labeled data if enabled
        if getattr(self.args, 'use_known_labeled_data'):
            for example in self.data_manager.train_labeled_examples:
                text_label_pairs.append((example.text_a, example.label))

        # add known-unlabeled data if enabled
        if getattr(self.args, 'use_known_unlabeled_data'):
            from llm_integration import create_label_generator
            llm_generator_known = create_label_generator()

            for cluster_id, cluster_info in known_clusters_info.items():
                assigned_label = cluster_info['assigned_label']
                texts = cluster_info['texts']
                order = cluster_info.get('sorted_indices_by_conf')
                if order is None:
                    order = np.arange(len(texts))

                # use only original known labels as LLM reference
                labels_for_style = [lab for lab in self.data_manager.known_labels if lab != assigned_label]
                filtered_texts = _collect_train_samples(
                    texts=texts,
                    order=order,
                    target_label=assigned_label,
                    known_labels_for_style=labels_for_style,
                    llm_generator=llm_generator_known,
                    header=f"[DEBUG] KnownCluster {cluster_id}"
                )

                for text in filtered_texts:
                    text_label_pairs.append((text, assigned_label))

        # add novel-clustered data if enabled
        if getattr(self.args, 'use_novel_clustered_data'):
            from llm_integration import create_label_generator
            llm_generator_novel = create_label_generator()

            for cluster_id, cluster_info in novel_clusters_info.items():
                novel_label = cluster_info.get('assigned_label', None)
                if not novel_label:
                    raise Exception(f"Failed to generate label for cluster {cluster_id}.")
                texts = cluster_info.get('texts', [])
                order = cluster_info.get('sorted_indices_by_conf')
                if order is None:
                    order = np.arange(len(texts))

                # use only original known labels as LLM reference
                labels_for_style_novel = [lab for lab in self.data_manager.known_labels if lab != novel_label]
                filtered_texts = _collect_train_samples(
                    texts=texts,
                    order=order,
                    target_label=novel_label,
                    known_labels_for_style=labels_for_style_novel,
                    llm_generator=llm_generator_novel,
                    header=f"[DEBUG] NovelCluster {cluster_id}"
                )

                for text in filtered_texts:
                    text_label_pairs.append((text, novel_label))

        # debug stats
        try:
            logger.info(f"[DEBUG] Phase2 pairs: total={len(text_label_pairs)}, labels={len(set([l for _, l in text_label_pairs]))}")
        except Exception:
            pass

        # build dataset
        dataset = Phase2Dataset(
            text_label_pairs,
            all_labels,
            self.data_manager.tokenizer,
            self.args.max_seq_length
        )

        # sampler ensures unique labels per batch
        sampler = UniqueLabelBatchSampler(
            dataset,
            batch_size=self.args.train_batch_size,
            drop_last=False
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=dataset.collate_fn
        )

        logger.info(f"Created Phase 2 dataloader with {len(dataset)} samples")
        return dataloader

    def _extract_unlabeled_features_and_texts(self):
        """Extract features and texts for unlabeled samples."""
        self.model.eval()
        all_features = []
        all_texts = []

        with torch.no_grad():
            for i, example in enumerate(self.data_manager.train_unlabeled_examples):
                all_texts.append(example.text_a)
                feature = self.data_manager.train_unlabeled_dataset[i]
                input_ids = feature[0].unsqueeze(0).to(self.device)
                attention_mask = feature[1].unsqueeze(0).to(self.device)
                token_type_ids = feature[2].unsqueeze(0).to(self.device)

                text_embedding = self.model.encode_text(
                    input_ids, attention_mask, token_type_ids
                )
                all_features.append(text_embedding.cpu().numpy())

        features = np.vstack(all_features)
        return features, all_texts

    def _extract_labeled_features_and_texts(self):
        """Extract features/texts/labels for labeled samples."""
        self.model.eval()
        all_features = []
        all_texts = []
        all_labels = []

        with torch.no_grad():
            for i, example in enumerate(self.data_manager.train_labeled_examples):
                all_texts.append(example.text_a)
                all_labels.append(example.label)
                feature = self.data_manager.train_labeled_dataset[i]
                input_ids = feature[0].unsqueeze(0).to(self.device)
                attention_mask = feature[1].unsqueeze(0).to(self.device)
                token_type_ids = feature[2].unsqueeze(0).to(self.device)

                text_embedding = self.model.encode_text(
                    input_ids, attention_mask, token_type_ids
                )
                all_features.append(text_embedding.cpu().numpy())

        features = np.vstack(all_features)
        return features, all_texts, all_labels

    def _create_optimizer(self, model, phase='warmup'):
        """Create optimizer."""
        if phase == 'warmup':
            lr = self.args.lr_encoder
            params = model.parameters()
        else:
            # different LR for different components
            bert_params = []
            projection_params = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'projection_head' in name:
                        projection_params.append(param)
                    else:
                        bert_params.append(param)

            param_groups = [
                {'params': bert_params, 'lr': self.args.lr_encoder},
                {'params': projection_params, 'lr': self.args.lr_proj}
            ]

            return optim.AdamW(
                param_groups,
                weight_decay=self.args.weight_decay
            )

        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.args.weight_decay
        )

    def evaluate_results(self, dataloader):
        """Evaluate clustering metrics on provided dataloader."""
        logger.info("Performing evaluation on provided dataloader...")
        features, true_labels = extract_features_and_labels(
            self.model, dataloader, self.device
        )

        # cluster count based on unique labels in this dataloader
        num_clusters = len(np.unique(true_labels))
        cluster_labels, _ = self.cluster_manager.perform_clustering(
            features, num_clusters=num_clusters
        )

        # compute metrics
        final_results = clustering_score(
            true_labels, cluster_labels,
            self.data_manager.known_label_indices
        )

        logger.info("Eval Results:")
        for metric, value in final_results.items():
            logger.info(f"  {metric}: {value}")

        return final_results

    def evaluate_dev_known_acc(self) -> Dict[str, float]:
        """Evaluate dev-set known-class zero-shot accuracy."""
        self.model.eval()
        import torch.nn.functional as F

        known_labels = list(self.data_manager.known_labels)
        label_templates = [f"the category of the text is: {lab}" for lab in known_labels]
        with torch.no_grad():
            enc = self.data_manager.tokenizer(
                label_templates,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=self.args.max_seq_length,
                return_tensors='pt'
            )
            label_input_ids = enc['input_ids'].to(self.device)
            label_attention_mask = enc['attention_mask'].to(self.device)
            label_token_type_ids = enc.get('token_type_ids', torch.zeros_like(label_input_ids)).to(self.device)

            label_emb = self.model.encode_labels(label_input_ids, label_attention_mask, label_token_type_ids)
            label_emb = F.normalize(label_emb, dim=-1)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.data_manager.dev_dataloader:
                input_ids, attention_mask, token_type_ids, labels = [t.to(self.device) for t in batch]

                text_emb = self.model.encode_text(input_ids, attention_mask, token_type_ids)
                text_emb = F.normalize(text_emb, dim=-1)

                sim = torch.matmul(text_emb, label_emb.t())
                pred_idx = torch.argmax(sim, dim=1).cpu().numpy()
                true_idx = labels.detach().cpu().numpy()

                correct += int((pred_idx == true_idx).sum())
                total += int(len(true_idx))

        acc = (correct / max(1, total)) * 100.0
        result = {'k-acc': round(acc, 2)}
        logger.info(f"Dev Known k-acc: {result['k-acc']}")
        return result

    def _print_training_summary(self):
        """Print training summary with saved files."""
        logger.info("=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)

        # check saved model files
        phase1_model = self._get_warmup_model_path()
        final_model = self._get_final_model_path()

        logger.info("Saved Files:")
        logger.info(f"   Base Directory: {self.base_save_dir}")

        if os.path.exists(phase1_model):
            logger.info(f"   Phase 1 Best Model: {phase1_model}")
        else:
            logger.info("   Phase 1 Best Model: Not found")

        if os.path.exists(final_model):
            logger.info(f"   Final Model: {final_model}")
        else:
            logger.info("   Final Model: Not found")

        logger.info("=" * 80)

    def _save_training_summary(self, final_results, training_time: Optional[float] = None) -> str:
        """Save training summary to JSON; return path."""
        import json
        from datetime import datetime

        summary = {
            "experiment_info": {
                "dataset": self.args.dataset,
                "known_cls_ratio": self.args.known_cls_ratio,
                "labeled_ratio": self.args.labeled_ratio,
                "seed": self.args.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "training_config": {
                "num_warmup_epochs": self.args.num_warmup_epochs,
                "num_train_epochs": self.args.num_train_epochs,
                "train_batch_size": self.args.train_batch_size,
                "lr_encoder": self.args.lr_encoder,
                "lr_proj": self.args.lr_proj,
                "weight_decay": self.args.weight_decay,
                # phase-2 toggles
                "use_known_labeled_data": getattr(self.args, 'use_known_labeled_data', False),
                "use_known_unlabeled_data": getattr(self.args, 'use_known_unlabeled_data', False),
                "use_novel_clustered_data": getattr(self.args, 'use_novel_clustered_data', False),
                # confidence bucketing
                "high_conf_ratio_start": getattr(self.args, 'high_conf_ratio', None),
                "high_conf_ratio_end": getattr(self.args, 'high_conf_ratio_end', None),
                "low_conf_ratio": getattr(self.args, 'low_conf_ratio', None),
                # fixed bucket sampling (always enabled)
                "num_high_sample": getattr(self.args, 'num_high_sample', 0),
                "num_mid_sample": getattr(self.args, 'num_mid_sample', 0),
                "num_low_sample": getattr(self.args, 'num_low_sample', 0),
                # label refinement (always enabled)
                "label_refine_start_epoch": getattr(self.args, 'label_refine_start_epoch', 1),
            },
            "data_info": {
                "total_labels": len(self.data_manager.all_labels),
                "known_labels": len(self.data_manager.known_labels),
                "train_labeled_samples": len(self.data_manager.train_labeled_examples),
                "train_unlabeled_samples": len(self.data_manager.train_unlabeled_examples),
            },
            "final_results": final_results,
            "training_time_sec": training_time,
            "saved_files": {
                "phase1_model": self._get_warmup_model_path(),
                "final_model": self._get_final_model_path(),
                "base_directory": self.base_save_dir,
                "models_directory": self.models_dir,
            }
        }

        summary_path = os.path.join(self.base_save_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"📋 Training summary saved to: {summary_path}")
        return summary_path
