#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_gpu_before_torch():
    """Set GPU env before importing torch."""
    gpu_id = None
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu_id' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            break
    
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES = {gpu_id} before importing torch")

setup_gpu_before_torch()
import torch

from config import get_config, validate_config
from dataloader import TLSADataManager
from trainer import TLSATrainer
from utils import set_seed, setup_logging


def main():
    """Main entry point."""
    parser = get_config()
    args = parser.parse_args()
    args = validate_config(args)
    set_seed(args.seed)
    log_file = os.path.join(args.save_results_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)
    
    logger.info("TLSA Training Configuration")

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    try:
        logger.info("Loading and preparing data...")
        data_manager = TLSADataManager(args)
        
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Total labels: {len(data_manager.all_labels)}")
        logger.info(f"Known labels: {len(data_manager.known_labels)}")
        logger.info(f"Known ratio: {args.known_cls_ratio}")
        logger.info(f"Labeled ratio: {args.labeled_ratio}")
        logger.info(f"Training samples - Labeled: {len(data_manager.train_labeled_examples)}, "
                   f"Unlabeled: {len(data_manager.train_unlabeled_examples)}")
        
        logger.info("Initializing trainer...")
        trainer = TLSATrainer(args, data_manager, device)
 
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()