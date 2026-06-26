import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description="TLSA for NLP GCD Configuration")

    # Data config
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .csv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset to train selected.",
    )

    parser.add_argument(
        "--known_cls_ratio",
        default=0.75,
        type=float,
        help="The ratio of known classes.",
    )

    parser.add_argument(
        "--labeled_ratio",
        default=0.1,
        type=float,
        help="The ratio of labeled samples in the training set.",
    )

    parser.add_argument(
        "--cluster_num_factor",
        default=1.0,
        type=float,
        help="The factor (magnification) of the number of clusters K.",
    )

    # Model config
    parser.add_argument(
        "--bert_model",
        default="./pretrained_models/bert-base-uncased",
        type=str,
        help="The path for the pre-trained bert model.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )

    parser.add_argument(
        "--feat_dim", default=768, type=int, help="The BERT feature dimension."
    )

    parser.add_argument(
        "--proj_dim", default=256, type=int, help="The projection dimension."
    )

    parser.add_argument(
        "--temperature", default=0.07, type=float, help="Temperature for InfoNCE loss."
    )

    parser.add_argument(
        "--use_mlp_projection",
        action="store_true",
        help="Use a 2-layer MLP projection head instead of linear.",
    )

    parser.add_argument(
        "--mlp_hidden_dim",
        default=768,
        type=int,
        help="Hidden dimension for the 2-layer MLP projection head (default: equal to input hidden size).",
    )

    # Training config
    parser.add_argument(
        "--num_warmup_epochs",
        default=10,
        type=int,
        help="The warm-up epochs for supervised warm-up (Phase 1).",
    )

    parser.add_argument(
        "--num_train_epochs",
        default=50,
        type=int,
        help="The training epochs for semi-supervised learning.",
    )

    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )

    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )

    parser.add_argument(
        "--lr_encoder", default=2e-5, type=float, help="The learning rate for encoder."
    )

    parser.add_argument(
        "--lr_proj",
        default=1e-3,
        type=float,
        help="The learning rate for projection head.",
    )

    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay."
    )

    # Phase-2 dataloader switches
    parser.add_argument(
        "--use_known_labeled_data",
        action="store_true",
        help="Whether to use known labeled data in phase 2 dataloader.",
    )

    parser.add_argument(
        "--use_known_unlabeled_data",
        action="store_true",
        help="Whether to use known unlabeled data (assigned by clustering) in phase 2 dataloader.",
    )

    parser.add_argument(
        "--use_novel_clustered_data",
        action="store_true",
        help="Whether to use novel clustered data in phase 2 dataloader.",
    )

    # System config
    parser.add_argument("--gpu_id", type=str, default="0", help="Select the GPU id.")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )

    # Output config
    parser.add_argument(
        "--save_results_path",
        type=str,
        default="results",
        help="The path to save results.",
    )

    parser.add_argument(
        "--save_phase2_model_at_epochs",
        type=int,
        nargs="+",
        default=[],
        help="A list of epochs (1-based) to save extra phase 2 model checkpoints. e.g., --save_phase2_model_at_epochs 10 35",
    )

    # Confidence bucket params
    parser.add_argument(
        "--high_conf_ratio",
        default=0.15,
        type=float,
        help="Top ratio treated as high confidence (auto accept)",
    )
    parser.add_argument(
        "--low_conf_ratio",
        default=0.25,
        type=float,
        help="Bottom ratio treated as low confidence (defer)",
    )
    parser.add_argument(
        "--high_conf_ratio_end",
        default=0.4,
        type=float,
        help="Target high confidence ratio at the end of phase 2 (linear schedule)",
    )
    # Fixed-bucket sampling config
    parser.add_argument(
        "--num_high_sample",
        default=8,
        type=int,
        help="Number of samples to take from high-confidence bucket",
    )
    parser.add_argument(
        "--num_mid_sample",
        default=12,
        type=int,
        help="Number of samples to take from mid-confidence bucket",
    )
    parser.add_argument(
        "--num_low_sample",
        default=16,
        type=int,
        help="Number of samples to take from low-confidence bucket",
    )

    # Loss config
    parser.add_argument(
        "--hardneg_weight",
        type=float,
        default=0.5,
        help="Weight of hard-negative penalty term for CLIP hard-negative loss.",
    )

    parser.add_argument(
        "--hardneg_topk",
        type=int,
        default=5,
        help="Top-K negatives per sample for the hard-negative penalty.",
    )

    # Label refinement start epoch
    parser.add_argument(
        "--label_refine_start_epoch",
        type=int,
        default=1,
        help="Phase-2 epoch index (1-based) from which to activate label refinement. Defaults to 1 (start immediately).",
    )

    return parser


def validate_config(args):
    """Validate and set defaults."""
    max_seq_lengths = {
        "clinc": 30,
        "stackoverflow": 45,
        "banking": 55,
        "hwu": 55,
        "mcid": 65,
        "ecdt": 65,
        "thucnews": 256,
    }

    if args.max_seq_length is None:
        args.max_seq_length = max_seq_lengths.get(args.dataset, 64)
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    return args
