import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from transformers import BertTokenizer
import csv
import sys
import random
from typing import List, Dict, Tuple, Optional
import logging
from utils import set_seed, create_label_template

logger = logging.getLogger(__name__)

class InputExample:
    """Input example for a single train/test sample."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures:
    """Features for a single sample."""
    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

class TLSADataset(Dataset):
    """TLSA dataset."""
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return (
            torch.tensor(feature.input_ids, dtype=torch.long),
            torch.tensor(feature.attention_mask, dtype=torch.long),
            torch.tensor(feature.token_type_ids, dtype=torch.long),
            torch.tensor(feature.label_id, dtype=torch.long)
        )

class DatasetProcessor:
    """Generic dataset processor."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Read TSV file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_examples(self, data_dir, mode):
        """Get examples."""
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """Get all labels."""
        train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(train_df['label']))
        return labels

    def _create_examples(self, lines, set_type):
        """Create input examples."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = line[1] if len(line) > 1 else None
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 is_label_template=False):
    """Convert examples to features."""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info(f"Converting example {ex_index} of {len(examples)}")

        text = example.text_a
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']

        if example.label is None:
            label_id = -1
        else:
            if example.label not in label_map:
                logger.warning(f"Label {example.label} not in label_map")
                label_id = -1
            else:
                label_id = label_map[example.label]

        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_id=label_id
        ))

    return features

class TLSADataManager:
    """TLSA data manager."""

    def __init__(self, args):
        set_seed(args.seed)

        self.args = args
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        self.processor = DatasetProcessor()
        self.all_labels = self.processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_labels) * args.known_cls_ratio)
        self.known_labels = self._load_known_labels()
        self.known_label_indices = [int(np.where(self.all_labels == label)[0])
                                    for label in self.known_labels]

        self.known_train_samples = self._load_known_train_samples()
        self.num_labels = int(len(self.all_labels) * args.cluster_num_factor)
        self._prepare_data()

        logger.info(f"Data prepared: {len(self.train_labeled_examples)} labeled, "
                   f"{len(self.train_unlabeled_examples)} unlabeled samples")

    def _load_known_labels(self):
        """Load known label list."""
        label_file = os.path.join(self.data_dir, f"label/label_{self.args.known_cls_ratio}.list")
        if os.path.exists(label_file):
            known_labels = pd.read_csv(label_file, header=None)[0].tolist()
        else:
            logger.warning(f"Known label file {label_file} not found. Randomly selecting known labels.")
            known_labels = np.random.choice(self.all_labels, self.n_known_cls, replace=False).tolist()

        return known_labels

    def _load_known_train_samples(self):
        """Load known training samples."""
        sample_file = os.path.join(self.data_dir, f"labeled_data/train_{self.args.labeled_ratio}.tsv")
        if os.path.exists(sample_file):
            known_train_samples = pd.read_csv(sample_file, sep='\t')
            known_train_samples = known_train_samples[
                known_train_samples['label'].isin(self.known_labels)
            ]
        else:
            logger.warning(f"Known train sample file {sample_file} not found. "
                          "Will extract from training set.")
            known_train_samples = pd.DataFrame()

        return known_train_samples

    def _prepare_data(self):
        """Prepare train/dev/test data."""
        train_examples = self.processor.get_examples(self.data_dir, 'train')
        dev_examples = self.processor.get_examples(self.data_dir, 'dev')
        test_examples = self.processor.get_examples(self.data_dir, 'test')

        self.train_labeled_examples, self.train_unlabeled_examples = self._split_train_examples(train_examples)
        self.dev_examples = [ex for ex in (dev_examples or []) if ex.label in self.known_labels]
        self.test_examples = test_examples or []
        self._create_dataloaders()
        self._create_label_templates()

    def _split_train_examples(self, train_examples):
        """Split train examples into labeled and unlabeled."""
        labeled_examples = []
        unlabeled_examples = []

        if not self.known_train_samples.empty:
            known_texts = self.known_train_samples['text'].tolist()
            known_labels_list = self.known_train_samples['label'].tolist()

            for example in train_examples:
                if (example.text_a in known_texts and
                    example.label in known_labels_list):
                    labeled_examples.append(example)
                else:
                    unlabeled_examples.append(example)
        else:
            for example in train_examples:
                if example.label in self.known_labels:
                    if random.random() < self.args.labeled_ratio:
                        labeled_examples.append(example)
                    else:
                        unlabeled_examples.append(example)
                else:
                    unlabeled_examples.append(example)

        return labeled_examples, unlabeled_examples

    def _create_dataloaders(self):
        """Create dataloaders."""
        labeled_features = convert_examples_to_features(
            self.train_labeled_examples, self.known_labels,
            self.args.max_seq_length, self.tokenizer
        )
        self.train_labeled_dataset = TLSADataset(labeled_features)
        self.train_labeled_dataloader = DataLoader(
            self.train_labeled_dataset,
            sampler=RandomSampler(self.train_labeled_dataset),
            batch_size=self.args.train_batch_size
        )

        unlabeled_features = convert_examples_to_features(
            self.train_unlabeled_examples, self.all_labels,
            self.args.max_seq_length, self.tokenizer
        )
        self.train_unlabeled_dataset = TLSADataset(unlabeled_features)

        dev_features = convert_examples_to_features(
            self.dev_examples, self.known_labels,
            self.args.max_seq_length, self.tokenizer
        )
        self.dev_dataset = TLSADataset(dev_features)
        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            sampler=SequentialSampler(self.dev_dataset),
            batch_size=self.args.eval_batch_size
        )

        test_features = convert_examples_to_features(
            self.test_examples, self.all_labels,
            self.args.max_seq_length, self.tokenizer
        )
        self.test_dataset = TLSADataset(test_features)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=self.args.eval_batch_size
        )

    def _create_label_templates(self):
        """Create label templates."""
        all_label_templates = [create_label_template(label) for label in self.all_labels]

        label_examples = [
            InputExample(guid=f"label-{i}", text_a=template, label=label)
            for i, (template, label) in enumerate(zip(all_label_templates, self.all_labels))
        ]

        self.label_features = convert_examples_to_features(
            label_examples, self.all_labels,
            self.args.max_seq_length, self.tokenizer,
            is_label_template=True
        )

        self.label_dataset = TLSADataset(self.label_features)



class Phase2Dataset(Dataset):
    """Phase-2 text-label alignment training dataset."""

    def __init__(self, text_label_pairs, all_labels, tokenizer, max_seq_length):
        self.text_label_pairs = text_label_pairs
        self.all_labels = all_labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.label_templates = {
            label: create_label_template(label) for label in all_labels
        }

    def __len__(self):
        return len(self.text_label_pairs)

    def __getitem__(self, idx):
        text, label = self.text_label_pairs[idx]
        label_template = self.label_templates[label]
        
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        label_encoding = self.tokenizer(
            label_template,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'text_token_type_ids': text_encoding.get('token_type_ids', torch.zeros_like(text_encoding['input_ids'])).squeeze(0),
            'label_input_ids': label_encoding['input_ids'].squeeze(0),
            'label_attention_mask': label_encoding['attention_mask'].squeeze(0),
            'label_token_type_ids': label_encoding.get('token_type_ids', torch.zeros_like(label_encoding['input_ids'])).squeeze(0),
            'label': label,
            'label_idx': self.label_to_idx[label]
        }

    def collate_fn(self, batch):
        """Batch collate function."""
        text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
        text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
        text_token_type_ids = torch.stack([item['text_token_type_ids'] for item in batch])

        label_input_ids = torch.stack([item['label_input_ids'] for item in batch])
        label_attention_mask = torch.stack([item['label_attention_mask'] for item in batch])
        label_token_type_ids = torch.stack([item['label_token_type_ids'] for item in batch])
        targets = torch.tensor([item['label_idx'] for item in batch], dtype=torch.long)


        return (
            text_input_ids, text_attention_mask, text_token_type_ids,
            label_input_ids, label_attention_mask, label_token_type_ids,
            targets
        )


class UniqueLabelBatchSampler:
    """Batch sampler ensuring unique labels per batch."""

    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.label_to_indices = {}
        for idx, (text, label) in enumerate(dataset.text_label_pairs):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.label_iterators = {}
        for label, indices in self.label_to_indices.items():
            random.shuffle(indices)
            self.label_iterators[label] = iter(indices)

        self.labels = list(self.label_to_indices.keys())
        random.shuffle(self.labels)

    def __iter__(self):
        """Generate batches."""
        label_idx = 0
        total_batches_generated = 0
        max_batches = self.__len__()
        for label, indices in self.label_to_indices.items():
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            self.label_iterators[label] = iter(shuffled_indices)

        while total_batches_generated < max_batches:
            batch = []
            used_labels = set()

            attempts = 0
            while len(batch) < self.batch_size and attempts < len(self.labels) * 2:
                current_label = self.labels[label_idx % len(self.labels)]
                label_idx += 1
                attempts += 1

                if current_label in used_labels:
                    continue

                try:
                    sample_idx = next(self.label_iterators[current_label])
                    batch.append(sample_idx)
                    used_labels.add(current_label)
                except StopIteration:
                    if total_batches_generated >= max_batches * 0.8:
                        continue
                    indices = self.label_to_indices[current_label]
                    random.shuffle(indices)
                    self.label_iterators[current_label] = iter(indices)
                    try:
                        sample_idx = next(self.label_iterators[current_label])
                        batch.append(sample_idx)
                        used_labels.add(current_label)
                    except StopIteration:
                        continue


            if not batch or (self.drop_last and len(batch) < self.batch_size):
                break

            yield batch
            total_batches_generated += 1

    def __len__(self):
        """Estimate batch count."""
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size