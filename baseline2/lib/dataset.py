"""
Dataset classes and data collation functions.
"""
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer,
                max_length, 
                sentence1_str, 
                sentence2_str,
                stage, fold_id = None,
                task_name=None,
                feature_columns=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.sentence1_str = sentence1_str
        self.sentence2_str = sentence2_str
        self.stage = stage
        self.fold_id = fold_id
        self.task_name = task_name
        self.feature_columns = feature_columns if feature_columns is not None else []

        df = pd.read_csv(file_path)
        if stage == "train":
            df = df[df['fold'] != fold_id]
        elif stage == "val":
            df = df[df['fold'] == fold_id]
        # no filtering for test stage   
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # Compose sentence1
        if ',' in self.sentence1_str:
            sentence1 = ' - '.join(
                [str(item[col]) for col in self.sentence1_str.split(',')]
            )
        else:
            sentence1 = str(item[self.sentence1_str])
        # Compose feature string
        feature_str = ''
        if self.feature_columns:
            feature_str = ' '.join([str(item[feat]) for feat in self.feature_columns])
        # Compose sentence2
        sentence2 = str(item[self.sentence2_str])
        # Compose input string: <task name> <sentence1> <feature1> ... [SEP] <sentence2>
        input_str = f"{self.task_name} {sentence1} {feature_str} [SEP] {sentence2}".strip()
        label = item['label']
        encoding = self.tokenizer(
            input_str,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'task_name': self.task_name,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(int(label), dtype=torch.long)
        }

class SentencePairPredictDataset(SentencePairDataset):
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if ',' in self.sentence1_str:
            sentence1 = ' - '.join(
                [str(item[col]) for col in self.sentence1_str.split(',')]
            )
        else:
            sentence1 = str(item[self.sentence1_str])
        feature_str = ''
        if self.feature_columns:
            feature_str = ' '.join([str(item[feat]) for feat in self.feature_columns])
        sentence2 = str(item[self.sentence2_str])
        input_str = f"{self.task_name} {sentence1} {feature_str} [SEP] {sentence2}".strip()
        encoding = self.tokenizer(
            input_str,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'id': item['id'],
            'language': item['language'],
            'origin_query': str(item['origin_query']),
            self.sentence1_str: sentence1,
            self.sentence2_str: sentence2,
            'input_str': input_str,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_data_collator(tokenizer):
    """Create data collator for dynamic padding."""
    return DataCollatorWithPadding(tokenizer, padding="longest")

def create_datasets(data_args, tokenizer):
    """Create train, validation, and test datasets."""
    # Support multi-task: expects data_args.task_names as a list, or single task_name
    task_names = getattr(data_args, "task_names", None)
    if task_names is None:
        task_names = [data_args.task_name]

    fold_id = getattr(data_args, "fold_id", 0)
    datasets = {}
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for task_name in task_names:
        tname, sentence1_str, feature_columns, sentence2_str = data_args.get_sentence_columns(task_name)
        # Training dataset
        if getattr(data_args, "train_file", None):
            train_datasets.append(
                SentencePairDataset(
                    data_args.train_file, tokenizer, data_args.max_seq_length,
                    sentence1_str, sentence2_str, stage="train", fold_id=fold_id,
                    task_name=tname, feature_columns=feature_columns
                )
            )
        # Validation dataset
        if getattr(data_args, "validation_file", None):
            val_datasets.append(
                SentencePairDataset(
                    data_args.validation_file, tokenizer, data_args.max_seq_length,
                    sentence1_str, sentence2_str, stage="val", fold_id=fold_id,
                    task_name=tname, feature_columns=feature_columns
                )
            )
        # Test dataset
        if getattr(data_args, "test_file", None):
            test_datasets.append(
                SentencePairPredictDataset(
                    data_args.test_file, tokenizer, data_args.max_seq_length,
                    sentence1_str, sentence2_str, stage='test', fold_id=None,
                    task_name=tname, feature_columns=feature_columns
                )
            )

    # Concatenate datasets for multi-task
    if train_datasets:
        from torch.utils.data import ConcatDataset
        datasets['train'] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    if val_datasets:
        from torch.utils.data import ConcatDataset
        datasets['validation'] = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    if test_datasets:
        from torch.utils.data import ConcatDataset
        datasets['test'] = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    # Return all columns for reference
    return datasets, task_names

def __test(filepath):
    """Test function to verify dataset loading."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    dataset = SentencePairDataset(filepath, tokenizer, max_length=128,
                                  sentence1_str='origin_query',
                                  sentence2_str='item_title',
                                  stage='train', fold_id=0)
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    collator = create_data_collator(tokenizer)
    batch = collator([dataset[0], dataset[1]])
    print(f"Batch input_ids: {batch['input_ids']}")
    print(f"Batch attention_mask: {batch['attention_mask']}")
    print(f"Batch labels: {batch['labels']}")
    print(f"Batch size: {len(batch['input_ids'])}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")

if __name__ == "__main__":
    filepath = '../data/translated/translated_train_QI_full_fold.csv'
    __test(filepath)