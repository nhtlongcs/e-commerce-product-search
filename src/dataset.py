"""
Dataset classes and data collation functions.
"""
import random
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from utils.heuristics import remove_nan, remove_duplicates, clean_text_fields

class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer,
                max_length, 
                sentence1_str, 
                sentence2_str,
                stage, fold_id = None, 
                apply_preprocessing=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.sentence1_str = sentence1_str
        self.sentence2_str = sentence2_str
        self.stage = stage
        self.fold_id = fold_id
        self.ext = file_path.split('.')[-1]

        if self.ext == 'csv':
            df = pd.read_csv(file_path)
            if fold_id != -1:
                if stage == "train":
                    df = df[df['fold'] != fold_id]
                elif stage == "val":
                    df = df[df['fold'] == fold_id]
            # no filtering for test stage   
            self.data = df
        elif self.ext == 'txt':
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.data.append(item)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")
        print(f"FILE EXT: {self.ext}, Loaded {len(self.data)} items from {file_path}")
        self.query_instruction = """
        You are an expert e-commerce relevance evaluator. Your primary task is to assess how relevant a given product item is to a user's search query.

        You will be provided with:
        1.  **Original Query**: The user's search term in its original language.
        2.  **Translated Query**: The English translation of the user's search term.
        3.  **Product Item Details**: Information about the product, such as its title, category, and attributes.

        Based on this information, you must classify the relevance into one of two categories:

        - **Exact**: The product is a perfect or highly relevant match for the user's search query. It directly fulfills the user's intent.
        - **Irrelevant**: The product has no logical connection to the search query.

        Analyze the product details against both the original and translated queries to determine the most accurate classification.
        """

        # Apply preprocessing if requested
        if apply_preprocessing:
            # Untested after refactor
            self.data = self._apply_preprocessing(self.data)

    def __len__(self):
        return len(self.data)

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the dataframe"""
        df = clean_text_fields(df)
        df = remove_nan(df)
        df = remove_duplicates(df)

        return df

    def __getitem__(self, idx):
        if self.ext == 'csv':
            item = self.data.iloc[idx]
        elif self.ext == 'txt':
            item = self.data[idx]
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")
        if ',' in self.sentence1_str:
            sentence1 = ' - '.join(
                [str(item[col]) for col in self.sentence1_str.split(',')]
            )
        else:
            sentence1 = str(item[self.sentence1_str])
            
        sentence2 = str(item[self.sentence2_str])
        if 'label' in item:
            label = item['label']
        elif 'prediction' in item:
            label = item['prediction']
        else:
            assert False, "No label or prediction found"

        encoding = self.tokenizer(
            sentence1,
            sentence2,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(int(label), dtype=torch.long)
        }

class SentencePairPredictDataset(SentencePairDataset):
    def __getitem__(self, idx):
        if self.ext == 'csv':
            item = self.data.iloc[idx]
        elif self.ext == 'txt':
            item = self.data[idx]
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")
        if ',' in self.sentence1_str:
            sentence1 = ' - '.join(
                [str(item[col]) for col in self.sentence1_str.split(',')]
            )
        else:
            sentence1 = str(item[self.sentence1_str])
        # sentence1 = self.query_instruction + '\n' + 'Query: ' + sentence1

        # if self.sentence2_str == "category_path":
        #     sentence2_ls = str(item[self.sentence2_str]).split(',')
        #     sentence2 = ', '.join(sentence2_ls)
        # else:
        sentence2 = str(item[self.sentence2_str])
        # sentence2_ls = str(item[self.sentence2_str]).split(',')
        # sentence2 = '/'.join([sentence2_ls[0], sentence2_ls[-1]])
        encoding = self.tokenizer(
            sentence1,
            sentence2,
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_data_collator(tokenizer):
    """Create data collator for dynamic padding."""
    return DataCollatorWithPadding(tokenizer, padding="longest")

def create_datasets(data_args, tokenizer):
    """Create train, validation, and test datasets."""
    sentence1_str, sentence2_str = data_args.get_sentence_columns()
    fold_id = getattr(data_args, "fold_id", 0)

    datasets = {}

    # Training dataset
    if getattr(data_args, "train_file", None):
        datasets['train'] = SentencePairDataset(
            data_args.train_file, tokenizer, data_args.max_seq_length,
            sentence1_str, sentence2_str, stage="train", fold_id=fold_id
        )

    # Validation dataset
    if getattr(data_args, "validation_file", None):
        datasets['validation'] = SentencePairDataset(
            data_args.validation_file, tokenizer, data_args.max_seq_length,
            sentence1_str, sentence2_str, stage="val", fold_id=fold_id
        )

    # Test dataset for prediction
    if getattr(data_args, "test_file", None):
        datasets['test'] = SentencePairPredictDataset(
            data_args.test_file, tokenizer, data_args.max_seq_length,
            sentence1_str, sentence2_str, stage='test', fold_id=None
        )

    return datasets, sentence1_str, sentence2_str

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
    filepath = 'data/translated/translated_train_QI_full_fold.csv'
    __test(filepath)