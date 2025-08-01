"""
Prediction script - load from checkpoint and predict.
"""

import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.params import load_prediction_parameters
from lib.model import load_model_and_tokenizer
from lib.dataset import create_datasets

def sentence_pair_collate_fn(batch, sentence1_str, sentence2_str, tokenizer):
    """
    Collate function for SentencePairDataset that pads input_ids and attention_mask and batches labels.
    Args:
        batch: list of dicts with keys 'input_ids', 'attention_mask', 'label'
        tokenizer: the tokenizer to use for padding
    Returns:
        dict with padded 'input_ids', 'attention_mask', and 'labels'
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    ids = [item['id'] for item in batch]
    languages = [item['language'] for item in batch]
    sentence1 = [item[sentence1_str] for item in batch]
    sentence2 = [item[sentence2_str] for item in batch]
    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        padding=True,
        return_tensors='pt'
    )
    return {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'id': ids,
        sentence1_str: sentence1,
        sentence2_str: sentence2,
        'language': languages
    }

def run_prediction(model, tokenizer, test_dataset, data_args, batch_size, sentence1_str, sentence2_str):
    """Run prediction on test dataset and save results."""
    print("Starting prediction...")
    collator = lambda batch: sentence_pair_collate_fn(batch, sentence1_str, sentence2_str, tokenizer)
    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    # Run prediction
    model.eval()
    print(f"Saving predictions to {data_args.outputs}...")
    with open(data_args.outputs, 'w', encoding='utf-8') as f:
        for batch in tqdm(dataloader, desc="Predicting"):
            # Predict
            batch['input_ids'] = batch['input_ids'].to(model.device)
            batch['attention_mask'] = batch['attention_mask'].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            # Get predictions (0 or 1)
            predictions = torch.argmax(outputs.logits, dim=1).tolist()
            
            # Create output JSON
            for k in range(len(batch['id'])):
                output_json = {
                    "id": batch["id"][k].item(),
                    "language": batch["language"][k],
                    sentence1_str: batch[sentence1_str][k],
                    sentence2_str: batch[sentence2_str][k],
                    "prediction": predictions[k]
                }
                
                # Write to file
                f.write(json.dumps(output_json, ensure_ascii=False) + "\n")
    
    print(f"Predictions saved to {data_args.outputs}")


def main():
    """Main prediction function."""
    try:
        # Load prediction parameters (no training config needed)
        model_args, data_args = load_prediction_parameters()
        
        print("="*60)
        print(f"Starting prediction for task: {data_args.task_name}")
        print(f"Loading model from: {model_args.model_name_or_path}")
        print(f"Test file: {data_args.test_file}")
        print(f"Output file: {data_args.outputs}")
        print("="*60)
        
        # Get batch size from data_args or use default
        batch_size = getattr(data_args, 'per_device_eval_batch_size', 8)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_args)
        
        # Create datasets
        datasets, sentence1_str, sentence2_str = create_datasets(data_args, tokenizer)
        print(f"Created datasets with columns: {sentence1_str}, {sentence2_str}")
        
        # Check test dataset exists
        if 'test' not in datasets:
            raise ValueError(f"Test dataset not found. Please check test_file: {data_args.test_file}")
        
        # Run prediction
        run_prediction(
            model, tokenizer, datasets['test'], 
            data_args, batch_size, 
            sentence1_str, sentence2_str
        )
        
        print("\n" + "="*60)
        print("Prediction completed successfully!")
        print(f"Results saved to: {data_args.outputs}")
        print("="*60)
        
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
