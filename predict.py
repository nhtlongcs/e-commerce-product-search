"""
Prediction script - load from checkpoint and predict.
"""

import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.params import load_parameters
from src.model import load_model_and_tokenizer
from src.dataset import create_datasets
from pathlib import Path
from accelerate import Accelerator

from accelerate.utils import gather_object

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
    # if ',' in sentence1_str:
    #     # hard code (origin_query, translated) -> only use origin query
    #     sentence1 = [item[sentence1_str].split('-')[0].strip() for item in batch]
    #     sentence1_str = sentence1_str.split(',')[0].strip()
    # else:
    sentence1 = [item[sentence1_str] for item in batch]
    queries = [item['origin_query'] for item in batch]
    sentence2 = [item[sentence2_str] for item in batch]
    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        padding=True,
        return_tensors='pt'
    )
    # print("COLLATE FN")
    # print(sentence1_str)
    # print(sentence1[0])
    # print('-'*10)
    # print(sentence2_str)
    # print(sentence2[0])
    return {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'id': ids,
        'origin_query': queries,
        sentence1_str: sentence1,
        sentence2_str: sentence2,
        'language': languages
    }


def run_prediction(model, tokenizer, test_dataset, data_args, batch_size, sentence1_str, sentence2_str, accelerator):
    """Run prediction on test dataset and save results."""
    collator = lambda batch: sentence_pair_collate_fn(batch, sentence1_str, sentence2_str, tokenizer)
    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    model.eval()
    model, dataloader = accelerator.prepare(model, dataloader)

    results_list = []

    # Chuyển model & dataloader sang Accelerate
    if accelerator.is_local_main_process:
        print(f"Starting predict {len(test_dataset)} examples with batch size {batch_size}...")

    for batch in tqdm(dataloader, desc="Predicting", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
        ids = batch['id']

        # Append results from this process's batch to its local list
        for i in range(len(ids)):
            results_list.append({
                "id": int(ids[i]),
                "language": batch["language"][i],
                "origin_query": batch["origin_query"][i], # hardcode
                sentence2_str: batch[sentence2_str][i],
                "prediction": int(predictions[i])
            })
                
    accelerator.wait_for_everyone()
    gathered_results = gather_object(results_list)

    # The main process writes the final, complete list of results to a file
    if accelerator.is_local_main_process:
        # Get all unique IDs from the original test dataset
        original_ids = set(test_dataset[i]['id'] for i in range(len(test_dataset)))
        
        # Create a dictionary to deduplicate and filter results by original IDs
        results_dict = {}
        for result in gathered_results:
            result_id = result['id']
            # Only keep results that correspond to original test dataset IDs
            if result_id in original_ids:
                # In case of duplicates (shouldn't happen but safety check), keep the first one
                if result_id not in results_dict:
                    results_dict[result_id] = result
        
        # Convert back to list and sort by id
        final_results = list(results_dict.values())
        final_results.sort(key=lambda x: x['id'])
        
        # Verify we have all expected results
        expected_count = len(test_dataset)
        actual_count = len(final_results)
        
        if actual_count != expected_count:
            print(f"WARNING: Expected {expected_count} results but got {actual_count}")
            missing_ids = original_ids - set(r['id'] for r in final_results)
            if missing_ids:
                print(f"Missing IDs: {sorted(missing_ids)}")
        
        print(f"Gathered {len(final_results)} results. Saving to {data_args.outputs}...")
        with open(data_args.outputs, 'w', encoding='utf-8') as f:
            for output_json in final_results:
                f.write(json.dumps(output_json, ensure_ascii=False) + "\n")

        print(f"Predictions saved to {data_args.outputs}")


def main():
    
    """Main prediction function."""
    try:
        # Load prediction parameters (no training config needed)

        model_args, data_args, training_args = load_parameters()
        accelerator = Accelerator()
        model_args.use_lora = False

        if accelerator.is_local_main_process:
            print("="*60)
            print(f"Loading model from: {model_args.model_name_or_path}")
            print(f"Using fast tokenizer: {model_args.use_fast_tokenizer}")
            print(f"Using LoRA: {model_args.use_lora}")
            print(f"LoRA target modules: {model_args.lora_target_modules}")
            print("="*60)
            print(f"Starting prediction for task: {data_args.task_name}")
            print(f"Test file: {data_args.test_file}")
            print(f"Output file: {data_args.outputs}")
            print("="*60)
            # data_args.outputs = Path(model_args.model_name_or_path).name + ".txt"
            print(f"Overwrite output file to: {data_args.outputs}")

        # Get batch size from data_args or use default
        batch_size = getattr(data_args, 'per_device_eval_batch_size', 64)
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_args)
        
        # hardcode - use fp16
        # Create datasets
        datasets, sentence1_str, sentence2_str = create_datasets(data_args, tokenizer)
        if accelerator.is_local_main_process:
            print(f"Created datasets with columns: {sentence1_str}, {sentence2_str}")
            print("Sample Dataset Item:")
            for key, value in datasets['test'][0].items():
                print(f"  {key}: {value}")

        # Check test dataset exists
        if 'test' not in datasets:
            raise ValueError(f"Test dataset not found. Please check test_file: {data_args.test_file}")
        

        # Run prediction
        run_prediction(
            model, tokenizer, datasets['test'], 
            data_args, batch_size, 
            sentence1_str, sentence2_str,
            accelerator=accelerator
        )

        if accelerator.is_local_main_process:
            print("\n" + "="*60)
            print("Prediction completed successfully!")
            print(f"Results saved to: {data_args.outputs}")
            print("="*60)
        
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"Prediction failed with error: {e}")
        raise


if __name__ == "__main__":
    from contextlib import nullcontext
    main()
