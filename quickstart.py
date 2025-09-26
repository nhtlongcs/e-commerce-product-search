"""
Quickstart module for easy prediction with pre-trained models.
Provides simple interface for Query-Category and Query-Item relevance prediction.
"""

import os
import torch
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
from src.model import load_model_and_tokenizer
from utils.nlp.translate import TranslatorWrapper


@dataclass
class SimpleModelArgs:
    """Simplified model arguments for quickstart."""
    model_name_or_path: str
    use_fast_tokenizer: bool = True
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None


def load_model(model_path: str, device: Optional[str] = None):
    """
    Load a pre-trained model for inference.
    
    Args:
        model_path (str): Path to the model directory
        device (str, optional): Device to load model on ('cuda', 'cpu', 'auto')
                               If None, automatically detects GPU availability
    
    Returns:
        tuple: (model, tokenizer) ready for inference
        
    Example:
        >>> model, tokenizer = load_model("models/best-gemma-3-QC-stage-02")
        >>> # Model is now ready for prediction
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Create model arguments
    model_args = SimpleModelArgs(
        model_name_or_path=model_path,
        use_fast_tokenizer=True,
        use_lora=False
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, num_labels=2)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer


def predict_relevance(
    model_or_path: Union[str, tuple],
    query: str,
    target: str,
    task: str = "QC",
    device: Optional[str] = None,
    translator: Optional[TranslatorWrapper] = None
) -> float:
    """
    Predict relevance between a query and target (category or item).
    
    Args:
        model_or_path: Either a path to model directory (str) or tuple of (model, tokenizer)
        query (str): Search query
        target (str): Target category (for QC) or item title/description (for QI)
        task (str): Task type - "QC" for Query-Category, "QI" for Query-Item
        device (str, optional): Device to run inference on
        translator (TranslatorWrapper, optional): Translator instance for query translation
    
    Returns:
        float: Relevance score between 0 and 1
        
    Examples:
        >>> # Using model path
        >>> score = predict_relevance(
        ...     "models/best-gemma-3-QC-stage-02",
        ...     "smartphone android", 
        ...     "Electronics > Mobile Phones",
        ...     task="QC"
        ... )
        >>> print(f"Relevance: {score:.3f}")
        
        >>> # Using loaded model
        >>> model, tokenizer = load_model("models/best-gemma-3-QI-stage-02")
        >>> score = predict_relevance(
        ...     (model, tokenizer),
        ...     "red iPhone 128GB",
        ...     "Apple iPhone 14 Pro Red 128GB Unlocked",
        ...     task="QI"
        ... )
    """
    # Handle model loading
    if isinstance(model_or_path, str):
        model, tokenizer = load_model(model_or_path, device)
    elif isinstance(model_or_path, tuple) and len(model_or_path) == 2:
        model, tokenizer = model_or_path
    else:
        raise ValueError("model_or_path must be either a string path or tuple of (model, tokenizer)")
    
    # Validate task
    if task not in ["QC", "QI"]:
        raise ValueError(f"Task must be 'QC' or 'QI', got: {task}")
    
    # Initialize translator if not provided
    if translator is None:
        translator = TranslatorWrapper()
    
    # Translate query to English
    translated_query = translator.translate([query], method="offline")[0]
    
    # Format sentence1 as: original_query - translated_query
    sentence1 = f"{query} - {translated_query}"
    sentence2 = target
    
    # Tokenize input using sentence pair format (same as dataset)
    inputs = tokenizer(
        sentence1,
        sentence2,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Apply softmax to get probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Return probability of positive class (relevance)
    relevance_score = probabilities[0][1].item()  # Index 1 for positive class
    
    return relevance_score


def batch_predict(
    model_or_path: Union[str, tuple],
    queries: List[str],
    targets: List[str], 
    task: str = "QC",
    batch_size: int = 32,
    device: Optional[str] = None,
    translator: Optional[TranslatorWrapper] = None
) -> List[float]:
    """
    Predict relevance for multiple query-target pairs in batches.
    
    Args:
        model_or_path: Either a path to model directory (str) or tuple of (model, tokenizer)
        queries (List[str]): List of search queries
        targets (List[str]): List of target categories or items
        task (str): Task type - "QC" for Query-Category, "QI" for Query-Item
        batch_size (int): Batch size for processing
        device (str, optional): Device to run inference on
        translator (TranslatorWrapper, optional): Translator instance for query translation
    
    Returns:
        List[float]: List of relevance scores
        
    Example:
        >>> queries = ["smartphone", "laptop gaming", "áo thun nam"]
        >>> targets = ["Electronics > Phones", "Computers > Laptops", "Fashion > Men's Clothing"]
        >>> scores = batch_predict(
        ...     "models/best-gemma-3-QC-stage-02",
        ...     queries, targets, task="QC"
        ... )
        >>> for q, t, s in zip(queries, targets, scores):
        ...     print(f"'{q}' -> '{t}': {s:.3f}")
    """
    if len(queries) != len(targets):
        raise ValueError(f"Number of queries ({len(queries)}) must match number of targets ({len(targets)})")
    
    # Handle model loading
    if isinstance(model_or_path, str):
        model, tokenizer = load_model(model_or_path, device)
    elif isinstance(model_or_path, tuple) and len(model_or_path) == 2:
        model, tokenizer = model_or_path
    else:
        raise ValueError("model_or_path must be either a string path or tuple of (model, tokenizer)")
    
    # Validate task
    if task not in ["QC", "QI"]:
        raise ValueError(f"Task must be 'QC' or 'QI', got: {task}")
    
    # Initialize translator if not provided
    if translator is None:
        translator = TranslatorWrapper()
    
    # Translate all queries to English in one batch (more efficient)
    print("Translating queries...")
    translated_queries = translator.translate(queries, method="offline")
    
    all_scores = []
    model.eval()
    
    # Process in batches
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_translated = translated_queries[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]
        
        # Format sentence1 as: original_query - translated_query for each pair
        sentence1_list = [f"{q} - {tq}" for q, tq in zip(batch_queries, batch_translated)]
        sentence2_list = batch_targets
        
        # Tokenize batch using sentence pair format
        inputs = tokenizer(
            sentence1_list,
            sentence2_list,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Extract relevance scores (positive class probability)
        batch_scores = probabilities[:, 1].cpu().tolist()
        all_scores.extend(batch_scores)
    
    return all_scores


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a model without fully loading it.
    
    Args:
        model_path (str): Path to the model directory
        
    Returns:
        dict: Model information including config, tokenizer info, etc.
    """
    import json
    from transformers import AutoConfig, AutoTokenizer
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    info = {"model_path": model_path}
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(model_path)
        info["model_type"] = config.model_type
        info["num_labels"] = getattr(config, 'num_labels', None)
        info["max_position_embeddings"] = getattr(config, 'max_position_embeddings', None)
        info["vocab_size"] = getattr(config, 'vocab_size', None)
    except Exception as e:
        info["config_error"] = str(e)
    
    try:
        # Check for LoRA
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            info["has_lora"] = True
            info["lora_r"] = adapter_config.get("r")
            info["lora_alpha"] = adapter_config.get("lora_alpha")
            info["target_modules"] = adapter_config.get("target_modules")
        else:
            info["has_lora"] = False
    except Exception as e:
        info["lora_error"] = str(e)
    
    try:
        # Tokenizer info
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        info["tokenizer_type"] = type(tokenizer).__name__
        info["vocab_size_tokenizer"] = len(tokenizer)
        info["pad_token"] = tokenizer.pad_token
        info["eos_token"] = tokenizer.eos_token
    except Exception as e:
        info["tokenizer_error"] = str(e)
    
    return info


def translate_queries(queries: List[str], translator: Optional[TranslatorWrapper] = None) -> List[str]:
    """
    Pre-translate a list of queries to English.
    
    Args:
        queries (List[str]): List of queries to translate
        translator (TranslatorWrapper, optional): Translator instance
        
    Returns:
        List[str]: List of translated queries
        
    Example:
        >>> queries = ["smartphone", "điện thoại", "스마트폰"]
        >>> translated = translate_queries(queries)
        >>> print(translated)
        ['smartphone', 'phone', 'smartphone']
    """
    if translator is None:
        translator = TranslatorWrapper()
    
    return translator.translate(queries, method="offline")


def predict_relevance_pretranslated(
    model_or_path: Union[str, tuple],
    query: str,
    translated_query: str,
    target: str,
    task: str = "QC",
    device: Optional[str] = None
) -> float:
    """
    Predict relevance with pre-translated query (for performance optimization).
    
    Args:
        model_or_path: Either a path to model directory (str) or tuple of (model, tokenizer)
        query (str): Original query
        translated_query (str): Pre-translated query
        target (str): Target category or item
        task (str): Task type - "QC" for Query-Category, "QI" for Query-Item
        device (str, optional): Device to run inference on
    
    Returns:
        float: Relevance score between 0 and 1
    """
    # Handle model loading
    if isinstance(model_or_path, str):
        model, tokenizer = load_model(model_or_path, device)
    elif isinstance(model_or_path, tuple) and len(model_or_path) == 2:
        model, tokenizer = model_or_path
    else:
        raise ValueError("model_or_path must be either a string path or tuple of (model, tokenizer)")
    
    # Validate task
    if task not in ["QC", "QI"]:
        raise ValueError(f"Task must be 'QC' or 'QI', got: {task}")
    
    # Format sentence1 as: original_query - translated_query
    sentence1 = f"{query} - {translated_query}"
    sentence2 = target
    
    # Tokenize input using sentence pair format (same as dataset)
    inputs = tokenizer(
        sentence1,
        sentence2,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Apply softmax to get probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Return probability of positive class (relevance)
    relevance_score = probabilities[0][1].item()  # Index 1 for positive class
    
    return relevance_score


# Convenience functions matching README examples
def load_qc_model(model_path: str = "models/best-gemma-3-QC-stage-02"):
    """Load Query-Category model."""
    return load_model(model_path)


def load_qi_model(model_path: str = "models/best-gemma-3-QI-stage-02"):
    """Load Query-Item model.""" 
    return load_model(model_path)


if __name__ == "__main__":
    # Example usage
    print("🚀 E-commerce Product Search Quickstart Demo")
    print("=" * 50)
    
    # Example 1: Query-Category Classification
    print("\n1. Query-Category Classification Example:")
    try:
        score = predict_relevance(
            "models/best-gemma-3-QC-stage-02",
            "smartphone android",
            "Electronics > Mobile Phones", 
            task="QC"
        )
        print(f"Query: 'smartphone android'")
        print(f"Category: 'Electronics > Mobile Phones'")
        print(f"Relevance: {score:.3f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model path exists!")
    
    # Example 2: Query-Item Classification  
    print("\n2. Query-Item Classification Example:")
    try:
        score = predict_relevance(
            "models/best-gemma-3-QI-stage-02", 
            "red iPhone 128GB",
            "Apple iPhone 14 Pro Red 128GB Unlocked",
            task="QI"
        )
        print(f"Query: 'red iPhone 128GB'")
        print(f"Product: 'Apple iPhone 14 Pro Red 128GB Unlocked'")
        print(f"Relevance: {score:.3f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model path exists!")
    
    # Example 3: Vietnamese Query Example
    print("\n3. Vietnamese Query Example:")
    try:
        score = predict_relevance(
            "models/best-gemma-3-QC-stage-02",
            "điện thoại thông minh",  # Vietnamese query
            "Electronics > Mobile Phones", 
            task="QC"
        )
        print(f"Query: 'điện thoại thông minh' (Vietnamese)")
        print(f"Category: 'Electronics > Mobile Phones'")
        print(f"Relevance: {score:.3f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model path exists!")
    
    # Example 4: Batch Processing
    print("\n4. Batch Processing Example:")
    try:
        queries = ["smartphone", "laptop gaming", "áo thun nam"]  # Mixed languages
        categories = ["Electronics > Phones", "Computers > Laptops", "Fashion > Men's Clothing"]
        
        scores = batch_predict(
            "models/best-gemma-3-QC-stage-02",
            queries, categories, task="QC", batch_size=2
        )
        
        print("Batch predictions:")
        for q, c, s in zip(queries, categories, scores):
            print(f"  '{q}' -> '{c}': {s:.3f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model path exists!")