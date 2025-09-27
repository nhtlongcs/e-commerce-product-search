# Multilingual E-commerce Product Search

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[🇻🇳 Vietnamese](README_vi.md) | [🇬🇧 English](README.md)

## 📖 Problem Overview

This repository addresses the challenge of **multilingual product search** in e-commerce platforms. In today's global marketplace, users search for products using queries in multiple languages, often with mixed scripts, informal language, and domain-specific terminology. Traditional search systems struggle with:

- **Multilingual queries**: Users search in their native language while product information might be in different languages
- **Code-mixing**: Queries mixing multiple languages (e.g., "smartphone màu đỏ" - Vietnamese + English)
- **Informal language**: Colloquial terms, abbreviations, and typos common in search queries
- **Relevance matching**: Determining if a query matches relevant product categories or specific items

### Tasks Solved

This system tackles two core problems:

1. **Query-Category (QC) Classification**: Determine if a search query is relevant to a specific product category
   - Input: Search query + Product category
   - Output: Relevance score (0-1)
   - Example: Query "smartphone" → Category "Electronics/Mobile Phones" → High relevance

2. **Query-Item (QI) Classification**: Determine if a search query matches a specific product
   - Input: Search query + Product title/description  
   - Output: Relevance score (0-1)
   - Example: Query "red iPhone" → Product "Apple iPhone 14 Red 128GB" → High relevance

## 🎯 Highlights

- **Multilingual Support**: Handles queries in multiple languages simultaneously
- **State-of-the-art LLMs**: Fine-tuned Gemma3-12B and Qwen models
- **Efficient Training**: LoRA fine-tuning with DeepSpeed for memory optimization
- **Production Ready**: Optimized inference pipeline for real-time applications

## 📊 Performance

Our models achieve state-of-the-art performance on multilingual e-commerce search (unseen records, unseen languages). Achieved the #1 in [CIKM 2025 Multilingual E-commerce Product Search Competition](https://tianchi.aliyun.com/competition/entrance/532369/rankingList).

| Task | Model | Dev F1-Score | Test F1-Score | Languages Tested |
|------|-------|----------|----------|------------------|
| QC   | Gemma3-12B | 89.56% | 89.65% | EN, FR, ES, KO, PT, JA, DE, IT, PL, AR |
| QI   | Gemma3-12B | 88.90% | 88.97% | EN, FR, ES, KO, PT, JA, DE, IT, PL, AR, TH, VN, ID |

## 🛠️ Applications in E-commerce

### Search & Discovery
- **Multilingual Search**: Enable users to search in their preferred language
- **Cross-language Matching**: Match English product descriptions with local language queries
- **Query Understanding**: Better interpret user intent from informal search terms

### Recommendation Systems  
- **Category Suggestion**: Recommend relevant categories based on user queries
- **Product Ranking**: Improve product ranking by better query-item relevance scoring
- **Personalization**: Adapt search results based on user's language preferences

### Business Intelligence
- **Search Analytics**: Analyze search patterns across different languages
- **Content Optimization**: Identify gaps in multilingual product information
- **Market Expansion**: Understand demand in different linguistic markets

## 🚀 Quick Start

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nhtlongcs/e-commerce-product-search.git
cd e-commerce-product-search

# Setup environment
uv sync
source .venv/bin/activate
```

### Checkpoint Download

- Download our final Gemma3-12B checkpoints from [gdrive](https://drive.google.com/file/d/1KxuDNLhxMKfJoC5y2d6MA5XsLTh_6J6M/view?usp=drive_link) and unzip these into `models`. In the folder `./models`, you should have the models' paths as follow:
```
./models/gemma-3-12b-pt 
./models/best-gemma-3-QC-stage-02
./models/best-gemma-3-QI-stage-02
```
### Basic Usage

#### 1. Query-Category Classification 

```python
from quickstart import predict_relevance

# Vietnamese query - automatically translated
score = predict_relevance(
    "models/best-gemma-3-QC-stage-02",
    "điện thoại thông minh",  # Vietnamese
    "Electronics > Mobile Phones", 
    task="QC"
)
print(f"Relevance: {score:.3f}")
# Output: Relevance: 0.997
```

#### 2. Query-Item Classification

```python
from quickstart import predict_relevance

# Direct prediction with model path
query = "red iPhone 128GB"
product = "Apple iPhone 14 Pro Red 128GB Unlocked"

relevance_score = predict_relevance(
    "models/best-gemma-3-QI-stage-02",
    query, product, task="QI"
)
print(f"Relevance: {relevance_score:.3f}")
# Output: Relevance: 0.956
```

#### 3. Batch Processing with Mixed Languages

```python
from quickstart import batch_predict
import pandas as pd

# Mixed language queries (Japanese, Vietnamese, etc.)
queries = ["スマートフォン", "điện thoại", "laptop gaming"]
categories = ["Electronics > Phones", "Electronics > Phones", "Computers > Laptops"]

# Batch prediction with automatic translation
scores = batch_predict(
    "models/best-gemma-3-QC-stage-02",
    queries, categories, task="QC"
)

# Create results dataframe
results = [
    {"query": q, "category": c, "score": s} 
    for q, c, s in zip(queries, categories, scores)
]
df = pd.DataFrame(results)
print(df)
# Output:
#         query             category     score
# 0  smartphone   Electronics > Phones  0.995
# 1   điện thoại   Electronics > Phones  0.998
# 2 laptop gaming  Computers > Laptops   0.975
```

#### 4. Performance Optimization (Pre-translation)

Our algorithm requires translating queries to English for best performance (check out our technical report for details). For performance-critical applications, you can pre-translate queries once and reuse them for multiple predictions:

```python
from quickstart import translate_queries, predict_relevance_pretranslated, load_model

# Pre-translate queries once for multiple predictions
queries = ["điện thoại", "máy tính", "áo thun"]
translated = translate_queries(queries)

print("Translation results:")
for orig, trans in zip(queries, translated):
    print(f"'{orig}' -> '{trans}'")
# Output:
# 'điện thoại' -> 'phone'
# 'máy tính' -> 'computer'  
# 'áo thun' -> 't-shirt'

# Load model once for multiple predictions
model, tokenizer = load_model("models/best-gemma-3-QC-stage-02")
targets = ["Electronics > Phones", "Computers > Laptops", "Fashion > Clothing"]

for orig, trans, target in zip(queries, translated, targets):
    score = predict_relevance_pretranslated(
        (model, tokenizer), orig, trans, target, task="QC"
    )
    print(f"'{orig}' -> '{target}': {score:.3f}")
# Output:
# 'điện thoại' -> 'Electronics > Phones': 0.998
# 'máy tính' -> 'Computers > Laptops': 0.987
# 'áo thun' -> 'Fashion > Clothing': 0.975
```

## 🌐 Translation Features

### Supported Functions

```python
# Standalone translation
from quickstart import translate_queries
translated = translate_queries(["điện thoại", "スマートフォン", "手机"])
# Output: ['phone', 'smartphone', 'mobile phone']
```

## 📦 Custom Model Training

### Training Requirements

#### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended: 4x 80GB+ for training)
- 32GB+ RAM for inference
- Linux

#### Dependencies
- PyTorch 2.0+
- Transformers 4.30+
- DeepSpeed (for distributed training)
- UV package manager

#### Hardware Recommendations

| Task | RAM | GPU Memory | GPUs | Training Time |
|------|-----|------------|------|---------------|
| Inference | 32GB | 32GB | 1 | - |
| Fine-tuning | 64GB | 80GB | 4 | 8-12  hours |

### Training Recipe

To train your own model, prepare your dataset in the same format as provided (`data/raw/`). Then start with data preprocessing, followed by model training. For detailed steps, refer to [REPRODUCE.md](REPRODUCE.md).

## 📋 Competition Results

This work achieved 1st place for the [**CIKM 2025 Multilingual E-commerce Product Search Competition**](https://tianchi.aliyun.com/competition/entrance/532369/rankingList).

**Team**: DcuRAGONS - Dublin City University, Ireland

**Members**:
- Thang-Long Nguyen Ho: thanglong.nguyenho27@mail.dcu.ie
- Hoang-Bao Le: bao.le2@mail.dcu.ie  
- Minh-Khoi Pham: minhkhoi.pham4@mail.dcu.ie

**Technical Report**: Available in `report/` directory

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Change master port in training scripts
export MASTER_PORT=29501
```

**Model Loading Error**
```bash
# Ensure model paths contain "gemma-3" for proper loading
mv models/my-model models/gemma-3-my-model
```

## 🙏 Acknowledgments

- Alibaba AIDC for the competition dataset
- Dublin City University for computational resources
- The open-source community for tools and libraries used
