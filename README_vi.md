# Tìm Kiếm Sản Phẩm Trên Sàn Thương Mại Điện Tử

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[🇻🇳 Vietnamese](README_vi.md) | [🇬🇧 English](README.md)

## 📖 Tổng Quan

Giải quyết bài toán tìm kiếm sản phẩm trong các nền tảng thương mại điện tử. Trong thị trường toàn cầu ngày nay, người dùng tìm kiếm sản phẩm bằng truy vấn ở nhiều ngôn ngữ khác nhau, ngôn ngữ không chính thức và thuật ngữ chuyên môn. Các hệ thống tìm kiếm truyền thống gặp khó khăn với:

- **Truy vấn đa ngôn ngữ**: Người dùng tìm kiếm bằng ngôn ngữ mẹ đẻ trong khi thông tin sản phẩm có thể ở ngôn ngữ khác
- **Trộn mã ngôn ngữ**: Truy vấn trộn nhiều ngôn ngữ (ví dụ: "smartphone màu đỏ" - tiếng Việt + tiếng Anh)
- **Ngôn ngữ không chính thức**: Thuật ngữ thông tục, viết tắt và lỗi chính tả phổ biến trong truy vấn tìm kiếm
- **Khớp độ liên quan**: Xác định xem truy vấn có khớp với danh mục sản phẩm liên quan hay sản phẩm cụ thể

### Các Bài Toán Được Giải Quyết

Hệ thống này giải quyết hai vấn đề cốt lõi:

1. **Phân Loại Truy Vấn-Danh Mục (QC)**: Xác định xem truy vấn tìm kiếm có liên quan đến danh mục sản phẩm cụ thể hay không
   - Đầu vào: Truy vấn tìm kiếm + Danh mục sản phẩm
   - Đầu ra: Điểm độ liên quan (0-1)
   - Ví dụ: Truy vấn "smartphone" → Danh mục "Electronics/Mobile Phones" → Độ liên quan cao

2. **Phân Loại Truy Vấn-Sản Phẩm (QI)**: Xác định xem truy vấn tìm kiếm có khớp với sản phẩm cụ thể hay không
   - Đầu vào: Truy vấn tìm kiếm + Tiêu đề/mô tả sản phẩm
   - Đầu ra: Điểm độ liên quan (0-1)
   - Ví dụ: Truy vấn "red iPhone" → Sản phẩm "Apple iPhone 14 Red 128GB" → Độ liên quan cao

## 🎯 Điểm Nổi Bật

- **Hỗ Trợ Đa Ngôn Ngữ**: Xử lý truy vấn ở nhiều ngôn ngữ đồng thời
- **LLM Tối Tiến**: Fine-tuned các mô hình Gemma3-12B và Qwen
- **Huấn Luyện Hiệu Quả**: Fine-tuning LoRA với DeepSpeed để tối ưu bộ nhớ
- **Sẵn Sàng Triển Khai**: Mô hình nhanh và hỗ trợ các hình thức triển khai theo Batch, dễ dàng tích hợp dưới dạng API.

## 📊 Hiệu Suất

Các mô hình của chúng tôi đạt hiệu suất tối ưu trong tìm kiếm thương mại điện tử đa ngôn ngữ (dữ liệu và ngôn ngữ không bao gồm trong tập train). Đạt top đầu trong [Cuộc Thi Tìm Kiếm Sản Phẩm Thương Mại Điện Tử Đa Ngôn Ngữ CIKM 2025](https://tianchi.aliyun.com/competition/entrance/532369/rankingList).

| Bài Toán | Mô Hình | Dev F1-Score | Test F1-Score | Ngôn Ngữ Kiểm Tra |
|----------|---------|----------|----------|------------------|
| QC       | Gemma3-12B | 89.56% | 89.65% | EN, FR, ES, KO, PT, JA, DE, IT, PL, AR |
| QI       | Gemma3-12B | 88.90% | 88.97% | EN, FR, ES, KO, PT, JA, DE, IT, PL, AR, TH, VN, ID |

## 🛠️ Ứng Dụng Trong Thương Mại Điện Tử

### Tìm Kiếm & Khám Phá
- **Tìm Kiếm Đa Ngôn Ngữ**: Cho phép người dùng tìm kiếm bằng ngôn ngữ ưa thích
- **Khớp Đa Ngôn Ngữ**: Khớp mô tả sản phẩm tiếng Anh với truy vấn ngôn ngữ địa phương
- **Hiểu Truy Vấn**: Diễn giải ý định người dùng tốt hơn từ thuật ngữ tìm kiếm không chính thức

### Hệ Thống Gợi Ý
- **Gợi Ý Danh Mục**: Gợi ý danh mục liên quan dựa trên truy vấn người dùng
- **Xếp Hạng Sản Phẩm**: Cải thiện xếp hạng sản phẩm bằng điểm độ liên quan truy vấn-sản phẩm tốt hơn
- **Cá Nhân Hóa**: Điều chỉnh kết quả tìm kiếm dựa trên ưu tiên ngôn ngữ của người dùng

### Thông Tin Kinh Doanh
- **Phân Tích Tìm Kiếm**: Phân tích mẫu tìm kiếm qua các ngôn ngữ khác nhau
- **Tối Ưu Nội Dung**: Xác định khoảng trống trong thông tin sản phẩm đa ngôn ngữ
- **Mở Rộng Thị Trường**: Hiểu nhu cầu trong các thị trường ngôn ngữ khác nhau

## 🚀 Hướng dẫn sử dụng

### Cài Đặt

```bash
# Cài đặt trình quản lý gói uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/your-username/e-commerce-product-search.git
cd e-commerce-product-search

# Thiết lập môi trường
uv sync
source .venv/bin/activate
```

### Tải Checkpoint

- Tải các checkpoint Gemma3-12B cuối cùng từ [gdrive](https://drive.google.com/file/d/1KxuDNLhxMKfJoC5y2d6MA5XsLTh_6J6M/view?usp=drive_link) và giải nén vào thư mục `models`. Trong thư mục `./models`, bạn nên có đường dẫn các mô hình như sau:
```
./models/gemma-3-12b-pt 
./models/best-gemma-3-QC-stage-02
./models/best-gemma-3-QI-stage-02
```

### Sử Dụng Cơ Bản

#### 1. Phân Loại Truy Vấn-Danh Mục (Query-Category)

```python
from quickstart import predict_relevance

# Truy vấn tiếng Việt - tự động dịch
score = predict_relevance(
    "models/best-gemma-3-QC-stage-02",
    "điện thoại thông minh",  # Tiếng Việt
    "Electronics > Mobile Phones", 
    task="QC"
)
print(f"Độ liên quan: {score:.3f}")
# Kết quả: Độ liên quan: 0.997
```

#### 2. Phân Loại Truy Vấn-Sản Phẩm (Query-Item Name)

```python
from quickstart import predict_relevance

# Dự đoán trực tiếp với đường dẫn mô hình
query = "red iPhone 128GB"
product = "Apple iPhone 14 Pro Red 128GB Unlocked"

relevance_score = predict_relevance(
    "models/best-gemma-3-QI-stage-02",
    query, product, task="QI"
)
print(f"Độ liên quan: {relevance_score:.3f}")
# Kết quả: Độ liên quan: 0.956
```

#### 3. Xử Lý theo Batch Với đa Ngôn Ngữ

```python
from quickstart import batch_predict
import pandas as pd

# Truy vấn ngôn ngữ hỗn hợp (Nhật, Việt, v.v.)
queries = ["スマートフォン", "điện thoại", "laptop gaming"]
categories = ["Electronics > Phones", "Electronics > Phones", "Computers > Laptops"]

# Dự đoán hàng loạt với dịch tự động
scores = batch_predict(
    "models/best-gemma-3-QC-stage-02",
    queries, categories, task="QC"
)

# Tạo dataframe kết quả
results = [
    {"query": q, "category": c, "score": s} 
    for q, c, s in zip(queries, categories, scores)
]
df = pd.DataFrame(results)
print(df)
# Kết quả:
#         query             category     score
# 0  smartphone   Electronics > Phones  0.995
# 1   điện thoại   Electronics > Phones  0.998
# 2 laptop gaming  Computers > Laptops   0.975
```

#### 4. Tối Ưu Hiệu Suất (Dịch Trước)

Thuật toán của yêu cầu dịch truy vấn sang tiếng Anh để có hiệu suất tốt nhất (xem báo cáo kỹ thuật để biết chi tiết). Với các ứng dụng quan trọng về hiệu suất, bạn có thể dịch trước truy vấn một lần và tái sử dụng cho nhiều dự đoán:

```python
from quickstart import translate_queries, predict_relevance_pretranslated, load_model

# Dịch trước truy vấn một lần cho nhiều dự đoán
queries = ["điện thoại", "máy tính", "áo thun"]
translated = translate_queries(queries)

print("Kết quả dịch:")
for orig, trans in zip(queries, translated):
    print(f"'{orig}' -> '{trans}'")
# Kết quả:
# 'điện thoại' -> 'phone'
# 'máy tính' -> 'computer'  
# 'áo thun' -> 't-shirt'

# Load mô hình một lần cho nhiều dự đoán
model, tokenizer = load_model("models/best-gemma-3-QC-stage-02")
targets = ["Electronics > Phones", "Computers > Laptops", "Fashion > Clothing"]

for orig, trans, target in zip(queries, translated, targets):
    score = predict_relevance_pretranslated(
        (model, tokenizer), orig, trans, target, task="QC"
    )
    print(f"'{orig}' -> '{target}': {score:.3f}")
# Kết quả:
# 'điện thoại' -> 'Electronics > Phones': 0.998
# 'máy tính' -> 'Computers > Laptops': 0.987
# 'áo thun' -> 'Fashion > Clothing': 0.975
```

## 🌐 Tính Năng Dịch Thuật

### Các Hàm Được Hỗ Trợ

```python
# Dịch độc lập
from quickstart import translate_queries
translated = translate_queries(["điện thoại", "スマートフォン", "手机"])
# Kết quả: ['phone', 'smartphone', 'mobile phone']
```

## 📦 Huấn Luyện Mô Hình Tùy Chỉnh

### Yêu Cầu Huấn Luyện

#### Yêu Cầu Hệ Thống
- Python 3.8+
- GPU tương thích CUDA (khuyến nghị: 4x 80GB+ cho huấn luyện)
- 32GB+ RAM cho suy luận
- Linux

#### Phụ Thuộc
- PyTorch 2.0+
- Transformers 4.30+
- DeepSpeed (cho huấn luyện phân tán)
- Trình quản lý gói UV

#### Khuyến Nghị Phần Cứng

| Bài Toán | RAM | Bộ Nhớ GPU | GPU | Thời Gian Huấn Luyện |
|----------|-----|------------|-----|----------------------|
| Inference | 32GB | 32GB | 1 | - |
| Fine-tuning | 64GB | 80GB | 4 | 8-12 giờ |

### Quy trình Huấn Luyện

Để huấn luyện mô hình của riêng bạn, chuẩn bị dataset theo định dạng tương tự như được cung cấp (`data/raw/`). Sau đó bắt đầu với tiền xử lý dữ liệu, tiếp theo là huấn luyện mô hình. Để biết các bước chi tiết, tham khảo [REPRODUCE.md](REPRODUCE.md).

## 📋 Kết Quả Cuộc Thi

Công trình này đạt vị trí thứ 1 trong [**Cuộc Thi Tìm Kiếm Sản Phẩm Thương Mại Điện Tử Đa Ngôn Ngữ CIKM 2025**](https://tianchi.aliyun.com/competition/entrance/532369/rankingList).

**Đội**: DcuRAGONS - Đại học Dublin City, Ireland

**Thành viên**:
- Thang-Long Nguyen Ho: thanglong.nguyenho27@mail.dcu.ie
- Hoang-Bao Le: bao.le2@mail.dcu.ie  
- Minh-Khoi Pham: minhkhoi.pham4@mail.dcu.ie

**Báo Cáo Kỹ Thuật**: Có sẵn trong thư mục `report/`


## Các Vấn Đề Thường Gặp

**Port Đã Được Sử Dụng**
```bash
# Thay đổi master port trong script huấn luyện
export MASTER_PORT=29501
```

**Lỗi Load Mô Hình**
```bash
# Đảm bảo đường dẫn mô hình chứa "gemma-3" cho việc load đúng
mv models/my-model models/gemma-3-my-model
```


## 🙏 Lời Cảm Ơn

- Alibaba AIDC cho dataset cuộc thi
- Đại học Dublin City cho tài nguyên tính toán
- Cộng đồng mã nguồn mở cho các công cụ và thư viện được sử dụng