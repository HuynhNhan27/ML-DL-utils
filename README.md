# ML-DL Utility Framework 🚀

**English | [Tiếng Việt](#tiếng-việt)**

A comprehensive, modular Machine Learning & Deep Learning utility framework for end-to-end data science workflows. Designed for tabular, text, and image datasets with focus on code modularity, reproducibility, and scalability.

## Overview

This project provides reusable utilities and pipelines covering the complete ML/DL workflow:
- **Exploratory Data Analysis (EDA)** - Data understanding and visualization
- **Data Preprocessing** - Cleaning, normalization, feature engineering
- **Pipeline Construction** - Scalable data processing pipelines
- **Model Training** - Unified PyTorch training utilities for all data types
- **Hyperparameter Tuning** - Random search and cross-validation
- **Evaluation & Metrics** - Comprehensive model evaluation

## Key Features

- ✅ **Modular Architecture** - Clean, reusable utilities for different ML tasks
- ✅ **Multi-Modal Support** - Tabular, text, and image data pipelines
- ✅ **PyTorch Integration** - Modern deep learning with CNN and Transformer models
- ✅ **Reproducibility** - Seed management, checkpoint saving/loading
- ✅ **Training Utilities** - Early stopping, learning rate scheduling, progress tracking
- ✅ **Hyperparameter Optimization** - Built-in cross-validation and random search

## Project Structure

```
ML-DL-utils/
├── modules/
│   ├── tabular.py          # Tabular data: regression, classification
│   ├── image.py            # Image data: preprocessing, CNN architectures
│   ├── text.py             # Text data: preprocessing, tokenization, text generation
│   ├── models.py           # Neural network models (CNN, ResNet, MobileNet, Transformers)
│   └── dl_training.py      # PyTorch training utilities (train_epoch, eval_epoch, etc.)
├── notebooks/
│   ├── Text_classification.ipynb      # Text classification examples
|   ├── Tabular_regression.ipynb       # Tabular regression
│   ├── transformer.ipynb              # Transformer model demonstrations
│   └── ...
├── data/
└── README.md
```

## Modules

### 1. **modules/tabular.py** 📊
Utilities for tabular data (structured/numerical data)
- Classification & Regression pipelines
- Feature scaling, encoding, missing value handling
- Hyperparameter search with cross-validation

### 2. **modules/image.py** 🖼️
Image data processing and CNN models
- Image loading, preprocessing, augmentation
- CNN architecture implementations
- Classification pipelines

### 3. **modules/text.py** 📝
Natural Language Processing utilities
- Text preprocessing (tokenization, stemming, lemmatization)
- Text embeddings
- Text classification pipelines

### 4. **modules/models.py** 🧠
Deep Learning model architectures
- **CNN Models**: Custom CNNs for image classification
- **ResNet, MobileNet, DenseNet**: Transfer learning backbones
- **Transformer Models**: Self-attention mechanisms for NLP tasks

### 5. **modules/dl_training.py** 🔄
Unified PyTorch training utilities
- `set_seed()` - Reproducible training
- `EarlyStopping` - Prevent overfitting
- `train_epoch()` - Single epoch training
- `eval_epoch()` - Validation/evaluation
- `train_loop()` - Full training pipeline
- `plot_training_history()` - Visualization
- `save/load_checkpoint()` - Resume training


## Features Highlights

### Data Preprocessing Pipeline
- Automated missing value handling
- Feature scaling and normalization
- Categorical encoding (One-Hot, Label Encoding)
- Train-test split and cross-validation

### Training Utilities
- **EarlyStopping**: Stop training when validation metric plateaus
- **Learning Rate Scheduling**: Adaptive learning rate schedules
- **Checkpoint System**: Save and resume training
- **Progress Tracking**: TQDM-based epoch visualization

### Model Architectures
- Efficient CNNs with batch normalization
- Pre-trained ResNet, MobileNet, DenseNet
- Transformer-based sequence models
- Custom hybrid architectures

### Hyperparameter Optimization
- Random search with cross-validation
- GridSearch support
- Metric tracking and logging
- Best model selection



## Future Enhancements

- Additional model architectures (Vision Transformers, LSTM)
- Multi-GPU training support
- Distributed training with DDP
- Model serving and deployment utilities
- Advanced hyperparameter optimization (Bayesian, Optuna)

---

# Tiếng Việt

## Khái Quát Chung

Một framework tiện ích Machine Learning & Deep Learning toàn diện, có cấu trúc module, dùng cho quy trình data science end-to-end. Được thiết kế cho dữ liệu bảng, văn bản và ảnh với tập trung vào tính modular, tái sử dụng và khả năng mở rộng.

## Tính Năng Chính

- ✅ **Kiến Trúc Module** - Các tiện ích sạch, có thể tái sử dụng cho các tác vụ ML khác nhau
- ✅ **Hỗ Trợ Đa Phương Thức** - Pipeline cho dữ liệu bảng, văn bản và ảnh
- ✅ **Tích Hợp PyTorch** - Deep learning hiện đại với mô hình CNN và Transformer
- ✅ **Tái Lập Lại Được** - Quản lý seed, lưu/tải checkpoint
- ✅ **Tiện Ích Training** - Early stopping, schedule learning rate, theo dõi tiến độ
- ✅ **Tối Ưu Hóa Tham Số** - Có sẵn cross-validation và random search

## Cấu Trúc Thư Mục

```
ML-DL-utils/
├── modules/
│   ├── tabular.py          # Dữ liệu bảng: hồi quy, phân loại
│   ├── image.py            # Dữ liệu ảnh: xử lý, kiến trúc CNN
│   ├── text.py             # Dữ liệu văn bản: xử lý, tokenization, sinh văn bản
│   ├── models.py           # Mô hình mạng nơ-ron (CNN, ResNet, MobileNet, Transformers)
│   └── dl_training.py      # Tiện ích PyTorch training (train_epoch, eval_epoch, v.v.)
├── notebooks/
│   ├── Text_classification.ipynb      # Ví dụ phân loại văn bản
|   ├── Tabular_regression.ipynb       # Ví dụ hồi quy bảng
│   ├── transformer.ipynb              # Minh họa mô hình Transformer
│   └── ...
├── data/
└── README.md
```


## Các Module

### 1. **modules/tabular.py** 📊
Tiện ích cho dữ liệu bảng (dữ liệu có cấu trúc/số)
- Pipeline phân loại & Hồi quy
- Scaling feature, encoding, xử lý giá trị thiếu
- Tìm kiếm siêu tham số với cross-validation
- Hỗ trợ nhiều mô hình (mô hình sklearn)

### 2. **modules/image.py** 🖼️
Xử lý dữ liệu ảnh và mô hình CNN
- Tải ảnh, xử lý trước, tăng cường dữ liệu
- Triển khai kiến trúc CNN
- Pipeline phân loại ảnh

### 3. **modules/text.py** 📝
Tiện ích Xử Lý Ngôn Ngữ Tự Nhiên
- Xử lý văn bản (tokenization, stemming, lemmatization)
- Embedding văn bản
- Pipeline phân loại văn bản

### 4. **modules/models.py** 🧠
Kiến trúc mô hình Deep Learning
- **Mô Hình CNN**: CNN tùy chỉnh cho phân loại ảnh
- **ResNet, MobileNet, DenseNet**: Backbone cho transfer learning
- **Mô Hình Transformer**: Cơ chế self-attention cho tác vụ NLP

### 5. **modules/dl_training.py** 🔄
Tiện ích training PyTorch thống nhất
- `set_seed()` - Training có thể tái lập lại
- `EarlyStopping` - Ngăn chặn overfitting
- `train_epoch()` - Training một epoch
- `eval_epoch()` - Validation/evaluation
- `train_loop()` - Pipeline training đầy đủ
- `plot_training_history()` - Trực quan hóa
- `save/load_checkpoint()` - Tiếp tục training

## Tính Năng Nổi Bật

### Pipeline Xử Lý Dữ Liệu
- Xử lý giá trị thiếu tự động
- Scaling feature và chuẩn hóa
- Encoding dữ liệu phân loại (One-Hot, Label Encoding)
- Chia train-test và cross-validation

### Tiện Ích Training
- **EarlyStopping**: Dừng training khi metric validation ngừa cải thiện
- **Learning Rate Scheduling**: Điều chỉnh learning rate động
- **Hệ Thống Checkpoint**: Lưu và tiếp tục training
- **Theo Dõi Tiến Độ**: Trực quan hóa epoch dựa trên TQDM

### Kiến Trúc Mô Hình
- CNN hiệu quả với batch normalization
- ResNet, MobileNet, DenseNet được đào tạo trước
- Mô hình chuỗi dựa trên Transformer
- Kiến trúc kết hợp tùy chỉnh

### Tối Ưu Hóa Siêu Tham Số
- Random search với cross-validation
- Hỗ trợ GridSearch
- Theo dõi và ghi chép metric
- Lựa chọn mô hình tốt nhất

## Cải Tiến Trong Tương Lai

- Kiến trúc mô hình bổ sung (Vision Transformers, LSTM)
- Hỗ trợ training đa GPU
- Training phân tán với DDP
- Tiện ích serving và triển khai mô hình
- Tối ưu hóa siêu tham số nâng cao (Bayesian, Optuna)

---

**Last Updated**: 2025-03-14