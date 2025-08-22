---
title: Heart Disease Baseline
emoji: ❤️
colorFrom: red
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

## Demo
👉 [Try on Hugging Face Spaces](https://huggingface.co/spaces/binhvv9999/heart-baseline)


# Heart Disease Baseline (Streamlit Demo)

Baseline Logistic Regression cho dự đoán nguy cơ bệnh tim.  
App hiển thị Accuracy + F1 và cho phép nhập thông số để dự đoán rủi ro.


# AI Baseline Project – Heart Disease Prediction

## Problem
Dự án này xây dựng mô hình dự đoán nguy cơ mắc bệnh tim dựa trên các chỉ số y tế cơ bản. Mục tiêu: tạo baseline model hỗ trợ bác sĩ sàng lọc bệnh nhân có nguy cơ cao, giúp cải thiện việc phòng ngừa và điều trị sớm.

## Dataset
- Nguồn: [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Mô tả: 303 mẫu, 14 features (tuổi, giới tính, huyết áp, cholesterol, ECG, nhịp tim, v.v.).  
- Target: `target` (1 = có bệnh tim, 0 = không).

## Baseline Plan (Tuần 1)
- EDA cơ bản: kích thước dữ liệu, missing values, quick plots.
- Train baseline model (Logistic Regression hoặc Random Forest).
- Lưu kết quả metrics (Accuracy, F1-score) trong `results/`.

## Baseline Results

- **Model**: Logistic Regression  
- **Accuracy**: 0.8341463414634146  
- **F1 Score**: 0.8508771929824561  
-  Confusion Matrix:
 [[74 26]
 [ 8 97]]
  
- Metrics file: [`results/metrics.json`](results/metrics.json)


## Repo Structure
data/ # raw/processed (không commit dữ liệu nặng)
notebooks/ # EDA, experiments
src/ # code tái sử dụng
results/ # metrics, charts

## How to Run
python3 -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt  
jupyter lab  # hoặc jupyter notebook
## Dataset
- Link Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset  
- Mô tả: ~303 hàng, 14 cột (age, sex, trestbps, chol, thalach, cp, thal, v.v.).  
- Target: `target` (0 = không bệnh tim, 1 = có bệnh tim).  
- Dữ liệu nhỏ gọn → phù hợp train baseline nhanh.

