# AI-Powered Smart Email Classifier for Enterprises

## Milestone 1: Data Collection & Preprocessing

---

## Overview

This project builds an AI-powered email classification system for enterprises. Milestone 1 focuses on collecting and cleaning email datasets for training machine learning models.

---

## What I Did

### 1. Collected Email Datasets
- **Classification Dataset**: Collected emails for 5 different categories
- **Urgency Dataset**: Collected emails with urgency priority levels

### 2. Cleaned the Data
Removed noise and normalized text:
- Removed HTML tags
- Removed URLs and email addresses
- Removed email signatures
- Converted to lowercase
- Removed extra whitespace
- Removed duplicates and empty messages

### 3. Organized Datasets
- Created separate cleaning scripts for each category
- Merged all classification data into one dataset
- Split urgency data into train/test/validation sets

---

## Categories & Labels

### Classification Dataset (5 Categories)
| Category | Description | Samples |
|----------|-------------|---------|
| Complaint | Customer complaints | 700 |
| Request | Customer requests | 700 |
| Social Media | Social media notifications | 700 |
| Spam | Spam emails | 641 |
| Promotion | Marketing/promotional emails | 574 |

**Total**: 3,315 cleaned emails

### Urgency Dataset (4 Priority Levels)
| Label | Priority Level | Description |
|-------|----------------|-------------|
| 0 | Low | Non-urgent emails |
| 1 | Medium-Low | Slightly urgent |
| 2 | Medium-High | Moderately urgent |
| 3 | High | Very urgent emails |

**Splits**:
- Training: 1,750+ emails
- Testing: 480+ emails
- Validation: 580+ emails

---

## Data Cleaning

Applied the following cleaning steps to all emails:

1. Remove HTML tags and entities
2. Remove URLs (http://, www.)
3. Remove email addresses
4. Remove email signatures
5. Remove phone numbers
6. Remove special characters
7. Lowercase all text
8. Normalize whitespace
9. Remove duplicates
10. Remove empty/very short messages

---

## Project Structure

```
Infosys/
├── Datset/
│   ├── Classification_Dataset/
│   │   ├── Raw_Dataset/              # Original data files
│   │   ├── Cleaning code/            # Cleaning scripts
│   │   └── cleaned_Dataset/          # merged_cleaned_dataset.csv
│   │
│   └── Urgency_Dataset/
│       ├── Raw_Dataset/              # train.csv, test.csv, validation.csv
│       ├── Cleaned_Dataset/          # Cleaned versions
│       └── data_cleaning.py          # Cleaning script
│
└── README.md
```

---

## Usage (Milestone 1)

**Classification Dataset Cleaning:**
```bash
cd Datset/Classification_Dataset/Cleaning\ code/
python clean_complaint.py
python clean_request.py
python clean_promotion.py
python clean_social_media.py
python clean_spam.py
python merge_cleaned_datasets.py
```

**Urgency Dataset Cleaning:**
```bash
cd Datset/Urgency_Dataset/
python data_cleaning.py
```

---

## Results (Milestone 1)

✅ **Classification Dataset**: 3,315 cleaned emails across 5 categories  
✅ **Urgency Dataset**: 2,810+ cleaned emails with 4 urgency levels  
✅ **Data Quality**: No duplicates, no missing values, all text normalized  
✅ **Code**: Modular cleaning scripts for reproducibility

---

## Milestone 2: Email Categorization Engine

---

## Overview

Milestone 2 focused on developing an NLP-based classification system to categorize emails into **Complaint**, **Request**, **Social Media**, **Spam**, and **Promotion**.

---

## What I Did

### 1. Baseline Classifiers
Implemented traditional machine learning models to establish a performance benchmark.
- **Models**: Logistic Regression, Multinomial Naive Bayes.
- **Preprocessing**: TF-IDF Vectorization (Top 5000 features).
- **Validation**:
    - **Stratified 5-Fold Cross-Validation**: Applied to ensure the model's performance is consistent across different subsets of data.
    - **Reason for Stratified CV**: Ensures each fold preserves the percentage of samples for each class, providing a statistically robust evaluation.

**Baseline Results:**
- **Logistic Regression**: ~98.0% Accuracy
- **Naive Bayes**: ~97.7% Accuracy

### 2. Transformer Fine-Tuning (DistilBERT)
Fine-tuned a pre-trained **DistilBERT** model for state-of-the-art performance.
- **Model**: `distilbert-base-uncased`
- **Method**: Fine-tuned using PyTorch with `AdamW` optimizer (Manual Loop for Windows stability).
- **Configuration**:
    - Epochs: 3
    - Batch Size: 8
    - Max Sequence Length: 128
    - Optimizer: AdamW (lr=5e-5)

**DistilBERT Results (Best Epoch):**
- **Accuracy**: **98.79%**
- **Weighted F1-Score**: **98.80%**
- **Macro F1-Score**: **98.85%**

DistilBERT outperformed the baselines, achieving nearly 99% performance across all metrics.

---

## Files Created (Milestone 2)

- `baseline_classification.py`: Script for training and evaluating Logistic Regression and Naive Bayes.
- `bert_finetuning.py`: Script for fine-tuning the DistilBERT model.
- `final_bert_model/`: Directory containing the saved fine-tuned model and tokenizer.

---

## Usage (Milestone 2)

**Run Baseline Models:**
```bash
python baseline_classification.py
```

**Run DistilBERT Training:**
```bash
python bert_finetuning.py
```

---

## Status

**Milestone 1**: ✅ Complete
**Milestone 2**: ✅ Complete
