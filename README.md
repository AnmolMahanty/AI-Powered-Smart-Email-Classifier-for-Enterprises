# AI-Powered Smart Email Classifier for Enterprises

Live Link: https://enterprises-email-classifier.onrender.com/

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


---

## Milestone 3: Urgency Detection & Scoring

---

## Overview

Milestone 3 focused on implementing a **Hybrid Urgency Detection System** to classify emails into priority levels (Low, Medium, High). The system prioritizes "Safety" using strict rules while leveraging an ML model for general context, ensuring critical issues are never missed.

## What I Did

### 1. Hybrid Architecture Implementation
Developed a "Confidence-Aware" Hybrid System:
- **Rule-Based Engine (`rule_based_urgency.py`)**: Detects high-risk keywords (e.g., "system down", "security breach") to force **High Urgency**.
- **ML Model (`urgency_model_training.py`)**: Fine-tuned **DistilBERT** on a 3-class dataset to handle general context.
- **Inference Logic (`hybrid_inference.py`)**:
    1.  **Critical Override**: Rule matches -> High.
    2.  **High Confidence ML**: Model > 85% confidence -> Trust Model.
    3.  **Safety Fallback**: Model uncertain & Medium keyword match -> Medium.

### 2. Model Training
- **Algorithm**: DistilBERT (`distilbert-base-uncased`)
- **Dataset**: `merged_3class_urgency.csv` (Mapped: 0=Low, 1=Medium, 2=High)
- **Results**:
    - **Accuracy**: **92.31%**
    - **F1 Score (Weighted)**: **92.31%**
    - **Validation Loss**: **0.32**
    - **Confusion Matrix**:
      ```text
      [[46  0  0]   <- Low (Predicted perfectly)
       [ 0 62  7]   <- Medium
       [ 0  6 48]]  <- High
      ```

### 3. Verification
Validated specific test cases:
- "System is down" -> **High** (Rule Override) ✅
- "Newsletter subscription" -> **Low** (ML Prediction) ✅
- "Help with issue" -> **Medium** (Rule Fallback) ✅

## Files Created (Milestone 3)
- `urgency_model_training.py`: Script to fine-tune DistilBERT for urgency.
- `rule_based_urgency.py`: Module defining critical/medium regex rules.
- `hybrid_inference.py`: Production-ready inference script combining ML + Rules.
- `final_urgency_model/`: Directory containing the saved model.

## Usage (Milestone 3)

**Train the Model:**
```bash
python urgency_model_training.py
```

**Run Hybrid Inference:**
```bash
python hybrid_inference.py
```

---

---

## Milestone 4: Dashboard & Deployment

---

## Overview

Milestone 4 focused on delivering an enterprise-ready solution by building a comprehensive **React Frontend Dashboard**, integrating it with a **FastAPI Backend**, and containerizing the entire application for deployment.

## What I Did

### 1. Interactive Dashboard (React + Material UI)
Built a responsive, multi-page web application:
- **Analysis Page**: Real-time email classification and urgency scoring for user input.
- **History Page**: Searchable table of past analyses with filters (Category, Urgency, Date) and generic CSV export.
- **Analytics Page**: Visual dashboard with charts (Pie, Bar) showing urgency distribution and category trends.
- **Split-View Design**: Optimized layouts for full-screen utilization on desktop while remaining mobile-responsive.

### 2. Backend Integration (FastAPI)
Developed a robust Python backend to serve models and data:
- **API Endpoints**: `/analyze` for inference, `/history` for data retrieval.
- **Data Persistence**: Local storage implementation for history tracking.
- **CORS Support**: Configured for seamless frontend-backend communication.

### 3. Split-Stack Deployment Architecture
Implemented a modern "Split Deployment" strategy to leverage the best usage:
- **Backend (Hugging Face Spaces)**: Hosts the Dockerized FastAPI application + Models (High RAM availability).
- **Frontend (Render)**: Hosts the React application as a global Static Site (CDN performance).
- **Interconnectivity**: Configured `VITE_API_URL` environment variables to enable secure Cross-Origin Resource Sharing (CORS) between the two distinct platforms.

## Files Created (Milestone 4)
- `frontend/`: Complete React project source code.
- `app/main.py`: FastAPI backend application.
- `Dockerfile`: Production-ready container configuration.
- `requirements.txt`: Python dependencies.

## Usage (Milestone 4)

**Run via Docker (Recommended):**
```bash
docker build -t email-classifier .
docker run -p 7860:7860 email-classifier
```

**Run Locally:**
```bash
# Terminal 1: Backend
uvicorn app.main:app --reload --port 7860

# Terminal 2: Frontend
cd frontend
npm run dev
```

---


## Status

**Milestone 1**: ✅ Complete
**Milestone 2**: ✅ Complete
**Milestone 3**: ✅ Complete
**Milestone 4**: ✅ Complete

## 🌟 Extra Features Implemented

Beyond the core requirements, I implemented several advanced features to enhance usability, explainability, and enterprise readiness.

### 1. 🌓 Dark / Light Mode System
- **What**: A fully persistent theme toggle (Sun/Moon icon).
- **Why**: Reduces eye strain for operators working night shifts and provides a modern, premium user experience.
- **Tech**: React Context API, LocalStorage persistence, Material-UI theming.

### 2. 🧠 XAI (Explainable AI) Wrapper
- **What**: Added an "Explainability" layer to model predictions.
- **Why**: Enterprise users need to know *why* an email was marked "High Urgency".
- **Tech**: Integrated rule-based keyword highlighting and confidence score visualization to build trust in AI decisions.

### 3. 🛡️ Hybrid Safety Net
- **What**: A failsafe system that overrides ML predictions for critical keywords (e.g., "system down").
- **Why**: Ensures 100% recall on critical incidents where ML might statistically faulter.
- **Tech**: Regex-based 'Critical' and 'Medium' rule sets in `rule_based_urgency.py`.

### 4. 📊 Advanced Analytics Dashboard
- **What**: A dedicated "Analytics" tab with a Split-View layout.
- **Why**: Managers need high-level insights (e.g., "How many complaints today?").
- **Tech**: Recharts library for data visualization, responsive Grid/Flexbox layouts.

### 5. ⚡ Toast Notification System
- **What**: Non-intrusive popup alerts for actions (e.g., "Analysis Complete", "Copied to Clipboard").
- **Why**: Improves user feedback loops without blocking the UI.
- **Tech**: Custom React "Toast" component.

### 6. 📄 PDF Report Generation
- **What**: One-click export of History tables and Analytics insights into professional PDF documents.
- **Why**: Facilitates offline reporting and archiving of email analysis data for upper management.
- **Tech**: `jspdf` and `jspdf-autotable` client-side generation.

### 7. ☁️ Cloud-Native Split Architecture
- **What**: A decoupled deployment strategy hosting the Backend on **Hugging Face Spaces** (Docker) and Frontend on **Render** (Static Site).
- **Why**: Maximizes performance and resources by using specialized platforms for each layer (High RAM for ML, Global CDN for UI) while remaining 100% free.
- **Tech**: Docker API-only builds, Environment Variable configuration (`VITE_API_URL`), CORS policy management.



---

## Conclusion

The **AI-Powered Smart Email Classifier** is now a complete, end-to-end solution. It successfully combines state-of-the-art NLP (DistilBERT) with practical enterprise requirements (Rule-based Safety, Dashboarding), delivered in a deployment-ready container.


