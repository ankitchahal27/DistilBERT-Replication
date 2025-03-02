# DistilBERT-Replication and Sentiment Analysis of Amazon reviews

# Replicating DistilBERT on the SST-2 Dataset

## Overview  
This project replicates the results of **DistilBERT** on the **SST-2 (Stanford Sentiment Treebank-2)** dataset.  
The goal was to fine-tune DistilBERT and compare its performance with the accuracy reported in the original paper.

## Steps Followed  

### 1️⃣ Setting Up the Environment  
Installed the necessary libraries:  
- `transformers` (Hugging Face)  
- `datasets` (Hugging Face)  
- `evaluate` (Hugging Face)  
- `torch` (PyTorch)  

### 2️⃣ Loading the SST-2 Dataset  
Used Hugging Face’s `load_dataset('glue', 'sst2')` to fetch the SST-2 dataset.

### 3️⃣ Initializing the Model & Tokenizer  
- Used **DistilBertForSequenceClassification** (2 output labels: positive/negative sentiment).  
- Tokenized the dataset using **DistilBertTokenizer**.  

### 4️⃣ Preprocessing the Data  
- Applied **padding** and **truncation** to ensure uniform input size.  
- Used `.map()` to apply tokenization across all text data.  

### 5️⃣ Training the Model  
Used **Hugging Face’s Trainer API** with the following parameters:  
- **Learning Rate:** `2e-5`  
- **Batch Size:** `16`  
- **Epochs:** `3`  

### 6️⃣ Evaluating the Model  
- Used `evaluate.load("accuracy")` to compute accuracy.  
- Compared results with the paper’s reported accuracy.  

## Results  
| Metric                 | Value |
|------------------------|-------|
| **Original Paper Accuracy** | **91.3%** |
| **Our Model Accuracy**  | **~91.3%** |

This successful replication validates that the fine-tuning process was implemented correctly. 
