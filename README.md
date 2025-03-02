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

# DistilBERT for Sentiment Analysis on Amazon Reviews  

## Overview  
This project applies **DistilBERT** for sentiment classification on **Amazon product reviews**.  
The goal was to fine-tune DistilBERT on a custom dataset by mapping review ratings to sentiment labels.

## Steps Followed  

### 1️⃣ Loading the Dataset  
- Read the **Amazon reviews** CSV file using `pandas`.  
- Sampled **10,000 reviews** for processing.  

### 2️⃣ Preprocessing the Text  
- Cleaned text by **removing newlines** and unnecessary characters.  
- Mapped **ratings (1-5 stars) to sentiment labels**:  
  - **4-5 → Positive**  
  - **3 → Neutral**  
  - **1-2 → Negative**  

### 3️⃣ Splitting the Data  
- Used `train_test_split(test_size=0.2)` to create training and test sets.  
- Converted sentiment labels into numerical format:  
  - **Positive = 0**  
  - **Neutral = 1**  
  - **Negative = 2**  

### 4️⃣ Tokenizing the Text  
- Used `DistilBertTokenizer.from_pretrained("distilbert-base-uncased")`.  
- Applied **truncation, padding, and max_length=128** for efficiency.  

### 5️⃣ Fine-Tuning DistilBERT  
- Used **Hugging Face’s Trainer API** with the following parameters:  
  - **Learning Rate:** `2e-5`  
  - **Batch Size:** `16`  
  - **Epochs:** `3`  
- Training process was similar to the **SST-2** task.  

### 6️⃣ Evaluating the Model  
- Computed accuracy using the **validation set**.  
- Printed evaluation results (further analysis needed).  

## Results  
- Successfully **fine-tuned DistilBERT** on **Amazon reviews**.  
- The model effectively classified sentiment, but accuracy results were **not explicitly compared** to other benchmarks.  

