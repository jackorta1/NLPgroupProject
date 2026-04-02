# YouTube Spam Comment Classifier

## NLP Group Project — Group 3

A Naive Bayes classifier that detects spam comments on YouTube videos using the Bag-of-Words model with TF-IDF weighting.

---

## Context

NLP group project: Build a Bag-of-Words + Naive Bayes spam classifier on the YouTube Spam Collection dataset (Group 3: Youtube03-LMFAO.csv). The assignment has very specific step-by-step requirements that must be followed exactly.

**Deliverable**: A well-commented Python script (`spam_classifier.py`) with inline results.

---

## Dataset

- **Source**: [UCI Machine Learning Repository — YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection)
- **File**: `Youtube03-LMFAO.csv` (LMFAO - Party Rock Anthem)
- **Size**: 438 comments (202 non-spam, 236 spam)
- **Columns used**: `CONTENT` (comment text) and `CLASS` (0 = not spam, 1 = spam)

---

## Implementation Steps

### 1. Load & Explore Data (15% of grade)
- Load `Youtube03-LMFAO.csv` into a pandas DataFrame
- Display `.head()`, `.shape`, `.info()`, `.describe()`
- Show class distribution (`value_counts` of CLASS column)
- Show sample spam vs non-spam comments
- Keep only the two relevant columns: **CONTENT** and **CLASS**

### 2. Data Pre-processing (25% of grade)
- Use `nltk` for text preparation (stopword removal)
- Apply `CountVectorizer.fit_transform()` on CONTENT column to create Bag-of-Words features
- Print the shape of the transformed data and feature names (initial features)
- Apply `TfidfTransformer` to downscale the count matrix using TF-IDF
- Print the shape of the TF-IDF transformed data (final features)

### 3. Shuffle & Split
- Shuffle using `df.sample(frac=1)` with a `random_state` for reproducibility
- Split manually with pandas (75% train / 25% test) — **NOT** using `train_test_split`
  - e.g., `train = shuffled[:split_idx]`, `test = shuffled[split_idx:]`
- Separate features (X) and labels (y) for both train and test sets

### 4. Model Training (20% of grade)
- Fit a `MultinomialNB` classifier on the training TF-IDF features
- Run 5-fold cross-validation on training data using `cross_val_score`
- Print mean cross-validation accuracy

### 5. Model Testing (20% of grade)
- Predict on test set
- Print confusion matrix (`confusion_matrix`)
- Print classification report and accuracy score

### 6. Custom Comments
- Create 6 new comments: 4 non-spam, 2 spam
- Transform them through the same `CountVectorizer` + `TfidfTransformer` pipeline
- Pass to classifier and print predictions with labels

### 7. Conclusions
- Present all results and conclusions in the PowerPoint presentation

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn nltk
```
or
```bash
py -m pip install pandas numpy scikit-learn nltk
```

### 2. Run the classifier
```bash
python spam_classifier.py
```

---

## Key Libraries

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import nltk
from nltk.corpus import stopwords
```

| Library | Purpose |
|---------|---------|
| **pandas** | Data loading and manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | CountVectorizer, TfidfTransformer, MultinomialNB, cross-validation, metrics |
| **nltk** | English stopwords list |

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | 438 comments (202 not spam, 236 spam) |
| Vocabulary Size | 856 unique words |
| 5-Fold CV Accuracy | 86.27% |
| Test Set Accuracy | **89.09%** |
| Custom Comments | 6/6 correct |

### Confusion Matrix
```
                Predicted
                Not Spam   Spam
Actual Not Spam     41        7
Actual Spam          5       57
```

### Output Screenshot
![Output Screenshot](snippetoutput.png)

---

## Project Pipeline (Summary)

| Step | Description |
|------|-------------|
| 1. Load & Explore | Load CSV, check shape, class distribution, missing values |
| 2. Pre-process | Bag-of-Words with `CountVectorizer`, then TF-IDF downscaling |
| 3. Shuffle & Split | `pandas.sample(frac=1)` shuffle, 75/25 train-test split |
| 4. Train | `MultinomialNB` classifier + 5-fold cross-validation |
| 5. Test | Confusion matrix, classification report, accuracy |
| 6. Custom Comments | 6 new comments (4 legit, 2 spam) tested on the model |

---

## Team Members

- Group 3

---
