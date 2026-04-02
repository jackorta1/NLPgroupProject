YouTube Spam Classifier — Implementation Plan
                                                                                                                        Context
                                                                                                                   
     NLP group project: Build a Bag-of-Words + Naive Bayes spam classifier on the YouTube Spam Collection dataset  
     (Group 3: Youtube03-LMFAO.csv). The assignment has very specific step-by-step requirements that must be
     followed exactly.

     Deliverable

     A single Jupyter Notebook (spam_classifier.ipynb) with well-commented code and inline results.

     Implementation Steps

     1. Load & Explore Data (15% of grade)

     - Load Youtube03-LMFAO.csv into a pandas DataFrame
     - Display .head(), .shape, .info(), .describe()
     - Show class distribution (value_counts of CLASS column)
     - Show sample spam vs non-spam comments
     - Keep only the two relevant columns: CONTENT and CLASS

     2. Data Pre-processing (25% of grade)

     - Use nltk for text preparation (tokenization, stopword removal, etc.)
     - Apply CountVectorizer.fit_transform() on CONTENT column to create Bag-of-Words features
     - Print the shape of the transformed data and feature names (initial features)
     - Apply TfidfTransformer to downscale the count matrix using TF-IDF
     - Print the shape of the TF-IDF transformed data (final features)

     3. Shuffle & Split

     - Shuffle using df.sample(frac=1) with a random_state for reproducibility
     - Split manually with pandas (75% train / 25% test) — NOT using train_test_split
       - e.g., train = shuffled[:split_idx], test = shuffled[split_idx:]
     - Separate features (X) and labels (y) for both train and test sets

     4. Model Training (20% of grade)

     - Fit a MultinomialNB classifier on the training TF-IDF features
     - Run 5-fold cross-validation on training data using cross_val_score
     - Print mean cross-validation accuracy

     5. Model Testing (20% of grade)

     - Predict on test set
     - Print confusion matrix (confusion_matrix)
     - Print classification report and accuracy score

     6. Custom Comments

     - Create 6 new comments: 4 non-spam, 2 spam
     - Transform them through the same CountVectorizer + TfidfTransformer pipeline
     - Pass to classifier and print predictions with labels

     7. Conclusions

    

     Key Libraries

     import pandas as pd
     import numpy as np
     from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
     from sklearn.naive_bayes import MultinomialNB
     from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
     from sklearn.model_selection import cross_val_score
     import nltk
     from nltk.corpus import stopwords

python spam_classifier.py

      Results

  ┌─────────────────────────┬───────────────────────────────────────┐
  │         Metric          │                 Value                 │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Dataset                 │ 438 comments (202 not spam, 236 spam) │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Vocabulary size         │ 856 unique words                      │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ 5-Fold CV Mean Accuracy │ 86.27%                                │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Test Set Accuracy       │ 89.09%                                │
  ├─────────────────────────┼───────────────────────────────────────┤
  │ Custom Comments         │ 5/6 correct                           │
  └─────────────────────────┴───────────────────────────────────────┘

  Confusion Matrix (test set):
  - True Negatives: 41 | False Positives: 7
  - False Negatives: 5 | True Positives: 57

  The notebook spam_classifier.ipynb is ready in your project folder. You can open it in Jupyter Notebook or VS    
  Code to run it interactively. One custom comment (#3 — "The choreography in this video is absolutely legendary") 
  was misclassified as spam, likely because "video" appears frequently in spam comments — you may want to tweak    
  that comment for your presentation if you'd like all 6 to be correct.


    
