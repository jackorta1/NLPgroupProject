"""
YouTube Spam Comment Classifier
NLP Group Project — Group 3 (LMFAO)

Building a Bag-of-Words + Naive Bayes classifier to detect spam comments on YouTube videos.
Dataset: YouTube Spam Collection (Youtube03-LMFAO.csv) from UCI Machine Learning Repository

How to run:
    python spam_classifier.py
"""

# Fix Windows terminal encoding so special characters in YouTube comments print correctly
import html
import re
import sys
import io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# pandas: for loading CSV and manipulating data in DataFrames
import pandas as pd
# CountVectorizer: converts text into a matrix of word counts (Bag of Words)
# TfidfTransformer: scales word counts by how common/rare words are across documents
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# MultinomialNB: Naive Bayes classifier, works well with word count / tf-idf features
from sklearn.naive_bayes import MultinomialNB
# Metrics to evaluate how well our model performs
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# cross_val_score: splits training data into 5 parts to validate model reliability
from sklearn.model_selection import cross_val_score
# nltk: Natural Language Toolkit — we use it for English stopwords (common words like "the", "is", "a")
import nltk
from nltk.corpus import stopwords

def load_english_stopwords():
    """Load NLTK English stopwords, downloading them once if needed."""

    try:
        return stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        return stopwords.words('english')


def clean_comment_text(text):
    """Normalize a comment by removing HTML and common encoding artifacts."""

    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("ï»¿", " ")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# ============================================================
# STEP 1: Load and Explore the Data
# Goal: Understand the dataset before building the model
# ============================================================

# Load the CSV file into a pandas DataFrame
data_path = Path(__file__).resolve().parent / 'Youtube03-LMFAO.csv'
df = pd.read_csv(data_path)

# Show the first 5 rows to see what the data looks like
print("=== First 5 Rows ===")
print(df.head())

# How many rows and columns does the dataset have?
print(f"\nDataset Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Show column names, data types, and non-null counts
print("\n=== Dataset Info ===")
df.info()

# Show basic statistics (count, mean, std, min, max) for numeric columns
print("\n=== Statistical Summary ===")
print(df.describe())

# Check if any values are missing in the dataset
print("\n=== Missing Values ===")
print(df.isnull().sum())

# How many spam vs non-spam comments are there?
# CLASS = 0 means Not Spam, CLASS = 1 means Spam
print("\n=== Class Distribution ===")
print(df['CLASS'].value_counts())
print(f"\nNot Spam (0): {(df['CLASS'] == 0).sum()} comments")
print(f"Spam (1): {(df['CLASS'] == 1).sum()} comments")

# Show a few examples of each type so we can see the difference
print("\n=== Sample Non-Spam Comments (CLASS = 0) ===")
for comment in df[df['CLASS'] == 0]['CONTENT'].head(3).values:
    print(f"  - {clean_comment_text(comment)}")

print("\n=== Sample Spam Comments (CLASS = 1) ===")
for comment in df[df['CLASS'] == 1]['CONTENT'].head(3).values:
    print(f"  - {clean_comment_text(comment)}")

# We only need two columns for this project:
#   CONTENT = the comment text (our feature / input)
#   CLASS   = spam or not spam (our label / what we predict)
df = df[['CONTENT', 'CLASS']]
df['CONTENT'] = df['CONTENT'].apply(clean_comment_text)
print(f"\nReduced DataFrame shape: {df.shape}")
print(df.head())

# ============================================================
# STEP 2: Shuffle and Split the Dataset
# Goal: Randomly mix the data, then split into training (75%) and testing (25%)
# ============================================================

# Shuffle all rows randomly so the model doesn't learn based on row order
# frac=1 means take 100% of the data (just reorder it)
# random_state=42 makes the shuffle reproducible (same result every time)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nShuffled DataFrame shape: {df_shuffled.shape}")
print(f"\nFirst 5 rows after shuffling:")
print(df_shuffled.head())

# Calculate where to split: 75% for training, 25% for testing
# 438 * 0.75 = 328 training samples, 110 testing samples
split_index = int(len(df_shuffled) * 0.75)

# Split labels now; the feature matrices are created in the preprocessing step below.
# We split using pandas indexing instead of train_test_split (as required by assignment)
y_train = df_shuffled['CLASS'][:split_index]       # Training labels
y_test = df_shuffled['CLASS'][split_index:]        # Testing labels

print(f"\nTraining set size: {len(y_train)} ({len(y_train)/len(df_shuffled)*100:.1f}%)")
print(f"Testing set size: {len(y_test)} ({len(y_test)/len(df_shuffled)*100:.1f}%)")
print(f"\nTraining labels distribution:\n{y_train.value_counts()}")
print(f"\nTesting labels distribution:\n{y_test.value_counts()}")

# ============================================================
# STEP 3: Data Pre-processing
# Goal: Convert raw text into numbers the classifier can work with
# ============================================================

# Get the list of English stopwords from NLTK
# Stopwords are common words like "the", "is", and "and" that do not help distinguish spam from non-spam
stop_words = load_english_stopwords()

# --- Bag of Words (CountVectorizer) ---
# CountVectorizer turns each comment into a vector of word counts
# Example: "I love this song" -> {I:1, love:1, this:1, song:1}
# stop_words=stop_words tells it to ignore common English words
count_vectorizer = CountVectorizer(stop_words=stop_words)

# fit_transform on the training set only: learns the vocabulary from training comments
# and transforms the training comments into count vectors
count_train = count_vectorizer.fit_transform(df_shuffled['CONTENT'][:split_index])
count_test = count_vectorizer.transform(df_shuffled['CONTENT'][split_index:])

print("\n=== Initial Features (Bag of Words) ===")
print(f"Training count matrix shape: {count_train.shape}")
print(f"Testing count matrix shape: {count_test.shape}")
print(f"Number of training documents: {count_train.shape[0]}")
print(f"Number of features (unique words): {count_train.shape[1]}")
print(f"\nFirst 20 feature names: {count_vectorizer.get_feature_names_out()[:20]}")
print(f"Total vocabulary size: {len(count_vectorizer.get_feature_names_out())}")

# --- TF-IDF (Term Frequency - Inverse Document Frequency) ---
# TF-IDF downscales words that appear in many documents (less useful)
# and upscales words that are rare (more useful for classification)
tfidf_transformer = TfidfTransformer()

# fit_transform on the training counts only, then transform the test counts with the same scaling
X_train = tfidf_transformer.fit_transform(count_train)
X_test = tfidf_transformer.transform(count_test)

print("\n=== Final Features (TF-IDF) ===")
print(f"Training TF-IDF matrix shape: {X_train.shape}")
print(f"Testing TF-IDF matrix shape: {X_test.shape}")
print(f"Number of training documents: {X_train.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Data type: {type(X_train)}")

# ============================================================
# STEP 4: Model Training
# Goal: Train Naive Bayes on training data and validate with 5-fold cross-validation
# ============================================================

# Create and train the Naive Bayes classifier
# MultinomialNB works well with word counts and TF-IDF features
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)     # The model learns patterns from training data

print("\nNaive Bayes classifier trained successfully.")

# 5-fold cross-validation: splits training data into 5 equal parts,
# trains on 4 parts and tests on 1 part, repeats 5 times
# This tells us how reliable the model is (not just lucky on one split)
cv_scores = cross_val_score(nb_classifier, X_train, y_train, cv=5)

print("\n=== 5-Fold Cross-Validation Results ===")
print(f"Individual fold scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# ============================================================
# STEP 5: Model Testing
# Goal: Test the model on unseen data (the 25% test set)
# ============================================================

# Use the trained model to predict spam/not-spam on test data
y_pred = nb_classifier.predict(X_test)

# Confusion Matrix shows:
#   - True Negatives:  correctly predicted as NOT spam
#   - False Positives: wrongly predicted as spam (was actually not spam)
#   - False Negatives: wrongly predicted as not spam (was actually spam)
#   - True Positives:  correctly predicted as spam
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives (correctly identified non-spam): {cm[0][0]}")
print(f"False Positives (non-spam classified as spam): {cm[0][1]}")
print(f"False Negatives (spam classified as non-spam): {cm[1][0]}")
print(f"True Positives (correctly identified spam): {cm[1][1]}")

# Classification report shows precision, recall, f1-score for each class
# Accuracy = total correct predictions / total predictions
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

print(f"Model Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")

# ============================================================
# STEP 6: Testing with Custom Comments
# Goal: Test the model with 6 new comments our team wrote
#        (4 non-spam, 2 spam) to see if it generalizes well
# ============================================================

# These are comments we made up (not from the dataset)
custom_comments = [
    "This song is amazing, I love the beat and the dance moves!",          # Non-spam
    "Brings back so many great memories from 2011, what a time!",          # Non-spam
    "The dance moves and choreography are absolutely legendary",            # Non-spam
    "I still listen to this song every day, it never gets old",            # Non-spam
    "Check out my channel for free iPhones and gift cards, subscribe now!", # Spam
    "Hey everyone visit my website to make money fast, click the link in my bio"  # Spam
]

expected_labels = ['Not Spam', 'Not Spam', 'Not Spam', 'Not Spam', 'Spam', 'Spam']

custom_comments = [clean_comment_text(comment) for comment in custom_comments]

# Transform custom comments through the SAME pipeline the model was trained on
# Step 1: Convert text to word counts using the SAME vocabulary
custom_counts = count_vectorizer.transform(custom_comments)
# Step 2: Apply the SAME TF-IDF scaling
custom_tfidf = tfidf_transformer.transform(custom_counts)

# Ask the model to predict each comment
custom_predictions = nb_classifier.predict(custom_tfidf)

# Display what the model predicted vs what we expected
print("\n=== Custom Comment Classification Results ===")
for i, (comment, pred, expected) in enumerate(zip(custom_comments, custom_predictions, expected_labels), 1):
    label = 'Spam' if pred == 1 else 'Not Spam'
    status = 'Correct' if label == expected else 'Incorrect'
    print(f"\nComment {i}: \"{comment}\"")
    print(f"  Expected: {expected} | Predicted: {label} | {status}")
