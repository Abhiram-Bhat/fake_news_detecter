import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) - {'not', 'no'}  # Retain negation words

# Text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))  # Keep hyphens
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Check dataset balance
print("True News Samples:", len(true_df))
print("Fake News Samples:", len(fake_df))

# Label the data
true_df["label"] = 1  # Real
fake_df["label"] = 0  # Fake

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop unnecessary columns
df = df.drop(columns=[col for col in ['subject', 'date'] if col in df.columns], errors='ignore')

# Clean text
df["text"] = df["text"].apply(clean_text)

# Features and labels
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_vectorized = vectorizer.fit_transform(X)
print("Vocabulary Size:", len(vectorizer.vocabulary_))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train using Logistic Regression
model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n🔍 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Inspect top features
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
real_news_words = sorted(zip(coefficients, feature_names), reverse=True)[:10]
fake_news_words = sorted(zip(coefficients, feature_names))[:10]
print("Top words for Real News:", real_news_words)
print("Top words for Fake News:", fake_news_words)

# Try a custom sample
print("\n📰 Try Your Own Example:")
sample = ["""
The President of the United States held a press conference today, announcing major infrastructure reforms.
Experts suggest that these developments will significantly boost economic growth, improve employment rates,
and enhance international trade relations. Independent analysts have reviewed the plan and confirmed its feasibility.
"""]
# fake sample = ["""
# Breaking News A secret alien base was discovered under the White House last night! Sources claim extraterrestrials have been controlling global governments for decades. Eyewitnesses report seeing glowing UFOs and mysterious figures. The government denies all allegations, but insiders say the truth is being hidden!
# """]
sample_cleaned = [clean_text(text) for text in sample]
print("Cleaned Sample:", sample_cleaned[0])
sample_vectorized = vectorizer.transform(sample_cleaned)
decision_score = model.decision_function(sample_vectorized)[0]
print("Decision Score:", decision_score)
prediction = model.predict(sample_vectorized)
print("Prediction:", "✅ Real News" if prediction[0] == 1 else "❌ Fake News")