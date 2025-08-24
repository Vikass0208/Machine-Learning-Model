# Task-4: Machine Learning Model Implementation
# Example: Spam Email Classification

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv("spam.csv")   # make sure spam.csv is in same folder

# Step 3: Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 & 1

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 5: Feature Extraction (Convert text → numeric vectors)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test_vec)

# Step 8: Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
