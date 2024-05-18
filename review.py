import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
## Main Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from review import preprocess_text
from sklearn.decomposition import PCA

df = pd.read_csv('customer_reviews.csv')

print("Columns in the dataset:", df.columns)

if 'review' not in df.columns or 'sentiment' not in df.columns:
    raise KeyError("The dataset must contain 'review' and 'sentiment' columns.")

df['processed_review'] = df['review'].apply(preprocess_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_review'])

y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

X_train_array = X_train.toarray()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_array[:, 0], X_train_array[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title('Scatter Plot of Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

def plot_decision_boundary(X, y, model, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train.toarray())
fig, ax = plt.subplots(figsize=(10, 6))
plot_decision_boundary(X_train_reduced, y_train, model, ax)
plt.title('Decision Boundary')
plt.show()
