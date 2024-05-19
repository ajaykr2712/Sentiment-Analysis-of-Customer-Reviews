import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocesses the input text by removing non-alphanumeric characters,
    converting to lowercase, removing stopwords, and tokenizing.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text.
    """
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
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
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('customer_reviews.csv')

# Print the columns of the dataset for verification
print("Columns in the dataset:", df.columns)

# Check if the required columns exist in the dataset
if 'review' not in df.columns or 'sentiment' not in df.columns:
    raise KeyError("The dataset must contain 'review' and 'sentiment' columns.")

# Apply text preprocessing to the 'review' column
df['processed_review'] = df['review'].apply(preprocess_text)

# Initialize CountVectorizer to convert text to a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_review'])

# Map sentiment labels to binary values: 'positive' -> 1, 'negative' -> 0
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the sentiment labels for the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Convert the training data to an array for plotting
X_train_array = X_train.toarray()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_array[:, 0], X_train_array[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title('Scatter Plot of Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

def plot_decision_boundary(X, y, model, ax):
    """
    Plots the decision boundary of the classifier.

    Args:
        X (array-like): The input data.
        y (array-like): The labels.
        model (sklearn model): The trained model.
        ax (matplotlib axis): The axis on which to plot the decision boundary.
    """
    # Create a mesh to plot the decision boundary
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, y[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Reduce the dimensionality of the training data to 2D using PCA
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train.toarray())

# Plot the decision boundary using the reduced data
fig, ax = plt.subplots(figsize=(10, 6))
plot_decision_boundary(X_train_reduced, y_train, model, ax)
plt.title('Decision Boundary')
plt.show()
