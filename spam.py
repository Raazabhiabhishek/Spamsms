import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        exit(1)

# Load the dataset
def load_data(file_path='spam.csv'):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found in the current directory.")
        exit(1)
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

# Preprocess text data
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# Train and evaluate model
def train_model(df, model_type='logistic'):
    try:
        df['processed_text'] = df['text'].apply(preprocess_text)
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Increased features, added bigrams
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Choose model
        if model_type == 'naive_bayes':
            model = MultinomialNB()
        else:
            model = LogisticRegression(class_weight='balanced', max_iter=1000)  # Handle imbalance
        
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return model, vectorizer, accuracy, precision, recall, f1, cm, X_test, y_test, y_pred
    except Exception as e:
        print(f"Error training model: {e}")
        exit(1)

# Visualize confusion matrix
def plot_confusion_matrix(cm, filename='confusion_matrix.png'):
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

# Predict new SMS
def predict_sms(model, vectorizer, text):
    try:
        processed_text = preprocess_text(text)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = model.predict(text_tfidf)[0]
        return 'Spam' if prediction == 1 else 'Ham'
    except Exception as e:
        print(f"Error predicting SMS: {e}")
        return "Unknown"

def main():
    download_nltk_resources()
    df = load_data('spam.csv')
    print("Dataset loaded. Shape:", df.shape)
    print("Label distribution:\n", df['label'].value_counts())
    
    # Train Logistic Regression model
    model, vectorizer, accuracy, precision, recall, f1, cm, X_test, y_test, y_pred = train_model(df, model_type='logistic')
    
    print("\nModel Evaluation (Logistic Regression):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(cm)
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    sample_texts = [
        "Free entry in 2 a weekly comp to win FA Cup final tickets! Text FA to 87121 now!",
        "Hey, are we still meeting for lunch tomorrow?"
    ]
    print("\nExample Predictions:")
    for text in sample_texts:
        result = predict_sms(model, vectorizer, text)
        print(f"Text: {text[:50]}... -> Predicted: {result}")

if __name__ == "__main__":
    main()