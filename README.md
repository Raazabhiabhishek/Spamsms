
# 📱 Spam SMS Detection

A machine learning project that classifies SMS messages as spam or ham using text preprocessing, TF-IDF vectorization, and Logistic Regression. Built using the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## 📌 Project Overview

- **🎯 Objective:** Detect spam SMS messages with high accuracy and F1-score.
- **📊 Dataset:** Kaggle SMS Spam Collection — 5,572 messages (86.6% ham, 13.4% spam).
- **⚙️ Technologies:** Python, Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn.
- **🧠 Model:** Logistic Regression with TF-IDF (alternatively, Naive Bayes).
- **📈 Performance:**
  - Accuracy: ~98%
  - F1-Score: ~94%
  - Precision: ~95–99%
  - Recall: ~92–95%

---

## ✅ Features

- Preprocesses text: removes punctuation, stopwords, and applies stemming.
- Uses TF-IDF with bigrams for feature extraction.
- Trains a Logistic Regression model with class balancing.
- Evaluates model with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix (saved as `confusion_matrix.png`)
- Predicts new SMS messages using a built-in function.

---

## 🧱 Prerequisites

- **Python 3.8+**

Install dependencies:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

---

## 📦 Dataset

Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
Place it in your project directory (e.g., `D:\spam_sms\`).

---

## ⚙️ Setup Instructions

### Clone or Download the Project

```bash
git clone https://github.com/Raazabhiabhishek/Spamsms.git
cd spam_sms
```

### Install Dependencies

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

### Download NLTK Resources (if not already downloaded)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run the Script

```bash
python spam_sms_detection_improved.py
```

---

## 🧪 Usage & Output

Run the script:

```bash
python spam_sms_detection_improved.py
```

Expected console output:

```
Dataset loaded. Shape: (5572, 2)
Label distribution:
0    4825
1     747
...

Model Evaluation:
Accuracy: 0.9820
Precision: 0.9565
Recall: 0.9262
F1-Score: 0.9412

Confusion matrix saved as 'confusion_matrix.png'
```

Example predictions:

```
Text: Free entry in 2 a weekly comp to win FA Cup final... -> Predicted: Spam  
Text: Hey, are we still meeting for lunch tomorrow?... -> Predicted: Ham
```

---

## 📂 Output Files

- `confusion_matrix.png` – visual representation of model performance.
- `spam_detector_model.pkl` – saved trained model (optional).
- `tfidf_vectorizer.pkl` – saved vectorizer (optional).

---

## 🔍 Predict New SMS

In the script:

```python
sample_texts = ["Your new SMS here"]
```

Or use saved model:

```python
import joblib
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
```

---

## 📊 Results

- **Accuracy:** ~98%
- **F1-score:** ~94%
- **Precision:** ~95–99% (low false positives)
- **Recall:** ~92–95% (low false negatives)

Confusion Matrix Interpretation (see `confusion_matrix.png`):

- **Top-left:** True Negatives (Ham predicted correctly)  
- **Top-right:** False Positives (Ham predicted as Spam)  
- **Bottom-left:** False Negatives (Spam predicted as Ham)  
- **Bottom-right:** True Positives (Spam predicted correctly)

---

## 📁 Project Structure

```
D:\spam_sms\
├── spam_sms_detection_improved.py   # Main script
├── spam.csv                         # Dataset
├── confusion_matrix.png             # Output visualization
└── README.md                        # This file
```

---

## 🚀 Future Improvements

- 🔧 **Hyperparameter Tuning:** Use `GridSearchCV` to optimize model.
- 🧠 **Alternative Models:** Try SVM, Random Forest, etc.
- 🌐 **Deployment:**
  - Create a Flask API:
    ```bash
    pip install flask
    python api.py
    ```
  - Integrate with web/mobile UI
- 📈 **Data Augmentation:**
  - Use SMOTE to balance classes
  - Add more datasets for improved generalization

---

## 🚼 Troubleshooting

- **File Not Found:** Ensure `spam.csv` is in the working directory.
- **NLTK Errors:** Run `nltk.download('all')` to fetch missing resources.
- **Dependency Issues:** Recheck installations with:
  ```bash
  pip show pandas numpy nltk scikit-learn matplotlib seaborn
  ```
- **Low Performance:** Increase `max_features` in TF-IDF or try SVM.

---

## 🤝 Contributing

Contributions are welcome!  
Fork this repo, open issues, or submit PRs to improve the model or docs.

---

## 🪖 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgments

- **Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Libraries:** Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn
