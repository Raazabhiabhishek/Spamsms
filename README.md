Spam SMS Detection
This project implements a machine learning model to classify SMS messages as spam or ham (non-spam) using the Kaggle SMS Spam Collection Dataset. The model leverages text preprocessing, TF-IDF vectorization, and a Logistic Regression classifier to achieve high accuracy and robust spam detection.
Project Overview

Objective: Build a machine learning model to detect spam SMS messages with high accuracy and F1-score.
Dataset: Kaggle SMS Spam Collection Dataset (spam.csv), containing 5,572 SMS messages labeled as ham (86.6%) or spam (13.4%).
Technologies: Python, Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn.
Model: Logistic Regression with TF-IDF vectorization (alternatively, Naive Bayes).
Performance:
Accuracy: ~98%
F1-Score: ~94%
Precision: ~95–99%
Recall: ~92–95%



Features

Preprocesses text by removing punctuation, stopwords, and applying stemming.
Uses TF-IDF vectorization with bigrams for feature extraction.
Trains a Logistic Regression model with class weighting to handle imbalanced data.
Evaluates performance with accuracy, precision, recall, F1-score, and a confusion matrix.
Saves a confusion matrix visualization as confusion_matrix.png.
Includes a prediction function for new SMS messages.

Prerequisites

Python 3.8 or higher

Required Python libraries:
pip install pandas numpy nltk scikit-learn matplotlib seaborn


Kaggle SMS Spam Collection Dataset (spam.csv)


Setup Instructions

Clone or Download the Project:

Clone this repository or download the project files to your local machine.
Ensure you’re in the project directory (e.g., D:\spam_sms).


Install Dependencies:
pip install pandas numpy nltk scikit-learn matplotlib seaborn


Download NLTK Resources:

The script automatically downloads required NLTK resources (punkt, punkt_tab, stopwords). If issues arise, run manually:
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')




Download the Dataset:

Download spam.csv from Kaggle.
Place spam.csv in the project directory (e.g., D:\spam_sms).


Run the Script:
cd D:\spam_sms
python spam_sms_detection_improved.py



Usage

Running the Model:

Execute the script to load the dataset, train the model, evaluate performance, and generate predictions:
python spam_sms_detection_improved.py


Expected Output:
Dataset loaded. Shape: (5572, 2)
Label distribution:
0    4825
1     747
Name: label, dtype: int64

Model Evaluation (Logistic Regression):
Accuracy: 0.9820
Precision: 0.9565
Recall: 0.9262
F1-Score: 0.9412

Confusion matrix saved as 'confusion_matrix.png'

Example Predictions:
Text: Free entry in 2 a weekly comp to win FA Cup final... -> Predicted: Spam
Text: Hey, are we still meeting for lunch tomorrow?... -> Predicted: Ham




Output Files:

confusion_matrix.png: A visualization of the model’s performance, showing true positives, false positives, true negatives, and false negatives.


Predicting New SMS:

Modify the sample_texts list in the script to test new messages:
sample_texts = ["Your new SMS here"]


Alternatively, save and load the model for predictions:
import joblib
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')





Results

Model Performance:

Achieves ~98% accuracy and ~94% F1-score, competitive with state-of-the-art text classification models.
High precision (~95–99%) ensures few false positives (ham misclassified as spam).
Improved recall (~92–95%) reduces false negatives (missed spam messages).


Confusion Matrix:

Visualize confusion_matrix.png to inspect model performance:
Top-left: True Negatives (Ham correctly predicted)
Top-right: False Positives (Ham predicted as Spam)
Bottom-left: False Negatives (Spam predicted as Ham)
Bottom-right: True Positives (Spam correctly predicted)





Project Structure
D:\spam_sms\
│── spam_sms_detection_improved.py  # Main script
│── spam.csv                       # Dataset
│── confusion_matrix.png           # Output visualization
│── README.md                      # This file

Improvements and Future Work

Hyperparameter Tuning:
Use GridSearchCV to optimize Logistic Regression parameters (e.g., C, solver).


Alternative Models:
Experiment with Support Vector Machines (SVM) or Random Forests for potentially better performance.


Deployment:
Create a Flask API to serve predictions:
pip install flask
python api.py


Integrate with a web or mobile app for real-time spam detection.



Dataset Augmentation:
Use oversampling (e.g., SMOTE) to address class imbalance.
Incorporate additional SMS datasets for robustness.



Troubleshooting

File Not Found:
Ensure spam.csv is in the project directory (D:\spam_sms).
Check the file name and path in load_data(file_path='spam.csv').


NLTK Errors:
If punkt_tab or other resources are missing:
import nltk
nltk.download('all')




Dependency Issues:
Verify all libraries are installed:
pip show pandas numpy nltk scikit-learn matplotlib seaborn




Low Performance:
If accuracy or F1-score is below expectations, try increasing max_features or switching to SVM.



Contributing
Feel free to fork this project, submit issues, or create pull requests to enhance the model, add features, or improve documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: SMS Spam Collection Dataset
Libraries: Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn
Built with guidance from xAI’s Grok 3



