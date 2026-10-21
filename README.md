Email Spam Detection:

Description:

This project uses Python and machine learning to classify emails as spam or not spam.
and for the dataset it is from kaggle : Spam Mails Dataset

Technologies
Python
Pandas
NLTK
Scikit-learn
CountVectorizer
Random Forest
Joblib
How it works
Load and preprocess the email dataset.
Remove punctuation and English stopwords.
Apply stemming using PorterStemmer.
Convert text into numerical features using CountVectorizer.
Train a Random Forest classifier.
Predict whether a new email is spam or legitimate.

How to run

Install the required libraries:

pip install pandas numpy nltk scikit-learn joblib


Download the NLTK stopwords:

import nltk
nltk.download("stopwords")


Run the training script:

python train.py


Run the prediction script:

python predict.py

Model

The classifier is trained on a labeled email dataset. Its accuracy depends on the training data and the emails being tested.
