import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


stemmer = PorterStemmer()
stopwords_set = set(stopwords.words("english"))


vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load('spam_detect.joblib')


email = input('put your email here ==> ')
email = email.replace("\r\n", " ")
email = email.lower()

email = email.translate(str.maketrans(" " , " ", string.punctuation)).split()
email = [stemmer.stem(word) for word in email if word not in stopwords_set]
email = " ".join(email)



x = vectorizer.transform([email]).toarray()


prediction = model.predict(x)

if prediction == 1:
  print("\n spam")
else:
  print("\n not a spam")




print(model.predict_proba(x))