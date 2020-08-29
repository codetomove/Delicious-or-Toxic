# Toxic-or-Delicious
An NLP model that determines if a Tweet is "Toxic or Delicious"

Imports:

import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

Language:
Python 3.8

Purpose:

Use the SVM (Support Vector Machines) algorithm to classify Tweet text. Use Stochastic Gradient Descent neural network to optimize text classification.

“Gradient descent is an iterative algorithm, that starts from a random point on a function and travels down its slope in steps until it reaches the lowest point of that function.”
This means that we just use randomness in our gradient descent. This article explains it beautifully:
https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31

Dataset:

Data is from: https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets?select=processedNegative.csv
I added in some of my own flavor of course.
