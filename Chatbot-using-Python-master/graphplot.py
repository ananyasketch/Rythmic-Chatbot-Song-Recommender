import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import datetime
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
intents = json.loads(open(r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\DATASET\intents.json', encoding="utf-8").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

from flask import Flask, render_template, request
from flask import Flask, render_template, url_for
import csv
from textblob import TextBlob
from sklearn.cluster import KMeans
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import random
import pandas as pd
import numpy as np
import re
from youtubesearchpython import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import nltk
import io
from textblob import Word
import re
import sys, os, csv
import string
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import emot
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO
from nltk.corpus import stopwords
app = Flask(__name__, template_folder='templates')
@app.route('/plot')
def plot():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Read in the final_result CSV file
    result = pd.read_csv(r'C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\DATASET\final_result.csv')

    # Reset the index to avoid ambiguity when grouping by datetime and predicted_emotion
    result = result.reset_index()

    # Group the data by datetime and predicted_emotion and count the occurrences of each emotion
    grouped_data = result.groupby(['datetime', 'predicted_emotion']).size().unstack(fill_value=0)
    label_map = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'}

    # Map the numeric labels to emotion names in the column names of grouped_data
    grouped_data.columns = [label_map[c] for c in grouped_data.columns]

    # Create the pie chart
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot the bar graph
    grouped_data.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Emotion')
    ax1.set_title('Emotions over time')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_ticklabels([])

    # Group the data by predicted_emotion and count the occurrences of each emotion
    grouped_data = result.groupby('predicted_emotion').size()

    # Map the numeric labels to emotion names
    label_map = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'}
    grouped_data.index = [label_map[c] for c in grouped_data.index]

    # Create the pie chart
    ax2.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Emotion Distribution')

    # Save the plot to a file
    plt.savefig('static/images/plot.png')

    return render_template('work.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1200)