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
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter


def word_prob(word): return dictionary[word] / total


def words(text): return re.findall('[a-z]+', text.lower())


dictionary = Counter(words(open(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\dataset\wordlists\merged.txt").read()))
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))


def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]


def fix_hashtag(text):
    text = text.group().split(":")[0]
    text = text[1:]  # remove '#'
    try:
        test = int(text[0])
        text = text[1:]
    except:
        pass
    output = ' '.join(viterbi_segment(text)[0])
    return output


def prep(tweet):
    tweet = str(tweet).lower()
    tweet = re.sub("(#[A-Za-z0-9]+)", fix_hashtag, tweet)
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    tweet = re.sub('\d+', '', str(tweet))

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    ps = PorterStemmer()
    words = tweet.split()
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    tweet = " ".join(lemma_words)

    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = tweet.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    tweet = " ".join(clean_words)

    tweet = tweet.strip()
    return tweet


def vectorise_label(label):
    if label == "empty":
        return 0
    elif label == "sadness":
        return 2
    elif label == "enthusiasm":
        return 1
    elif label == "neutral":
        return 0
    elif label == "worry":
        return 2
    elif label == "surprise":
        return 1
    elif label == "love":
        return 3
    elif label == "fun":
        return 1
    elif label == "hate":
        return 4
    elif label == "happiness":
        return 1
    elif label == "boredom":
        return 0
    elif label == "relief":
        return 1
    elif label == "anger":
        return 4
# Read in data

    data1 = pd.read_csv(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\crawled_csv\sad_processes.csv", sep=',', encoding='utf-8')

    # Write preprocessed data to a new CSV file
    dataWriter = csv.writer(open(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\crawled_csv\prep\sad_prep.csv", 'w'), delimiter=',', lineterminator="\n")
    total = data1.shape[0]  # set total to the number of rows
    for i in range(total):
        tweet = prep(data1.iloc[:, 0][i])
        dataWriter.writerow([tweet, 2])

    # Count the number of rows in the CSV file
    count = 0
    with open(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\crawled_csv\prep\sad_prep.csv", encoding = "utf8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            count+=1

    # Read in additional data
    data2 = pd.read_csv(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\dataset\data\text_emotion.csv", sep=',', encoding='utf-8')
    count = 0
    with open(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\cleaned_data\data_prep.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            count+=1


    # Read in training data and split into X and y
data_train = pd.read_csv(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\cleaned_data\emotion_data_prep.csv", sep=',', encoding='utf-8')
count= data_train.iloc[:,1].value_counts()
X_train = data_train.iloc[:,0][:49611]
y_train = data_train.iloc[:,-1][:49611]
X_val = data_train.iloc[:,0][49612:]
y_val = data_train.iloc[:,-1][49612:]

    # Create TF-IDF vectors from X_train and X_val
tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train.astype('U'))
X_val_tfidf = tfidf.fit_transform(X_val.astype('U'))

    # Create bag of words vectors from data_train
bow = tfidf.fit_transform(data_train.iloc[:,0].astype('U'))
word_freq = dict(zip(tfidf.get_feature_names_out(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(30), columns = ['word', 'freq'])

    # Create count vectors from X_train and X_val
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data_train.iloc[:,0].astype('U'))
X_train_count =  count_vect.transform(X_train.astype('U'))
X_val_count =  count_vect.transform(X_val.astype('U'))

    # Create bag of words vectors from data_train
bow = count_vect.fit_transform(data_train.iloc[:,0].astype('U'))
word_freq = dict(zip(count_vect.get_feature_names_out(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(30), columns = ['word', 'freq'])

    # Train a logistic regression model on X_train and y_train
logreg1 = LogisticRegression(C=1, max_iter=500)
logreg1.fit(X_train_count, y_train)

def predict_emotions2(user_text_list, count_vect, logreg):
    # join all the text in user_text_list
    all_text = ' '.join(user_text_list)

    # convert the text to vector counts
    tweet_count = count_vect.transform([all_text])

    # predict the emotion of the tweets
    tweet_pred = logreg.predict(tweet_count)

    # replace the numeric label with the corresponding emotion name
    tweet_pred = np.vectorize({0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'}.get)(tweet_pred)

    return list(tweet_pred)


def predict_emotions(csv_path):
    # Read in training data and split into X and y
    data_train = pd.read_csv(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\cleaned_data\emotion_data_prep.csv",
        sep=',', encoding='utf-8')
    count = data_train.iloc[:, 1].value_counts()
    X_train = data_train.iloc[:, 0][:20611]
    y_train = data_train.iloc[:, -1][:20611]
    X_val = data_train.iloc[:, 0][20612:]
    y_val = data_train.iloc[:, -1][20612:]

    # Create count vectors from X_train and X_val
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(data_train.iloc[:, 0].astype('U'))
    X_train_count = count_vect.transform(X_train.astype('U'))

    # Train a logistic regression model on X_train and y_train
    logreg1 = LogisticRegression(C=1, max_iter=500)
    logreg1.fit(X_train_count, y_train)

    # Read in user conversation data and preprocess it
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['DATE'])
    grouped_texts = df.groupby('datetime')['USER_TEXTS'].agg(lambda x: ' '.join(x))
    tweets = pd.DataFrame({'USER_TEXTS': grouped_texts})

    # Convert the user conversation to vector counts
    tweet_count = count_vect.transform(tweets['USER_TEXTS'])

    # Predict the emotion of the user conversation
    tweet_pred = logreg1.predict(tweet_count)

    # Create a new DataFrame to store the final result
    final_result = pd.DataFrame(
        {'datetime': grouped_texts.index, 'USER_TEXTS': grouped_texts, 'predicted_emotion': tweet_pred})
    final_result.to_csv(r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\DATASET\final_result.csv', index=False)

    # Replace the numeric label with the corresponding emotion name
    final_result['predicted_emotion'] = final_result['predicted_emotion'].replace(
        {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'})

    return final_result

csv_path = r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\DATASET\user_conversation.csv'
result = predict_emotions(csv_path)
