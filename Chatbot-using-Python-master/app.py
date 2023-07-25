import nltk
from flask import Flask, render_template

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import datetime
from keras.models import load_model
model = load_model(r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\chatbot_model.h5')
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
from songsuggestion import songs_by_mood
import sys
from TextEmotionAnalysismaster.mlmodels import predict_emotions2
from TextEmotionAnalysismaster.mlmodels import predict_emotions

user_messages = {}
# create a dictionary to store the conversations for each date
csv_path = r'C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\DATASET\user_conversation.csv'
data_train = pd.read_csv(r"C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\TextEmotionAnalysismaster\cleaned_data\emotion_data_prep.csv",
        sep=',', encoding='utf-8')
X_train = data_train.iloc[:,0][:20611]
y_train = data_train.iloc[:,-1][:20611]

# Create count vectors from X_train
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data_train.iloc[:,0].astype('U'))
X_train_count =  count_vect.transform(X_train.astype('U'))

# Train a logistic regression model on X_train and y_train
logreg = LogisticRegression(C=1, max_iter=500)
logreg.fit(X_train_count, y_train)
# create a CSV file to store user's conversation
csv_file = open(r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\DATASET\user_conversation.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template("login.html")

@app.route('/music_', methods=['GET', 'POST'])
def music_():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    global user_messages
    user_input = request.args.get('msg')

    # add user's message to the user_messages dictionary
    date = datetime.date.today().strftime("%Y-%m-%d")
    if date not in user_messages:
        user_messages[date] = [user_input]
    else:
        user_messages[date].append(user_input)

    # check if user_input is 'bye' or the number of messages for the day exceeds 10
    if user_input.lower() == 'bye':
        # extract messages from all days into a single list
        all_messages = []
        for messages in user_messages.values():
            all_messages.extend(messages)
        # write the user messages and sentiment for the day to the CSV file
        csv_writer.writerow([date, ".".join(all_messages)])
        csv_file.flush()
        result = predict_emotions(csv_path)
        # predict sentiment for all messages
        sentimentlabel = predict_emotions2(user_messages[date], count_vect, logreg)
        mood = ' '.join(sentimentlabel)

        del user_messages[date]
        msg = songs_by_mood(mood)

        response = f"{msg}"
        return str(response)
    elif user_input.lower()!='bye':
        # get chatbot response
        response = chatbot_response(user_input)
        return str(response)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):

    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

import re
data = pd.read_csv(r'C:\Users\Ananya Gupta\Desktop\MINOR 2\Chatbot-using-Python-master\DATASET\dataset.csv')
data.drop(columns=data.columns[0], axis=1, inplace=True)
non_ascii = re.compile('[^\x00-\x7F]+')
data = data[~data['track_name'].str.contains(non_ascii, na=False)]

tracks = data.sort_values(by=['popularity'], ascending=False)
tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)

# %%capture
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['track_genre'])


def get_link(song_name, artist):
    search_query = str(song_name + " song by " + artist)
    custom_search = CustomSearch(search_query, VideoSortOrder.relevance, limit=1)
    for result in custom_search.result()['result']:
        return result['link']


def get_similarities(song_name, data):
    text_array1 = song_vectorizer.transform(data[data['track_name'] == song_name]['track_genre']).toarray()
    num_array1 = data[data['track_name'] == song_name].select_dtypes(include=np.number).to_numpy()

    sim = []
    for idx, row in data.iterrows():
        name = row['track_name']

        text_array2 = song_vectorizer.transform(data[data['track_name'] == name]['track_genre']).toarray()
        num_array2 = data[data['track_name'] == name].select_dtypes(include=np.number).to_numpy()

        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim


def recommend_songs(song_name, data=tracks):
    music = {}

    if tracks[tracks['track_name'] == song_name].shape[0] == 0:

        for row in data.sample(n=20).values:
            song = row[3].translate({ord('"'): None})
            song = song.replace("(", "")
            song = song.replace(")", "")
            song_url = get_link(song, row[1])
            music[song] = song_url
        return music

    data['similarity_factor'] = get_similarities(song_name, data)

    data.sort_values(by=['similarity_factor', 'popularity'],
                     ascending=[False, False],
                     inplace=True)
    data = data[:20]

    for index, row in data.iterrows():
        song = row["track_name"].translate({ord('"'): None})
        song = song.replace("(", "")
        song = song.replace(")", "")
        song_url = get_link(song, row["artists"])
        music[song] = song_url

    return music
#pretty woman, kesariya, woh din, ghagra, tum se hi, balam pichkari, shape of you, deva deva, saware, darasal, khairiyat
songs_cache = {'Pretty Woman': [[('The World Was Wide Enough', 'https://www.youtube.com/watch?v=BQ1ZwqaXJaQ'), ('Mi Gente', 'https://www.youtube.com/watch?v=wnJ6LuUFpMo'), ('Anything You Want - Not That', 'https://www.youtube.com/watch?v=mEfuToVUAY8'), ('With A Little Help From My Friends', 'https://www.youtube.com/watch?v=LpwabPQW4p4')], [('Rise Up feat. Vamero', 'https://www.youtube.com/watch?v=J46ohNSbX0g'), ('Sundown', 'https://www.youtube.com/watch?v=TpM-G0kQO_4'), ('Aabaad Barbaad', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Shots & Squats', 'https://www.youtube.com/watch?v=Cr4DDZqGm2k')], [('Hey I Miss You', 'https://www.youtube.com/watch?v=2a80cWBgT4k'), ('All We Got feat. KIDDO', 'https://www.youtube.com/watch?v=lznH7FZBBYs'), ('Counting Stars', 'https://www.youtube.com/watch?v=hT_nvWreIhg'), ('Boundaries', 'https://www.youtube.com/watch?v=2IG10xfd3tA')], [('Softcore', 'https://www.youtube.com/watch?v=ggG9ySCChYw'), ('Believer - Kaskade Remix', 'https://www.youtube.com/watch?v=U5x96y4mApI'), ('Machi Open the Bottle', 'https://www.youtube.com/watch?v=68ixlbMQaY0'), ('To Be Alone With You', 'https://www.youtube.com/watch?v=QocXrEcpWpU')], [('Kaise Mujhe', 'https://www.youtube.com/watch?v=uC1iJcYOyeY'), ('Valentine', 'https://www.youtube.com/watch?v=tyKu0uZS86Q'), ('She Wonders Why', 'https://www.youtube.com/watch?v=k4_lQoXw2mo'), ('No More Drama', 'https://www.youtube.com/watch?v=dFU2X5CFp5I')]], 'Kesariya': [[('Kesariya', 'https://www.youtube.com/watch?v=BddP6PYo2gs'), ('Mat Aazma Re', 'https://www.youtube.com/watch?v=p_dtI2bLWhY'), ('Theethiriyaai From Brahmastra Tamil', 'https://www.youtube.com/watch?v=-QnEDRpTMm0'), ('Shaayraana', 'https://www.youtube.com/watch?v=6HUYAfCB728')], [('Kunkumamaake From Brahmastra Malayalam', 'https://www.youtube.com/watch?v=hlnt3opKi-c'), ('Sochta Houn', 'https://www.youtube.com/watch?v=rtDw-_1Gh0Q'), ('Kesariya Rangu From Brahmastra Kannada', 'https://www.youtube.com/watch?v=WCDXUgvddR4'), ('Woh Din', 'https://www.youtube.com/watch?v=xC1cj9zhh6k')], [('Lehra Do', 'https://www.youtube.com/watch?v=nDsIy6kRhms'), ('I Love You', 'https://www.youtube.com/watch?v=0JLRExeOH-k'), ('Lehra Do From 83', 'https://www.youtube.com/watch?v=nDsIy6kRhms'), ('Kalank Title Track', 'https://www.youtube.com/watch?v=Grr0FlC8SQA')], [('Emptiness', 'https://www.youtube.com/watch?v=73ZWuAcqGKg'), ('Abhi Kuch Dino Se', 'https://www.youtube.com/watch?v=fbdxYoFb64g'), ('Tu Hi Haqeeqat', 'https://www.youtube.com/watch?v=x3bJeJDgRJQ'), ('Hum Nashe Mein Toh Nahin', 'https://www.youtube.com/watch?v=CYmGrNSPCiQ')], [('Channa Mereya', 'https://www.youtube.com/watch?v=284Ov7ysmfA'), ('Bheegi Si Bhaagi Si', 'https://www.youtube.com/watch?v=yHWPO9DDnsk'), ('Tu Hi Mera', 'https://www.youtube.com/watch?v=yBa3FVQKAvY'), ('Tum Mile', 'https://www.youtube.com/watch?v=w6AZ52kaWTw')]], 'Woh Din': [[('Woh Din', 'https://www.youtube.com/watch?v=xC1cj9zhh6k'), ('Theethiriyaai From Brahmastra Tamil', 'https://www.youtube.com/watch?v=-QnEDRpTMm0'), ('Abhi Kuch Dino Se', 'https://www.youtube.com/watch?v=fbdxYoFb64g'), ('Kunkumamaake From Brahmastra Malayalam', 'https://www.youtube.com/watch?v=hlnt3opKi-c')], [('Kesariya Rangu From Brahmastra Kannada', 'https://www.youtube.com/watch?v=WCDXUgvddR4'), ('Tum Jo Aaye', 'https://www.youtube.com/watch?v=kTXilT_KbUM'), ('Bheegi Si Bhaagi Si', 'https://www.youtube.com/watch?v=yHWPO9DDnsk'), ('Emptiness', 'https://www.youtube.com/watch?v=73ZWuAcqGKg')], [('Channa Mereya', 'https://www.youtube.com/watch?v=284Ov7ysmfA'), ('Ye Tune Kya Kiya', 'https://www.youtube.com/watch?v=w9Qo6p4XsXE'), ('Kesariya', 'https://www.youtube.com/watch?v=BddP6PYo2gs'), ('Pee Loon', 'https://www.youtube.com/watch?v=D8XFTglfSMg')], [('Channa Mereya From Ae Dil Hai Mushkil', 'https://www.youtube.com/watch?v=284Ov7ysmfA'), ('Mere Bina', 'https://www.youtube.com/watch?v=EPHdh1KIdVA'), ('Mat Aazma Re', 'https://www.youtube.com/watch?v=p_dtI2bLWhY'), ('Tu Hi Mera', 'https://www.youtube.com/watch?v=yBa3FVQKAvY')], [('Tere Hawaale From Laal Singh Chaddha', 'https://www.youtube.com/watch?v=KUpwupYj_tY'), ('Shaayraana', 'https://www.youtube.com/watch?v=6HUYAfCB728'), ('Jiyein Kyun', 'https://www.youtube.com/watch?v=6MKL6KZUSJc'), ('Ghagra', 'https://www.youtube.com/watch?v=caoGNx1LF2Q')]], 'Ghagra': [[('Ghagra', 'https://www.youtube.com/watch?v=caoGNx1LF2Q'), ('Bhool Bhulaiyaa', 'https://www.youtube.com/watch?v=B9_nql5xBFo'), ('Aao Milo Chalo', 'https://www.youtube.com/watch?v=Mo5tQDcs__g'), ('Ghar More Pardesiya', 'https://www.youtube.com/watch?v=ntC3sO-VeJY')], [('Labon Ko', 'https://www.youtube.com/watch?v=-FP2Cmc7zj4'), ('Abhi Kuch Dino Se', 'https://www.youtube.com/watch?v=fbdxYoFb64g'), ('Tere Hawaale From Laal Singh Chaddha', 'https://www.youtube.com/watch?v=KUpwupYj_tY'), ('Bheegi Si Bhaagi Si', 'https://www.youtube.com/watch?v=yHWPO9DDnsk')], [('Tu Hi Mera', 'https://www.youtube.com/watch?v=yBa3FVQKAvY'), ('Channa Mereya From Ae Dil Hai Mushkil', 'https://www.youtube.com/watch?v=284Ov7ysmfA'), ('Dil Pe Zakham Khate Hain', 'https://www.youtube.com/watch?v=BTGTtRtN8Ho'), ('Kesariya Rangu From Brahmastra Kannada', 'https://www.youtube.com/watch?v=WCDXUgvddR4')], [('Kunkumamaake From Brahmastra Malayalam', 'https://www.youtube.com/watch?v=hlnt3opKi-c'), ('Tum Mile', 'https://www.youtube.com/watch?v=w6AZ52kaWTw'), ('Sochta Houn', 'https://www.youtube.com/watch?v=rtDw-_1Gh0Q'), ('Shaayraana', 'https://www.youtube.com/watch?v=6HUYAfCB728')], [('Theethiriyaai From Brahmastra Tamil', 'https://www.youtube.com/watch?v=-QnEDRpTMm0'), ('Sajde', 'https://www.youtube.com/watch?v=zfABYXP_NSA'), ('I Love You', 'https://www.youtube.com/watch?v=0JLRExeOH-k'), ('Main Tera Boyfriend', 'https://www.youtube.com/watch?v=FQS7i2z1CoA')]], 'Tum Se Hi': [[('Tum Se Hi', 'https://www.youtube.com/watch?v=mt9xg0mmt28'), ('Ye Ishq Hai', 'https://www.youtube.com/watch?v=dXpG0kavjUo'), ('Deva Deva', 'https://www.youtube.com/watch?v=WjAPDofGg28'), ('Balam Pichkari', 'https://www.youtube.com/watch?v=0WtRNGubWGA')], [('Tu Hi Haqeeqat', 'https://www.youtube.com/watch?v=x3bJeJDgRJQ'), ('Kalank Title Track', 'https://www.youtube.com/watch?v=Grr0FlC8SQA'), ('Darasal', 'https://www.youtube.com/watch?v=uCMYzolEbO0'), ('Mauja Hi Mauja', 'https://www.youtube.com/watch?v=PaDaoNnOQaM')], [('Hawayein', 'https://www.youtube.com/watch?v=cYOB941gyXI'), ('Hawayein From Jab Harry Met Sejal', 'https://www.youtube.com/watch?v=cs1e0fRyI18'), ('Aabaad Barbaad From Ludo', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Badtameez Dil', 'https://www.youtube.com/watch?v=II2EO3Nw4m0')], [('Jhak Maar Ke', 'https://www.youtube.com/watch?v=R5CxtjmrIE4'), ('Aabaad Barbaad', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Raabta Title Track [From Raabta]', 'https://www.youtube.com/watch?v=zAU_rsoS5ok'), ('Main Tera Boyfriend', 'https://www.youtube.com/watch?v=FQS7i2z1CoA')], [('Lehra Do', 'https://www.youtube.com/watch?v=nDsIy6kRhms'), ('Love Me Thoda Lofi', 'https://www.youtube.com/watch?v=B9baRStFq7Q'), ('Lehra Do From 83', 'https://www.youtube.com/watch?v=nDsIy6kRhms'), ('Saware', 'https://www.youtube.com/watch?v=CsOsmgUmT9U')]], 'Balam Pichkari': [[('Balam Pichkari', 'https://www.youtube.com/watch?v=0WtRNGubWGA'), ('Deva Deva', 'https://www.youtube.com/watch?v=mNuhKUOD_A0'), ('Hawayein From Jab Harry Met Sejal', 'https://www.youtube.com/watch?v=cs1e0fRyI18'), ('Darasal', 'https://www.youtube.com/watch?v=uCMYzolEbO0')], [('Ye Ishq Hai', 'https://www.youtube.com/watch?v=dXpG0kavjUo'), ('Hawayein', 'https://www.youtube.com/watch?v=VKcG63CqB18'), ('Tum Se Hi', 'https://www.youtube.com/watch?v=mt9xg0mmt28'), ('Mauja Hi Mauja', 'https://www.youtube.com/watch?v=PaDaoNnOQaM')], [('Tu Hi Haqeeqat', 'https://www.youtube.com/watch?v=x3bJeJDgRJQ'), ('Badtameez Dil', 'https://www.youtube.com/watch?v=II2EO3Nw4m0'), ('Saware', 'https://www.youtube.com/watch?v=CsOsmgUmT9U'), ('Dil Ibaadat', 'https://www.youtube.com/watch?v=-2kl2re74Dk')], [('Raabta Title Track [From Raabta]', 'https://www.youtube.com/watch?v=zAU_rsoS5ok'), ('Aabaad Barbaad From Ludo', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Kyon', 'https://www.youtube.com/watch?v=Rq7tyOcVgLQ'), ('Phir Na Aisi Raat Aayegi From Laal Singh Chaddha', 'https://www.youtube.com/watch?v=cpH8tCyVGPo')], [('Aabaad Barbaad', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Jhak Maar Ke', 'https://www.youtube.com/watch?v=R5CxtjmrIE4'), ('Ae Dil Hai Mushkil Title Track From Ae Dil Hai Mushkil', 'https://www.youtube.com/watch?v=6FURuLYrR_Q'), ('Teri Jhuki Nazar', 'https://www.youtube.com/watch?v=ZgIzch1Pqj4')]], 'Shape Of You': [[('I Hate Everything feat. Action Bronson', 'https://www.youtube.com/watch?v=r-x8S-QTVIw'), ('haunt me x 3', 'https://www.youtube.com/watch?v=4yHuDKlcn-k'), ('Those Eyes', 'https://www.youtube.com/watch?v=_-YjO6KfEMk'), ('Calling All the Monsters', 'https://www.youtube.com/watch?v=bmSpn3EnsE0')], [("She's My Collar feat. Kali Uchis", 'https://www.youtube.com/watch?v=J5G0GC3s3cM'), ('Leave Before You Love Me with Jonas Brothers', 'https://www.youtube.com/watch?v=hmUyEDG7Jy0'), ('Get Money', 'https://www.youtube.com/watch?v=e4oB6wYMcrI'), ('Ik Junoon Paint It Red', 'https://www.youtube.com/watch?v=5PbWtDGOL8A')], [('skeletons', 'https://www.youtube.com/watch?v=w_6fWYY6pRw'), ('Puppy Dog Pals Main Title Theme', 'https://www.youtube.com/watch?v=aAsVwDv6OBs'), ('Tennessee Orange', 'https://www.youtube.com/watch?v=MPacja3hGdA'), ('Mitra Re - From Runway 34', 'https://www.youtube.com/watch?v=qpoVa9B7tNc')], [('Betrayed', 'https://www.youtube.com/watch?v=_dL3AygsCMc'), ('Bad Reputation', 'https://www.youtube.com/watch?v=nO6YL09T8Fw'), ('Still Waiting', 'https://www.youtube.com/watch?v=qO-mSLxih-c'), ('The Setup', 'https://www.youtube.com/watch?v=kGSf-Ngzz_A')], [('Grown Man Sport', 'https://www.youtube.com/watch?v=d9LhFN3H20M'), ("Jude's Song", 'https://www.youtube.com/watch?v=3bn1Aa1UttI'), ('Smack That', 'https://www.youtube.com/watch?v=bKDdT_nyP54'), ('UN DIA ONE DAY', 'https://www.youtube.com/watch?v=BjhW3vBA1QU')]], 'Deva Deva': [[('Deva Deva', 'https://www.youtube.com/watch?v=WjAPDofGg28'), ('Balam Pichkari', 'https://www.youtube.com/watch?v=0WtRNGubWGA'), ('Tum Se Hi', 'https://www.youtube.com/watch?v=mt9xg0mmt28'), ('Tu Hi Haqeeqat', 'https://www.youtube.com/watch?v=x3bJeJDgRJQ')], [('Ye Ishq Hai', 'https://www.youtube.com/watch?v=dXpG0kavjUo'), ('Darasal', 'https://www.youtube.com/watch?v=uCMYzolEbO0'), ('Hawayein From Jab Harry Met Sejal', 'https://www.youtube.com/watch?v=cs1e0fRyI18'), ('Raabta Title Track [From Raabta]', 'https://www.youtube.com/watch?v=zAU_rsoS5ok')], [('Hawayein', 'https://www.youtube.com/watch?v=VKcG63CqB18'), ('Aabaad Barbaad From Ludo', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Aabaad Barbaad', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Mauja Hi Mauja', 'https://www.youtube.com/watch?v=PaDaoNnOQaM')], [('Kalank Title Track', 'https://www.youtube.com/watch?v=Grr0FlC8SQA'), ('Sochta Houn', 'https://www.youtube.com/watch?v=rtDw-_1Gh0Q'), ('Shaayraana', 'https://www.youtube.com/watch?v=6HUYAfCB728'), ('Badtameez Dil', 'https://www.youtube.com/watch?v=II2EO3Nw4m0')], [('Saware', 'https://www.youtube.com/watch?v=CsOsmgUmT9U'), ('Jhak Maar Ke', 'https://www.youtube.com/watch?v=R5CxtjmrIE4'), ('Lehra Do', 'https://www.youtube.com/watch?v=nDsIy6kRhms'), ('Mat Aazma Re', 'https://www.youtube.com/watch?v=p_dtI2bLWhY')]], 'Saware': [[('Saware', 'https://www.youtube.com/watch?v=CsOsmgUmT9U'), ('Raabta Title Track [From Raabta]', 'https://www.youtube.com/watch?v=zAU_rsoS5ok'), ('Hawayein From Jab Harry Met Sejal', 'https://www.youtube.com/watch?v=cs1e0fRyI18'), ('Balam Pichkari', 'https://www.youtube.com/watch?v=0WtRNGubWGA')], [('Aabaad Barbaad', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Ye Ishq Hai', 'https://www.youtube.com/watch?v=dXpG0kavjUo'), ('Aabaad Barbaad From Ludo', 'https://www.youtube.com/watch?v=jh66Pjtqr4k'), ('Te Amo Duet', 'https://www.youtube.com/watch?v=9Ugw2b7HvVI')], [('Dil Ibaadat', 'https://www.youtube.com/watch?v=-2kl2re74Dk'), ('Deva Deva', 'https://www.youtube.com/watch?v=WjAPDofGg28'), ('Tu Hi Haqeeqat', 'https://www.youtube.com/watch?v=x3bJeJDgRJQ'), ('Tum Se Hi', 'https://www.youtube.com/watch?v=mt9xg0mmt28')], [('Hawayein', 'https://www.youtube.com/watch?v=VKcG63CqB18'), ('Darasal', 'https://www.youtube.com/watch?v=uCMYzolEbO0'), ('Kyon', 'https://www.youtube.com/watch?v=Rq7tyOcVgLQ'), ('Teri Jhuki Nazar', 'https://www.youtube.com/watch?v=ZgIzch1Pqj4')], [('Main Tera Boyfriend', 'https://www.youtube.com/watch?v=FQS7i2z1CoA'), ('Subha Hone Na De', 'https://www.youtube.com/watch?v=LHBaiphe2bM'), ('Ik Vaari Aa From Raabta', 'https://www.youtube.com/watch?v=zXLgYBSdv74'), ('Hawayein - Lofi Flip', 'https://www.youtube.com/watch?v=yv1EhF9b1Jg')]], 'Darasal': [[('Darasal', 'https://www.youtube.com/watch?v=uCMYzolEbO0'), ('Hawayein', 'https://www.youtube.com/watch?v=cYOB941gyXI'), ('Hawayein From Jab Harry Met Sejal', 'https://www.youtube.com/watch?v=cs1e0fRyI18'), ('Ye Ishq Hai', 'https://www.youtube.com/watch?v=dXpG0kavjUo')], [('Balam Pichkari', 'https://www.youtube.com/watch?v=0WtRNGubWGA'), ('Mauja Hi Mauja', 'https://www.youtube.com/watch?v=PaDaoNnOQaM'), ('Phir Na Aisi Raat Aayegi From Laal Singh Chaddha', 'https://www.youtube.com/watch?v=cpH8tCyVGPo'), ('Deva Deva', 'https://www.youtube.com/watch?v=WjAPDofGg28')], [('Kyon', 'https://www.youtube.com/watch?v=Rq7tyOcVgLQ'), ('Tum Se Hi', 'https://www.youtube.com/watch?v=mt9xg0mmt28'), ('Ae Dil Hai Mushkil Title Track From Ae Dil Hai Mushkil', 'https://www.youtube.com/watch?v=6FURuLYrR_Q'), ('Dil Ibaadat', 'https://www.youtube.com/watch?v=-2kl2re74Dk')], [('Badtameez Dil', 'https://www.youtube.com/watch?v=II2EO3Nw4m0'), ('Ae Dil Hai Mushkil Title Track', 'https://www.youtube.com/watch?v=6FURuLYrR_Q'), ('Ik Vaari Aa', 'https://www.youtube.com/watch?v=zXLgYBSdv74'), ('Ik Vaari Aa From Raabta', 'https://www.youtube.com/watch?v=zXLgYBSdv74')], [('Subha Hone Na De', 'https://www.youtube.com/watch?v=LHBaiphe2bM'), ('Khairiyat', 'https://www.youtube.com/watch?v=hoNb6HuNmU0'), ('Teri Jhuki Nazar', 'https://www.youtube.com/watch?v=ZgIzch1Pqj4'), ('Jhak Maar Ke', 'https://www.youtube.com/watch?v=R5CxtjmrIE4')]], 'Khairiyat': [[('Khairiyat', 'https://www.youtube.com/watch?v=hoNb6HuNmU0'), ('Ae Dil Hai Mushkil Title Track', 'https://www.youtube.com/watch?v=6FURuLYrR_Q'), ('Phir Na Aisi Raat Aayegi From Laal Singh Chaddha', 'https://www.youtube.com/watch?v=cpH8tCyVGPo'), ('Mauja Hi Mauja', 'https://www.youtube.com/watch?v=PaDaoNnOQaM')], [('Tum Mile - Lofi Flip', 'https://www.youtube.com/watch?v=HhkyEKko868'), ('Mehrama', 'https://www.youtube.com/watch?v=HYUpNJJELeE'), ('Aaftaab', 'https://www.youtube.com/watch?v=U77d9912lrw'), ('Samjhawan - Lofi Flip', 'https://www.youtube.com/watch?v=Ap-TS_8MaFQ')], [('Badtameez Dil', 'https://www.youtube.com/watch?v=II2EO3Nw4m0'), ('Ae Dil Hai Mushkil Title Track From Ae Dil Hai Mushkil', 'https://www.youtube.com/watch?v=6FURuLYrR_Q'), ('Hawayein', 'https://www.youtube.com/watch?v=VKcG63CqB18'), ('Darasal', 'https://www.youtube.com/watch?v=uCMYzolEbO0')], [('Jhak Maar Ke', 'https://www.youtube.com/watch?v=R5CxtjmrIE4'), ('Ik Vaari Aa', 'https://www.youtube.com/watch?v=zXLgYBSdv74'), ('Mere Rashke Qamar', 'https://www.youtube.com/watch?v=gY01irEl8Eo'), ('Deva Deva From Brahmastra', 'https://www.youtube.com/watch?v=mNuhKUOD_A0')], [('Kyon', 'https://www.youtube.com/watch?v=Rq7tyOcVgLQ'), ('Tu Mileya - LoFi', 'https://www.youtube.com/watch?v=wRV-6bHCm5A'), ('Hawayein From Jab Harry Met Sejal', 'https://www.youtube.com/watch?v=cs1e0fRyI18'), ('Raabta', 'https://www.youtube.com/watch?v=piUHBTXsoiY')]]}

@app.route('/music_lib')
def music_lib():
    return render_template('music_library.html')

@app.route('/chat')
def chat():
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')




@app.route('/process_form', methods=['POST'])
def process_form():
    user_input = request.form['user_input']
    print(songs_cache)
    sn = user_input.title()
    if sn in songs_cache.keys():
        rows = songs_cache[sn]
        return render_template('music_library2.html', rows=rows)

    result = recommend_songs(user_input.title())
    rows = []
    row = []
    counter = 1
    for key, value in result.items():
        row.append((key, value))
        if counter % 4 == 0:
            rows.append(row)
            row = []
        counter += 1

    songs_cache[user_input.title()] = rows
    print(songs_cache)
    return render_template('music_library2.html', rows=rows)


def get_rows(result):
    rows = []
    row = []
    counter = 1
    for key, value in result.items():
        row.append((key, value))
        if counter % 4 == 0:
            rows.append(row)
            row = []
        counter += 1
    return rows


def get_words(sentence):
    words = sentence.split()
    if len(sentence) > 40:
        return ' '.join(words[:4])
    else:
        return sentence


def get_music(data):
    music = {}
    for row in data.sample(n=20).values:
        song = row[3].translate({ord('"'): None})
        song = song.replace("(", "")
        song = song.replace(")", "")
        song_url = get_link(song, row[1])
        song2 = get_words(song)
        music[song2] = song_url

    return (get_rows(music))


@app.route('/myrandom', methods=['POST'])
def myrandom():
    selected_option = [str(request.form.get('q1')).lower(), str(request.form.get('q2')).lower(),
                       str(request.form.get('q5')).lower(), str(request.form.get('q6')).lower(),
                       str(request.form.get('q7')).lower(), str(request.form.get('q8')).lower(),
                       str(request.form.get('q9')).lower(), str(request.form.get('q10')).lower(),
                       str(request.form.get('q11')).lower()]

    if all(val is None for val in selected_option):
        result = recommend_songs("wow")
        music = get_rows(result)

    else:
        data["ans"] = data["track_genre"].isin(selected_option)
        df3 = data[data["ans"] == True]
        df3.sort_values(by=['popularity'], ascending=False, inplace=True)
        print(df3.head(10))
        music = get_music(df3[:200])

    return render_template('music_library2.html', rows=music)

@app.route("/plot")
def plot():
    emotion_messages = {
        "Neutral": "Hey there! It looks like you've been feeling pretty neutral lately. Remember that it's okay to feel this way, and it's important to take care of yourself regardless of your emotions. Take some time for yourself today and do something that makes you happy. Even small things like going for a walk or calling a friend can make a big difference. Keep up the good work!.",
        "Happy": "Looks like you've been feeling happy recently! Keep up the good work.",
        "Sad": "I noticed that you've been feeling sad recently. Remember that it's okay to reach out for help if you need it.",
        "Love": "It seems like you're feeling a lot of love lately. That's great to hear!",
        "Anger":"I'm sorry to hear that you're feeling angry. Remember that anger is a natural emotion, but it's important to find healthy ways to express and manage it. Consider taking a break, practicing deep breathing exercises, or talking to someone you trust about your feelings. You can also try channeling your anger into a creative outlet like art or music. Hang in there, and remember that things will get better."
    }
    # Read in the final_result CSV file
    result = pd.read_csv(r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\DATASET\final_result.csv')
    # Reset the index to avoid ambiguity when grouping by datetime and predicted_emotion
    result = result.reset_index()
    # Group the data by datetime and predicted_emotion and count the occurrences of each emotion
    grouped_data = result.groupby(['datetime', 'predicted_emotion']).size().unstack(fill_value=0)
    label_map = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'}
    # Map the numeric labels to emotion names in the column names of grouped_data
    grouped_data.columns = [label_map[c] for c in grouped_data.columns]
    # Create the pie chart
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), facecolor='none')
    ax1.title.set_color('white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    # Plot the bar graph
    grouped_data.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Emotion')
    ax1.set_title('Emotions over time')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.set_xticklabels(grouped_data.index, rotation=45, ha='right')
    ax1.yaxis.set_ticklabels([])

    # Group the data by predicted_emotion and count the occurrences of each emotion
    grouped_data = result.groupby('predicted_emotion').size()

    # Map the numeric labels to emotion names
    label_map = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'}
    grouped_data.index = [label_map[c] for c in grouped_data.index]

    # Get the emotion that occurs the most
    # Group the data by predicted_emotion and count the occurrences of each emotion
    grouped_data = result.groupby('predicted_emotion').size()

    # Map the numeric labels to emotion names
    label_map = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Love', 4: 'Anger'}
    grouped_data.index = [label_map[c] for c in grouped_data.index]

    # Get the emotion that occurred the most
    most_occurred_emotion = grouped_data.idxmax()
    message=emotion_messages[most_occurred_emotion]
    # Create the pie chart
    ax2.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90)
    ax2.title.set_color('white')
    ax2.axis('equal')
    ax2.set_title('Emotion Distribution')

    # Convert the plot to a base64-encoded image for
    # Convert the plot to a base64-encoded image for display on the web page
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')
    buffer.close()
    return render_template('work.html', image=image_base64, message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1500)
    csv_file.close()
