import pandas as pd
import numpy as np
import re
from youtubesearchpython import *


data = pd.read_csv(r'C:\Users\Ananya Gupta\Desktop\minor 2\Chatbot-using-Python-master\DATASET\dataset.csv')
data.drop(columns=data.columns[0], axis=1,  inplace=True)
non_ascii = re.compile('[^\x00-\x7F]+')
data = data[~data['track_name'].str.contains(non_ascii, na=False)]


tracks = data.sort_values(by=['popularity'], ascending=False)
tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)


def get_link(song_name, artist):
  search_query=str(song_name+" song by "+artist)
  custom_search=CustomSearch(search_query, VideoSortOrder.relevance, limit = 1)
  for result in custom_search.result()['result']:
    return result['link']

def songs_by_mood(mood):

    genres = {
      'Happy': ['happy', 'pop', 'reggae', 'party'],
      'Sad': ['sad', 'emo', 'blues', 'country'],
      'Neutral': ['ambient', 'classical', 'jazz', 'folk'],
      'Love': ['romance', 'r-n-b', 'indie', 'soul'],
      'Anger': ['heavy-metal', 'industrial', 'punk-rock', 'hip-hop']
    }

    data["ans"] = data["track_genre"].isin(genres[mood])
    df4 = data[data["ans"] == True]
    df4.sort_values(by=['popularity'], ascending=False, inplace=True)
    df4 = df4[:400]
    # print(df4.head())
    music = {}
    for row in df4.sample(n=11).values:
      song = row[3].translate({ord('"'): None})
      song = song.replace("(", "")
      song = song.replace(")", "")
      song_url = get_link(song, row[1])
      music[song] = song_url

    output_msg = "Here are some song recommendations for you: <br><br>"
    a=1
    for i in music:
      output_msg += f"<u><a href='{music[i]}' target='_blank'>{a}. {i}</u></a><br>"
      a+=1
    return output_msg

import matplotlib.pyplot as plt

genre_count = data['track_genre'].value_counts()
plt.bar(genre_count.index, genre_count.values)
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Number of Songs in Each Genre')
plt.show()

genre_count = data['track_genre'].value_counts()

plt.pie(genre_count.values, labels=genre_count.index, autopct='%1.1f%%')
plt.title('Number of Songs in Each Genre')
plt.show()

