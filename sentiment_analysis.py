from transformers import pipeline
import mysql.connector
import pandas as pd
from tqdm import tqdm
import pickle
from config import *
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os.path


def get_posts(type):
    reddit_db = mysql.connector.connect(
        host=HOST, user=USER, password=PASSWORD, database=DATABASE_NAME
    )
    cursor = reddit_db.cursor()

    cursor.execute(f"select * from posts where subreddit = '{type}'")
    posts = cursor.fetchall()
    posts = pd.DataFrame(
        posts, columns=["post_id", "post_title",
                        "post_body", "subreddit_name", "timestamp"]
    ).drop(columns=["timestamp"])
    return posts


def get_posts_sentiment(posts):
    type = posts[:1].subreddit_name[0]

    if os.path.exists(f"data/{type}_sentiment.pickle"):
        with open(f"data/{type}_sentiment.pickle", "rb") as f:
            sentiment = pickle.load(f)
        return sentiment
    else: 
        classifier = pipeline(
            "text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None, truncation=True)

        sentiment = dict()

        for _, post in tqdm(posts.iterrows(), total=len(posts)):
            sentiment[post.post_id] = classifier(post.post_body)[0]
        
        with open(f"data/{type}_sentiment.pickle", "wb") as f:
            pickle.dump(sentiment, f)
        return sentiment 


def get_emotions_avg(sentiment, cardinality):
    emotions_sum = dict({"love": 0, "joy": 0, "anger": 0,
                         "sadness": 0, "surprise": 0, "fear": 0})

    for eval in tqdm(list(sentiment.values())):
        for emotion in eval:
            emotions_sum[emotion["label"]] += emotion["score"]

    emotions_avg = {k: emotions_sum[k]/cardinality for k in emotions_sum}
    return emotions_avg


def get_emotions_max(sentiment):
    emotions_max = dict({"love": 0, "joy": 0, "anger": 0,
                         "sadness": 0, "surprise": 0, "fear": 0})
    for eval in tqdm(list(sentiment.values())):
        emotions_max[max(eval, key=lambda x: x["score"])["label"]] += 1
    return emotions_max


def get_wordcloud(posts):
    text = " ".join([word.lower() for word in " ".join(posts["post_body"]).split(
    ) if word.lower() not in stopwords.words("english")])
    wordcloud = WordCloud(width=1600, height=800).generate(text)
    return wordcloud
