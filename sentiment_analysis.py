from transformers import pipeline
import mysql.connector
import pandas as pd
from tqdm import tqdm
import pickle
from config import *
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os.path
import numpy as np
from collections import Counter
import scipy.stats as stats


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


def get_posts_sentiment(type):
    if os.path.exists(f"data/{type}_sentiment.pickle"):
        with open(f"data/{type}_sentiment.pickle", "rb") as f:
            sentiment = pickle.load(f)
        return sentiment
    else:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=None, truncation=True
        )

        sentiment = dict()

        posts = get_posts(type)
        for _, post in tqdm(posts.iterrows(), total=len(posts)):
            sentiment[post.post_id] = sentiment_pipeline(post.post_body)[0]

        with open(f"data/{type}_sentiment.pickle", "wb") as f:
            pickle.dump(sentiment, f)
        return sentiment


def get_posts_emotions(type):

    if os.path.exists(f"data/{type}_emotions.pickle"):
        with open(f"data/{type}_emotions.pickle", "rb") as f:
            sentiment = pickle.load(f)
        return sentiment
    else:
        classifier = pipeline(
            "text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None, truncation=True)

        emotions = dict()

        posts = get_posts(type)
        for _, post in tqdm(posts.iterrows(), total=len(posts)):
            emotions[post.post_id] = classifier(post.post_body)[0]

        with open(f"data/{type}_emotions.pickle", "wb") as f:
            pickle.dump(emotions, f)
        return emotions


def get_sentiment_max(type):
    sentiment = get_posts_sentiment(type)
    sentiment_max = dict({"POSITIVE": 0, "NEGATIVE": 0})

    for eval in list(sentiment.values()):
        sentiment_max[max(eval, key=lambda x: x["score"])["label"]] += 1

    return sentiment_max, len(sentiment)


def get_emotions_avg(type):
    emotions = get_posts_emotions(type)
    emotions_sum = dict({"love": 0, "joy": 0, "anger": 0,
                         "sadness": 0, "surprise": 0, "fear": 0})

    for eval in list(emotions.values()):
        for emotion in eval:
            emotions_sum[emotion["label"]] += emotion["score"]

    emotions_avg = {k: emotions_sum[k]/len(emotions) for k in emotions_sum}
    return emotions_avg


def get_emotions_max(emotions):
    emotions_max = dict({"love": 0, "joy": 0, "anger": 0,
                         "sadness": 0, "surprise": 0, "fear": 0})
    for eval in list(emotions.values()):
        emotions_max[max(eval, key=lambda x: x["score"])["label"]] += 1
    return emotions_max


def get_posts_emotion_scores(personality, emotion):
    posts = get_posts(personality)
    emotions = get_posts_emotions(posts)
    emotion_scores = []
    for post_evaluation in emotions.values():
        for emotion_evaluation in post_evaluation:
            if emotion_evaluation["label"] == emotion:
                emotion_scores.append(emotion_evaluation["score"])
    return emotion_scores


def get_trait_types(trait):
    TYPES = ["infj", "intj", "infp", "intp", "istj", "isfj", "istp", "isfp",
             "enfj", "enfp", "entp", "entj", "estj", "esfj", "estp", "esfp"]
    return [type for type in TYPES if trait in type]


def get_correlation(trait, non_trait, emotion):
    trait_presence = []
    emotion_scores = []
    for type in tqdm(get_trait_types(trait)):
        type_scores = get_posts_emotion_scores(type, emotion)
        emotion_scores.extend(type_scores)
        trait_presence.extend(np.ones(len(type_scores)))
    for type in tqdm(get_trait_types(non_trait)):
        type_scores = get_posts_emotion_scores(type, emotion)
        emotion_scores.extend(type_scores)
        trait_presence.extend(np.zeros(len(type_scores)))

    corr, pvalue = stats.pointbiserialr(trait_presence, emotion_scores)
    return corr, pvalue


def get_wordcloud(posts):
    text = " ".join([word.lower() for word in " ".join(posts["post_body"]).split(
    ) if word.lower() not in stopwords.words("english")])
    wordcloud = WordCloud(width=1600, height=800).generate(text)
    return wordcloud


def get_types_avg_emotion(types):
    types_sum = dict({"love": 0, "joy": 0, "anger": 0,
                     "sadness": 0, "surprise": 0, "fear": 0})

    for personality in types:
        personality_posts = get_posts(personality)
        personality_avg = get_emotions_avg(get_posts_emotions(
            personality_posts), len(personality_posts))
        types_sum = dict(Counter(types_sum) + Counter(personality_avg))

    return {k: types_sum[k]/len(types) for k in types_sum}
