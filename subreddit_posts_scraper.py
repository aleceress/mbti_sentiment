import praw
from psaw import PushshiftAPI
import mysql.connector
import datetime as dt
from config import *

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

reddit_db = mysql.connector.connect(
    host=HOST, user=USER, password=PASSWORD, database=DATABASE_NAME
)


def add_subreddit_posts(subreddit_name):
    api = PushshiftAPI()

    for post in api.search_submissions(subreddit=subreddit_name):
        sql = "INSERT INTO posts (id, title, body, subreddit, timestamp) VALUES (%s, %s, %s, %s, %s)"
        try:
            val = (
                post.id,
                post.title,
                post.selftext,
                post.subreddit,
                dt.datetime.utcfromtimestamp(post.created),
            )
            cursor.execute(sql, val)
            reddit_db.commit()
        except AttributeError:
            continue


def is_table_present(table_name):
    cursor = reddit_db.cursor()
    sql = f"SHOW tables like '{table_name}'"
    cursor.execute(sql)
    return len(cursor.fetchall()) != 0


cursor = reddit_db.cursor()

# create posts table if  not present
if not is_table_present("posts"):
    cursor.execute(
        "CREATE TABLE posts(id VARCHAR(255) PRIMARY KEY, title TEXT, body TEXT, subreddit VARCHAR(255))"
    )

# adds posts from the specified subreddit
subreddit_name = input("name of the subreddit you want to scrape: ")
print("scraping posts...")
add_subreddit_posts(SUBREDDIT_NAME)

sql = "SELECT id from posts"
cursor.execute(sql)
post_ids = cursor.fetchall()
post_ids = [post_id[0] for post_id in post_ids]
