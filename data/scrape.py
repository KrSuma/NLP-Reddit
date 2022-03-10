import pandas as pd
import praw
import datetime as dt
import os

reddit = praw.Reddit(client_id='yYxiZScsfYdn8Q',
                     client_secret='4zGqZWR58WfvNV9e1M3eAGSrkhg',
                     user_agent='mochohihi')

subreddits_to_get = ['offmychest', 'Showerthoughts', 'MGTOW',
                     'traaaaaaannnnnnnnnns', 'genderqueer', 'GenderCritical']
csv_path = 'reddit_posts.csv'

post_count = 500

for subr in subreddits_to_get:
    posts_dict = {
        "body": [],
        "class": [],
        "date": [],
        "score": [],
        "subreddit": [],
        "id": []
    }
    post_index = 0
    comments_index = 0
    for submission in reddit.subreddit(subr).hot(limit=post_count):

        posts_dict["date"].append(submission.created_utc)
        posts_dict["score"].append(submission.score)
        posts_dict["subreddit"].append(subr)

        if submission.selftext == '':
            posts_dict["body"].append(submission.title)
        else:
            posts_dict["body"].append(submission.selftext)

        posts_dict["id"].append(submission.id)
        posts_dict["class"].append(0)

        post_index += 1
        if post_index >= post_count:
            break

    data = pd.DataFrame(posts_dict)

    if os.path.exists(csv_path):
        data.to_csv(csv_path, index=False, mode='a', header=False)
    else:
        data.to_csv(csv_path, index=False)

    data = None
    posts_dict = None

    print(f"===== Completed {subr}, got {post_index}/{post_count} =====")

