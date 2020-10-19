import requests
import csv
import pandas as pd
from bs4 import BeautifulSoup
import re

# Done q = qanon ; restrict_sr = 0
# Done sr = conspiracytheories ; restrict_sr = 0 ; q = qanon, pedophile, traffickers, satanic

after = ""
print_s = 0
start = True
while start or after:
    with open("qanon_all_extract_data_raw.csv", mode="a") as file:
        file_object = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # if start:
        #     file_object.writerow(["ID", "Title", "Text", "Subreddit", "URL", "Upvote Ratio", "Media Only", "Media Type",
        #                           "Media URL", "Media Title", "Permalink", "Subreddit Subscribers", "Report Reasons"
        #                                                                                             "Removal Reasons",
        #                           "Author", "Is Original Content", "Is Crosspostable", "Num Crossposts",
        #                           "Ups", "View Count", "Crosspost Parent List - Subdreddit(s)"])

        start = False

        query = 'https://www.reddit.com/r/conspiracytheories/search.json?q=satanic&sort=new&restrict_sr=1&limit=100' + after

        q_powerpoint = requests.get(
            query,
            headers={'User-agent': 'your bot 0.1'})

        reddit_data = q_powerpoint.json()["data"]

        if reddit_data["after"] != None:
            after = "&after=" + reddit_data["after"]
        else:
            after = None

        for r in reddit_data["children"]:
            data = r["data"]
            row = []
            row.append(data["id"])
            row.append(data["title"].replace("\n", " ").strip())
            row.append(data["selftext"].replace("\n", " ").strip())
            row.append(data["subreddit"])
            row.append(data["url"])
            row.append(data["upvote_ratio"])
            row.append(data["media_only"])
            if data["secure_media"]:
                if "type" in data["secure_media"]: row.append(data["secure_media"]["type"])
                else: row.append(None)
                if "url" in data["secure_media"]["oembed"]: row.append(data["secure_media"]["oembed"]["url"])
                else: row.append(None)
                if "title" in data["secure_media"]["oembed"]: row.append(data["secure_media"]["oembed"]["title"].replace("\n", " ").strip())
                else: row.append(None)
            else:
                row.append(None)
                row.append(None)
                row.append(None)
            row.append(data["permalink"])
            row.append(data["subreddit_subscribers"])
            row.append(data["report_reasons"])
            row.append(data["removal_reason"])
            row.append(data["author_fullname"])
            row.append(data["is_original_content"])
            row.append(data["is_crosspostable"])
            row.append(data["num_crossposts"])
            row.append(data["ups"])
            row.append(data["view_count"])

            crosspost_parent_list_subreddits = ""
            if "crosspost_parent_list" in data:
                for c in data["crosspost_parent_list"]:
                    crosspost_parent_list_subreddits += c["subreddit"] + ","
                row.append(crosspost_parent_list_subreddits)
            else:
                row.append(None)

            file_object.writerow(row)
            print_s += 1

print(print_s)