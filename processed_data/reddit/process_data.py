import csv
import pandas as pd

def process_pro_qanon():

    conspiracies = pd.read_csv("../../raw_data/reddit/conspiracies_reddit/conspiracies_raw.csv", error_bad_lines=False)
    conspiracies_undone = pd.read_csv("../../raw_data/reddit/conspiracies_reddit/conspiracies_undone_raw.csv", error_bad_lines=False)
    drain_the_swamp = pd.read_csv("../../raw_data/reddit/extreme_right_wingers_reddits/draintheswamp_raw.csv", error_bad_lines=False)

    all_pro_qanon = pd.concat([conspiracies, conspiracies_undone, drain_the_swamp])
    all_pro_qanon.drop_duplicates(subset=['ID'])

    all_pro_qanon["Title"] = all_pro_qanon['Title'].apply(lambda x: str(x).encode('ascii', errors='ignore'))
    all_pro_qanon["Text"] = all_pro_qanon['Text'].apply(lambda x: str(x).encode('ascii', errors='ignore'))

    all_pro_qanon.to_csv("processed_pro_qanon.csv")

def process_anti_qanon():

    debunking_qanon = pd.read_csv("../../raw_data/reddit/debunking_QAnon_reddit/debunking_qanon_raw.csv", error_bad_lines=False)
    enough_trump = pd.read_csv("../../raw_data/reddit/debunking_QAnon_reddit/enough_trump_spam_raw.csv", error_bad_lines=False)
    qanon_recent_mentions = pd.read_csv("../../raw_data/reddit/QAnon_mentions_reddit/qanon_all_extract_data_raw.csv", error_bad_lines=False)

    all_anti_qanon = pd.concat([debunking_qanon, enough_trump, qanon_recent_mentions])
    all_anti_qanon.drop_duplicates(subset=['ID'])

    all_anti_qanon["Title"] = all_anti_qanon['Title'].apply(lambda x: str(x).encode('ascii', errors='ignore'))
    all_anti_qanon["Text"] = all_anti_qanon['Text'].apply(lambda x: str(x).encode('ascii', errors='ignore'))

    all_anti_qanon.to_csv("processed_anti_qanon.csv")

process_pro_qanon()
process_anti_qanon()


