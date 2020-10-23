"""Based on an in-lab implementation.
with efficiency update.
"""
# Author: Zixiaofan (Brenda) Yang, brenda@cs.columbia.edu.
# Please contact me if you have any questions
# The LIWC dictionary file is only distributed to users who bought the original software, so please don't distribute it outside speech lab, thanks!


from collections import defaultdict
import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm
from typing import List, Text, Optional, Dict


# we separate this out into a independent directory.
LIWC_PATH = './LIWC2015_English.dic'


def get_all_categories(dic_file: Text = LIWC_PATH) -> List[Text]:
    """
    Get all category names in the LIWC dictionray

    Output:
        cat_list: all categories in the dictionary

    Parameters:
        dic_file: a LIWC style dictionary file
            English: ./LIWC2015_English.dic
            Chinese: ./Simplified_Chinese_LIWC2015_Dictionary.dic
    """
    liwc_dic = open(dic_file, 'r', encoding="utf-8")
    phase = 0
    cat_list = []
    for line in liwc_dic:
        line = line.strip()
        if '%' in line:
            phase += 1
            continue
        if phase == 1:  # category phase
            while '\t' in line:
                line = line.replace('\t', ' ')
            while '  ' in line:
                line = line.replace('  ', ' ')
            if line.split(' ')[0].isdigit():
                cat_list.append(line.split(' ')[1])
        if phase == 2:
            break
    return cat_list


def get_words_in_category(category: Text,
                          dic_file: Text = LIWC_PATH):
    """
    Get all words in dictionary for a certain LIWC category

    Output:
        word_list: all words in the category

    Parameters:
        dic_file: a LIWC style dictionary file
            English: ./LIWC2015_English.dic
            Chinese: ./Simplified_Chinese_LIWC2015_Dictionary.dic
        category: a LIWC category name
    """
    liwc_dic = open(dic_file, 'r', encoding="utf-8")
    phase = 0
    word_list = []
    index = None
    for line in liwc_dic:
        line = line.strip()
        if '%' in line:
            phase += 1
            if phase == 2 and index is None:
                print('cannot find this liwc category')
                return []
            continue
        if phase == 1:  # category phase
            while '\t' in line:
                line = line.replace('\t', ' ')
            while '  ' in line:
                line = line.replace('  ', ' ')
            if line.split(' ')[1] == category:
                index = line.split(' ')[0]
        if phase == 2:  # word phase
            while '\t' in line:
                line = line.replace('\t', ' ')
            while '  ' in line:
                line = line.replace('  ', ' ')
            fields = line.split(' ')
            if index in fields:
                word_list.append(fields[0])
    # print('found words: ', word_list)
    return word_list


def liwc_categories_in_data(categories: List[Text], liwc_list: List[Text],
                            word_with_star: Optional[bool] = True,
                            normalize: Optional[bool] = True) -> List[Dict[Text, float]]:
    """A more efficient implementation of the liwc feature extraction
    process, where we can run multiple categories on multiple input,
    only once.

    categories: --- category specification to be calculated.
    liwc_list: --- text list that is to be analysed, for each line of text an individual liwc feature is calculated.
    word_with_star: --- whether taking wild-card matching into account
    normalize: --- whether to normalize the LIWC features to [0, 1]
    """

    cat_word_lists = {}
    for category in categories:
        word_list = get_words_in_category(category)
        #  elif isinstance(category, list):
        #      word_list = category

        affix_word_list = []
        if word_with_star:
            original_word_list = word_list[:]
            word_list = []
            for word in original_word_list:
                if '*' in word:
                    affix_word_list.append(word)
                else:
                    word_list.append(word)

        cat_word_lists[category] = (word_list, affix_word_list)

    #  lines = liwc_file.readlines()[start_row:]
    result = []
    for line in liwc_list:
        word_count = defaultdict(int)
        #  line = line.strip().split(',')[text_column].lower()
        # TODO: adding more regularization if needed
        text = line.lower().split(' ')
        for wt in text:
            word_count[wt] += 1

        sum_len = sum([word_count[wt] for wt in word_count])
        cat_feat = {}

        for category in categories:
            word_list, affix_word_list = cat_word_lists[category]
            c_count = 0
            for w in word_list:
                if w in word_count:
                    c_count += word_count[w]

            for w in affix_word_list:
                for text_w in word_list:
                    if text_w[:len(w)-1] == w[:-1]:
                        c_count += word_count[text_w]

            cat_feat[category] = c_count

        result.append({cat: cat_feat[cat] / (sum_len if normalize else 1) for cat in categories})

    return result


'''
Given a list of words, compute LIWC features for both word-level and category-level, including
 (1) sentence length (#words in the text)
 (2) score for each of the word in the input word list - LIWC software doesn't have these word-level scores
 (3) treat the input word list as a LIWC category, compute a LIWC score for the category

Outputs:
    liwc_feat: LIWC features
    liwc_names: the name of the features ('length'/word/'all_words')
    file_list: the names of the files (rows) being analyzed

Parameters:
    word_list: a list of words (all words are from one single category)
    liwc_file_name:  a csv file with texts in one column
    text_column: the column index (start from 0) containing texts to analyze
    name_column: the column index (start from 0) containing the names of the text (e.g. wav file names)
    start_row: the row index(start from 0) to start analysis, used to skip headers
    word_with_star: whether to consider affix words with star (set True for English, False for Chinese)
'''


def compute_customized_liwc_feature(word_list, liwc_file_name, start_row=2,
                                    word_with_star=True,
                                    text_column=4, name_column=0):

    if len(set(word_list)) != len(word_list):
        print('Found duplications in the provided word list, removing duplications')
        word_list = list(set(word_list))

    affix_word_list = []
    if word_with_star:
        original_word_list = word_list[:]
        word_list = []
        for word in original_word_list:
            if '*' in word:
                affix_word_list.append(word)
            else:
                word_list.append(word)
    print('# words:', len(word_list), '; # affix words: ', len(affix_word_list))

    liwc_names = ['length']+word_list[:]+affix_word_list[:]+['all_words']

    # compute liwc score
    liwc_file = open(liwc_file_name, 'r')
    lines = liwc_file.readlines()
    liwc_feat = []
    file_list = []
    for line in lines[start_row:]:
        line = line.strip().split(',')

        file_list.append(line[name_column])

        text = line[text_column].strip().lower().split(' ')
        length = len(text)
        feat = []
        feat.append(length)
        for w in word_list:
            c = text.count(w)
            feat.append(float(c)/length)

        for w in affix_word_list:
            assert(w[-1] == '*' and '*' not in w[:-1])
            c = 0
            for text_w in text:
                if text_w[:len(w)-1] == w[:-1]:
                    c += 1
            feat.append(float(c)/length)
        feat.append(sum(feat[1:]))
        liwc_feat.append(feat)

    liwc_file.close()

    return liwc_feat, liwc_names, file_list


'''
Significance test using Pearson correlation. Can do either category-level or word-level.

Parameters:
    liwc_feat: LIWC features
    liwc_names: the name of the features ('length'/word/'all_words')
    scores: a list of scores or labels (e.g. charisma ratings)
    only_output_all: If Ture, treat all the words as one category; if False, do significance test on each word
'''


def significance_test(liwc_feat, liwc_names, scores, only_output_all=False):

    if len(scores) != len(liwc_feat) or len(liwc_names) != len(liwc_feat[0]):
        print('input dimension wrong, please check')
        return
    if liwc_names[-1] != 'all_words':
        print('the last liwc feature should be all_words, please check')
        return

    liwc_feat = np.asarray(liwc_feat)
    corr = []
    count = 0
    for i in range(len(liwc_names)):
        feats = liwc_feat[:, i]
        r, p = scipy.stats.pearsonr(scores, feats)
        corr.append((r, p))
        if not np.isnan(r) and abs(p) < 0.05:
            count += 1
    print(count, ' out of ', len(liwc_names), ' are significant on 0.05 level')

    if only_output_all:
        if corr[-1][1] < 0.05:
            print('all words as one category - significant! ', corr[-1])
        else:
            print('all words as one category - not significant')
    else:
        importance_tuple = list(zip(liwc_names, corr))
        importance_tuple = sorted(importance_tuple,
                                  key=lambda i: i[1][1], reverse=False)
        sig = [t for t in importance_tuple if t[1][1] < 0.05]
        print('significant words are: ', sig)
