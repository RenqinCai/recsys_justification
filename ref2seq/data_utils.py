import csv
from nltk import word_tokenize
import random
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
#import cPickle as pickle
import string
import json
from collections import defaultdict
import sys
import argparse
from utilities import *
import pandas as pd
from nltk.tokenize import TweetTokenizer
import pickle

def process_data(df, args):

    df = filter_infrequent_user_item(df, args)

    """
    split train, valid, test
    """
    train_df, valid_df, test_df = split_train_valid_test(df)

    """
    tokenize reviews
    """
    train_df, valid_df, test_df = tokenize_reviews(train_df, valid_df, test_df)

    """
    get vocab: w2i, i2w, user2index, item2index
    """
    vocab = get_vocab(train_df, args)

    """
    map tokens, user, item
    """
    train_df, valid_df, test_df = map_id(vocab, train_df, valid_df, test_df, args)

    return vocab, train_df, valid_df, test_df

def filter_infrequent_user_item(df, args):

    min_user_review_num = args.min_user_review_num
    min_item_review_num = args.min_item_review_num

    print("... filter_infrequent_user_item ...")
    for i in range(2):
        df = remove_users(df, min_user_review_num)
        df = remove_items(df, min_item_review_num)

    df = remove_users(df, min_user_review_num)
    print("sample num", len(df))
    print("user num", df.userid.nunique())
    print("item num", df.itemid.nunique())

    return df

def remove_users(df, min_user_review_num):
    user_freq_map = df['userid'].value_counts()
    filter_users = user_freq_map[user_freq_map>=1000].index
    filter_users_num = len(filter_users)

    filter_users_num_total = 0
    print("---"*20)
    print("filter %d users whose actions >= 1000"%(filter_users_num))
    filter_users_num_total += filter_users_num
    df = df[~df.userid.isin(filter_users)]

    filter_users = user_freq_map[user_freq_map<min_user_review_num].index
    filter_users_num = len(filter_users)
    print("filter %d users whose actions <= %d"%(filter_users_num, min_user_review_num))
    filter_users_num_total += filter_users_num
    df = df[~df.userid.isin(filter_users)]
    
    print("filter %d users"%filter_users_num_total)
    print("after filtering, left user num %d"%df.userid.nunique())
    print("---"*20)
    return df

def remove_items(df, min_item_review_num):
    remove_item_list = []
    item_freq_map = df['itemid'].value_counts()
    for item, freq in item_freq_map.items():
        if freq < min_item_review_num:
            remove_item_list.append(item)
    remove_item_num = len(remove_item_list)
    print("remove item num %d, ratio %.4f"%(remove_item_num, remove_item_num*1.0/len(item_freq_map.keys())))
    print("before filtering, item num %d"%df.itemid.nunique())
    df = df[~df.itemid.isin(remove_item_list)]
    print("after filtering, item num %d"%df.itemid.nunique())
    return df

def split_train_valid_test(df):
    print("... split train valid test ...")

    user_group_itemid_list = df.groupby('userid').itemid.apply(list)

    train = []
    valid = []
    test = []

    for user, itemid_list in user_group_itemid_list.items():
        item_num_user = len(itemid_list)
        if item_num_user < 3:
            continue

        random.shuffle(itemid_list)
        sub_train_item_list = itemid_list[:-2]
        sub_valid_item = itemid_list[-2]
        sub_test_item = itemid_list[-1]

        for sub_train_item in sub_train_item_list:
            train.append([user, sub_train_item])
        
        valid.append([user, sub_valid_item])
        test.append([user, sub_test_item])

    print("train:%d"%len(train))
    print("valid:%d"%len(valid))
    print("test:%d"%len(test))
    
    columns = ['userid', 'itemid']
    train_df = pd.DataFrame(train)
    train_df.columns = columns

    train_user_id_list = list(train_df.userid.unique())
    train_item_id_list = list(train_df.itemid.unique())
    print("---"*20)
    print("train user num", len(train_user_id_list))
    print("train item num", len(train_item_id_list))

    valid_df = pd.DataFrame(valid)
    valid_df.columns = columns
    print("---"*20)
    print("before valid num", len(valid_df))
    valid_df = valid_df[valid_df.userid.isin(train_user_id_list)]
    valid_df = valid_df[valid_df.itemid.isin(train_item_id_list)]
    print("after valid num", len(valid_df))

    test_df = pd.DataFrame(test)
    test_df.columns = columns
    print("---"*20)
    print("before test num", len(test_df))
    test_df = test_df[test_df.userid.isin(train_user_id_list)]
    test_df = test_df[test_df.itemid.isin(train_item_id_list)]
    print("after test num", len(test_df))

    train_df = train_df.merge(df)
    valid_df = valid_df.merge(df)
    test_df = test_df.merge(df)
    print("---"*20)
    print(" "*10, "train:%d"%len(train_df))
    print(" "*10, "valid:%d"%len(valid_df))
    print(" "*10, "test:%d"%len(test_df))

    print("---"*20)

    return train_df, valid_df, test_df

def tokenize_reviews(train_df, valid_df, test_df):

    print("... tokenize ...")
    tweet_tokenizer = TweetTokenizer(preserve_case=False)
    
    def my_tokenizer(s):
        return tweet_tokenizer.tokenize(s)

    train_df['tokens'] = train_df.apply(lambda row: my_tokenizer(row['review']), axis=1)
    valid_df['tokens'] = valid_df.apply(lambda row: my_tokenizer(row['review']), axis=1)
    test_df['tokens'] = test_df.apply(lambda row: my_tokenizer(row['review']), axis=1)

    return train_df, valid_df, test_df

def get_vocab(train_df, args):
    print("... get vocab ...")
    words = []

    extra_words=['<pad>','<unk>','<sos>','<eos>']

    vocab = {}
    min_count = args.min_word_num

    """
    get word2id map
    """
    reviews = train_df.tokens.tolist()

    words = [word.lower() for review in reviews for word in review]
    word_counter = Counter(words)

    words = [x[0] for x in word_counter.most_common() if x[1]>min_count]

    word_index = {w:i+len(extra_words) for i, w in enumerate(words)}
    for i, w in enumerate(extra_words):
        word_index[w] = i

    index_word = {index:word for word, index in word_index.items()}

    vocab['w2i'] = word_index
    vocab['i2w'] = index_word

    """
    get usermap, itemmap
    """

    user_ids = train_df.userid.unique()
    item_ids = train_df.itemid.unique()

    user_index = {key:index for index, key in enumerate(user_ids)}
    item_index = {key:index for index, key in enumerate(item_ids)}

    vocab['user_index'] = user_index
    vocab['item_index'] = item_index

    return vocab

def map_id(vocab, train_df, valid_df, test_df, args):
    print("... map id ...")
    word_index = vocab['w2i']

    min_review_len = args.min_review_len
    max_review_len = args.max_review_len

    def map_token2id(review):
        review_word_idxs_list = []
        for token in review:
            if token in word_index:
                review_word_idxs_list.append(word_index[token])
            else:
                review_word_idxs_list.append(word_index['<unk>'])
    
        return review_word_idxs_list

    train_df['token_idxs'] = train_df.apply(lambda row: map_token2id(row['tokens']), axis=1)

    print("---"*30)
    print("  "*5, "before removing short reviews and long reviews", "  "*5)
    print("  "*5, len(train_df), "  "*5)

    train_df = train_df[train_df['token_idxs'].map(len) >= min_review_len]
    print("  "*5, "after removing short reviews", "  "*5)
    print("  "*5, len(train_df), "  "*5)
    train_df = train_df[train_df['token_idxs'].map(len) < max_review_len]
    print("  "*5, "after removing long reviews", "  "*5)
    print("  "*5, len(train_df), "  "*5)
    print("---"*30)

    user_index = vocab['user_index']
    def map_user2index(userid):
        return user_index[userid]

    train_df['userid'] = train_df.apply(lambda row: map_user2index(row['userid']), axis=1)

    item_index = vocab['item_index']
    def map_item2index(itemid):
        return item_index[itemid]

    train_df['itemid'] = train_df.apply(lambda row: map_item2index(row['itemid']), axis=1)

    print("---"*20)
    print("tokenize valid df")
    print("---"*20)
    valid_df['token_idxs'] = valid_df.apply(lambda row: map_token2id(row['tokens']), axis=1)
    valid_df['userid'] = valid_df.apply(lambda row: map_user2index(row['userid']), axis=1)
    valid_df['itemid'] = valid_df.apply(lambda row: map_item2index(row['itemid']), axis=1)

    print("---"*20)
    print("tokenize test df")
    print("---"*20)
    test_df['token_idxs'] = test_df.apply(lambda row: map_token2id(row['tokens']), axis=1)
    test_df['userid'] = test_df.apply(lambda row: map_user2index(row['userid']), axis=1)
    test_df['itemid'] = test_df.apply(lambda row: map_item2index(row['itemid']), axis=1)

    train_pickle = train_df[["userid", "itemid", "review", "token_idxs"]]
    valid_pickle = valid_df[["userid", "itemid", "review", "token_idxs"]]
    test_pickle = test_df[["userid", "itemid", "review", "token_idxs"]]

    return train_pickle, valid_pickle, test_pickle

def save_data2pickle(data, file_name):
    print("save file to pickle ", file_name)
    data.to_pickle(file_name)

def save_data2json(data, file_name):
    print("save file to json ", file_name)
    with open(file_name, 'w') as f:
        f.write(json.dumps(data))