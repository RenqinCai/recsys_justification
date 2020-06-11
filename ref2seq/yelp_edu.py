"""
1. transform data into dataframe
2. split train, valid, test
3. tokenize reviews, get token mapping
4. get user, item mapping 
5. map user, item, tokens in train, valid, test
"""

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
from data_utils import *

parser = argparse.ArgumentParser(description='Dataset Settings')
parser.add_argument('--dataset', dest='dataset') # category
parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--data_file', type=str, default="./pickle")
parser.add_argument('--max_review_num', type=float, default=1e6)

parser.add_argument('--min_user_review_num', type=int, default=10)
parser.add_argument('--min_item_review_num', type=int, default=10)

parser.add_argument('--max_review_len', type=int, default=100)
parser.add_argument('--min_review_len', type=int, default=5)
# parser.add_argument('--max_word_num', type=int, default=100)
parser.add_argument('--min_word_num', type=int, default=5)

args = parser.parse_args()

dataset = 'data/{}'.format(args.dataset)
print("read data from ", dataset)


"""
read raw data
transform raw data into dataframe
"""

def read_raw_data2df(raw_file):
    interactions = []

    with open(in_fp, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            
            d = json.loads(line)
            user = d['user_id']
            item = d['business_id']
            text = d['edu'].strip(string.punctuation+' '+'\n')   #remove punctuation space \n
            text = text.replace('\n','')

            interactions.append([user, item, text])

    columns = ['userid', 'itemid', 'review']
    df = pd.DataFrame(interactions)
    df.columns = columns

    print("sample num", len(df))

    return df

in_fp = os.path.join(args.data_dir, args.data_file)
# in_fp = '{}/{}_filter_flat_positive.large.json'.format(args.data_dir, args.dataset)
df = read_raw_data2df(in_fp)

vocab, train_df, valid_df, test_df = process_data(df, args)

new_data_dir = args.data_dir

train_file = new_data_dir+"train.pickle"
save_data2pickle(train_df, train_file)

valid_file = new_data_dir+"valid.pickle"
save_data2pickle(valid_df, valid_file)

test_file = new_data_dir+"test.pickle"
save_data2pickle(test_df, test_file)

vocab_file = new_data_dir+"vocab.json"
save_data2json(vocab, vocab_file)


