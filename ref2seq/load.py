import re
import os
import unicodedata
import pickle
import numpy as np
from config import MAX_LENGTH, save_dir
from utilities import *
# Our dataset are three dicts: user_review, business_review, user_business_EDU
# depends on the word_vocab file
PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3
class Voc:
    def __init__(self,env):
        self.index_word = env['index_word']
        self.word_index = env['word_index']
        self.n_words = len(self.word_index)

class Data:
    def __init__(self,env, dmax=10, smax = 20):
        self.dmax = dmax  #max number of edu
        self.smax = smax   #max number of token for a edu
        self.voc = Voc(env)
        self.train = env['train']
        self.train_mask_idx = env['train_mask_idx']
        self.dev = env['dev'] 
        self.test = env['test']
        self.user_text = env['user_text'] 
        self.item_text = env['item_text'] 
        self.user_index = env['user_index'] 
        self.item_index = env['item_index']
        
    

def pad_to_max(seq, seq_max, pad_token=0):
    while(len(seq)<seq_max): 
        seq.append(pad_token)
    return seq[:seq_max]

def prep_hierarchical_data_list(data, smax, dmax):
   
    all_data = dict()
    all_lengths = dict()
    for key, value in tqdm(data.items(), desc='building H-dict'):
        new_data = []
        data_lengths = []

        loop = range(0, len(value))
        # print()
        for idx in loop:
            data_list = [SOS_token]+value[idx][:smax-2]+[EOS_token]
            sent_lens = len(data_list)
            if sent_lens==0:
                continue
            if sent_lens>smax:
                sent_lens = smax

            _data_list = pad_to_max(data_list, smax)
            new_data.append(_data_list)
            data_lengths.append(sent_lens)
            if len(new_data) >= dmax: # skip if already reach dmax
                break            
        new_data = pad_to_max(new_data, dmax, # dmax - early skip!
                            pad_token=[SOS_token]+[EOS_token]+[0 for _ in range(smax-2)])

        data_lengths = pad_to_max(data_lengths, dmax, pad_token=2)
        
        all_data[key] = new_data
        all_lengths[key] = data_lengths
    return all_data, all_lengths

def get_train_mask(train_df):
        ### load user text
    user_review = train_df.groupby('userid').review.apply(list)

    ### load item text
    item_review = train_df.groupby('itemid').review.apply(list)

    print("Number of Users={}".format(len(user_review)))
    print("Number of Items={}".format(len(item_review)))

    ### obtain train mask
    train_mask_idx = []

    train_userid_list = train_df.userid.tolist()
    train_itemid_list = train_df.itemid.tolist()
    train_review_list = train_df.review.tolist()

    train_review_num = len(train_review_list)

    for i in range(train_review_num):
        train_userid = train_userid_list[i]
        train_itemid = train_itemid_list[i]

        train_review = train_review_list[i]

        for idx, review_i_user in enumerate(user_review[train_userid]):
            if review_i_user == train_review:
                user_mask_idx = idx
                break

        for idx, review_i_item in enumerate(item_review[train_itemid]):
            if review_i_item == train_review:
                item_mask_idx = idx
                break

        train_mask_idx.append((user_mask_idx, item_mask_idx))

    print("train num")
    print(len(train_mask_idx))
    print(train_review_num)

    return train_mask_idx

def wrap_data(data_df):
    data = []

    userid_list = data_df.userid.tolist()
    itemid_list = data_df.itemid.tolist()
    token_idxs_list = data_df.token_idxs.tolist()

    sample_num = len(userid_list)
    print('sample num', sample_num)
    
    for i in range(sample_num):
        userid = userid_list[i]
        itemid = itemid_list[i]
        token_idxs = token_idxs_list[i]

        sub_data = []
        sub_data.append(userid)
        sub_data.append(itemid)
        sub_data.append(token_idxs)

        data.append(sub_data)

    return data

def loadPrepareData(args):

    print("Start loading...")
    # path = '{}/{}/env.json'.format(save_dir, args.corpus)
    # print(path)
    # env = dictFromFileUnicode(path)
    data_dir = args.data_dir

    train_data_file = data_dir+"train.pickle"
    valid_data_file = data_dir+"valid.pickle"
    test_data_file = data_dir+"test.pickle"

    print("---"*30)
    print("train: ", train_data_file)
    print("valid: ", valid_data_file)
    print("test: ", test_data_file)
    print("---"*30)
    ### load train, valid, test
    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)

    train_mask_idx = get_train_mask(train_df)

    print("--"*20, "train", "--"*20)
    train = wrap_data(train_df)

    print("--"*20, "valid", "--"*20)
    valid = wrap_data(valid_df)

    print("--"*20, "test", "--"*20)
    test = wrap_data(test_df)

    ### obtain user tokenized reviews
    user_token_idxs = dict(train_df.groupby('userid').token_idxs.apply(list))

    ### obtain item tokenized reviews
    item_token_idxs = dict(train_df.groupby('itemid').token_idxs.apply(list))

    vocab_file = "{}/{}.json".format(data_dir, 'vocab')
    vocab = dictFromFileUnicode(vocab_file)

    word_index = vocab['w2i']
    index_word = vocab['i2w']
    
    env = {
        'train':train,
        'train_mask_idx':train_mask_idx,
        'dev':dev,
        'test':test,
        'user_text':user_text,
        'item_text':item_text,
        'index_word':index_word,
        'word_index':word_index,
        'user_index':user_index,
        'item_index':item_index
    }
    print("++"*20, "new loading", "++"*20)

    print('loading done...')
    ##prepare review data
    data = Data(env, args.dmax, args.smax)
    data.user_text, user_length = prep_hierarchical_data_list(data.user_text,  data.smax, data.dmax)
    data.item_text, item_length = prep_hierarchical_data_list(data.item_text, data.smax, data.dmax)
    length = [user_length, item_length]#, user_length2, item_length2]
    return data, length   #voc, user_review, business_review, user_business_EDU, train_pairs, valid_pairs, test_pairs

if __name__ == '__main__':
    loadPrepareData()
    
