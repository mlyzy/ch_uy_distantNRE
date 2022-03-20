from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import collections
import json
import os.path as osp
from scipy.io import loadmat
import numpy as np
from typing import Optional, List, Union, Dict
from tqdm import tqdm
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from extract_chinese_and_punct import ChineseAndPunctuationExtractor
from transformers import XLMRobertaTokenizer
InputFeature = collections.namedtuple("InputFeature", [
    "input_ids, inputid_ch","inputid_uy", "seq_len", "entity_start_index_ch","entity_end_index_ch", "entity_start_index_uy","entity_end_index_uy", "entities_ch","entities_uy",
    "label"
])

def convert_example_to_feature(example, tokenizer, chineseandpunctuationextractor, max_length, pad_to_max_length):
    ch_sent=example["ch"]
    uy_sent=example["uy"]
    label=example["label"]
    ch_entity=[]
    uy_entity=[]
    for i in range(len(example["entity"])):
        ch_entity.append(example["entity"][i]["chentity"])
        uy_entity.append(example["entity"][i]["uyentity"])
    sub_ch = []
    sub_uy = uy_sent.split(" ")
    sub_ch = ch_sent.split(" ")
    entity_start_index_ch = []
    entity_end_index_ch = []
    entity_start_index_uy = []
    entity_end_index_uy = []
    tokens_ch = []
    tokens_uy = []



    for i in range(len(ch_entity)):
        entity_start_index_ch.append(ch_sent.find(ch_entity[i]))
        entity_end_index_ch.append(ch_sent.find(ch_entity[i])+len(ch_entity[i]))
        entity_start_index_uy.append(uy_sent.find(uy_entity[i]))
        entity_end_index_uy.append(uy_sent.find(uy_entity[i])+len(uy_entity[i]))
    for (i,token_ch) in enumerate(sub_ch):
        sub_tokens_ch = tokenizer._tokenize(token_ch)

        for sub_tokens in sub_tokens_ch:
            tokens_ch.append(sub_tokens)
            if len(tokens_ch) >= (max_length/2) - 2:
                break
        else:
            continue
        break
    for (j,token_uy) in enumerate(sub_uy):
        sub_tokens_uy = tokenizer._tokenize(token_uy)
        for sub_tokens in sub_tokens_uy:
            tokens_uy.append(sub_tokens)
            if len(tokens_uy) >= (max_length/2) - 2:
                break
        else:
            continue
        break
    seq_len_ch = len(tokens_ch)
    seq_len_uy = len(tokens_uy)
    if seq_len_ch > (max_length/2):
        tokens_ch = tokens_ch[0:((max_length/2) - 2)]
    if seq_len_uy > (max_length/2):
        tokens_uy = tokens_uy[0:((max_length/2) - 2)]
    tokens = ["<s>"] + tokens_ch + ["</s>"] + ["</s>"]+ tokens_uy+ ["</s>"]
    print(tokens)
    seq_len = len(tokens)
    if seq_len < max_length:
        tokens = tokens + ["<pad>"] * (max_length - seq_len - 2)
    print(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
    return InputFeature(
        input_ids=np.array(token_ids)
        inputid_ch=np.array(token_ids)[1:seq_len_ch],
        inputid_uy=np.array(token_ids)[seq_len_ch+2:],
        seq_len=np.array(seq_len),
        entity_start_index_ch = np.array(entity_start_index_ch), 
        entity_end_index_ch = np.array(entity_end_index_ch),
        entity_start_index_uy = np.array(entity_start_index_uy), 
        entity_end_index_uy = np.array(entity_end_index_uy),
        entities_ch = np.array(ch_entity),
        entities_uy = np.array(uy_entity),
        label=label)

class Sentence:
    def __init__(self, 
                file_path: Union[str, os.PathLike],
                tokenizer: XLMRobertaTokenizer,
                max_length: Optional[int]=512,
                pad_to_max_length: Optional[bool]=None):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.dataset = self._process_data(file_path, tokenizer, max_length, pad_to_max_length)
        self.length = self.dataset.label.shape[0]
    def __len__(self):
        return self.length
    def _process_data(self, 
                file_path: Union[str, os.PathLike],
                tokenizer: XLMRobertaTokenizer,
                max_length: Optional[int]=512,
                pad_to_max_length: Optional[bool]=None):
        sen_pair_path = os.path.join(
            os.path.dirname(file_path), "sent_label.json")
        assert os.path.exists(sen_pair_path) and os.path.isfile(
            sen_pair_path
        ), f"{sen_pair_path} dose not exists or is not a file."
        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()
        input_ids, inputid_ch,inputid_uy, seq_len , entities_ch,entity_start_index_ch,entity_end_index_ch,entity_start_index_uy,entity_end_index_uy,entities_uy,label = (
            [] for _ in range(11))
        print("Preprocessing data, loaded from %s" % file_path)
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature = convert_example_to_feature(
                    example, tokenizer,chineseandpunctuationextractor
                    , max_length, pad_to_max_length)
                input_ids.append(input_feature.input_ids)   
                inputid_ch.append(input_feature.inputid_ch)
                inputid_uy.append(input_feature.inputid_uy)
                seq_len.append(input_feature.seq_len)
                entity_start_index_ch.append(input_feature.entity_start_index_ch)
                entity_end_index_ch.append(input_feature.entity_end_index_ch)
                entity_start_index_uy.append(input_feature.entity_start_index_uy)
                entity_end_index_uy.append(input_feature.entity_end_index_uy)
                entities_ch.append(input_feature.entities_ch)
                entities_uy.append(input_feature.entities_uy)
                label.append(input_feature.label)


        dataset= InputFeature(
            input_ids=np.array(input_ids),
            inputid_ch=np.array(inputid_ch),
            inputid_uy=np.array(inputid_uy),
            seq_len=np.array(seq_len),
            entity_start_index_ch = np.array(entity_start_index_ch), 
            entity_end_index_ch = np.array(entity_end_index_ch),
            entity_start_index_uy = np.array(entity_start_index_uy), 
            entity_end_index_uy = np.array(entity_end_index_uy),
            entities_ch = np.array(entities_ch),
            entities_uy = np.array(entities_uy),
            label=np.array(label), )
        return dataset


if __name__ == '__main__':
    # nltk.download('punkt')
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        '/home/yangzhenyu/uy_NRE/Roberta_XLM', do_lower_case=False, cache_dir=None, use_fast=False)
    dataset = Sentence('/home/yangzhenyu/uy_NRE/data/sent_label.json', tokenizer, 512, True)
    print(dataset.length)
