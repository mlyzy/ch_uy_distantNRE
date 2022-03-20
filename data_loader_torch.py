from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import copy

from extract_chinese_and_punct import ChineseAndPunctuationExtractor
from torch.utils.data import DataLoader
from data_manager import *
class Duie_loader(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset[0].shape[0]

    def __getitem__(self, index):

        input_ids = torch.from_numpy(self.dataset[0][index])
        inputid_ch = torch.from_numpy(self.dataset[1][index])
        inputid_uy = torch.from_numpy(self.dataset[2][index])
        seq_len = torch.tensor(self.dataset[3][index])
        entity_start_index_ch = torch.from_numpy(self.dataset[4][index])
        entity_end_index_ch = torch.from_numpy(self.dataset[5][index])
        entity_start_index_uy = torch.from_numpy(self.dataset[6][index])
        entity_end_index_uy = torch.from_numpy(self.dataset[7][index])
        entities_ch = torch.from_numpy(self.dataset[8][index])
        entities_uy = torch.from_numpy(self.dataset[9][index])
        label = torch.from_numpy(self.dataset[10][index]).float()


        return (input_ids, inputid_ch ,inputid_uy ,seq_len, entity_start_index_ch,entity_end_index_ch, entity_start_index_uy,entity_end_index_uy, entities_ch,entities_uy,
        label)


