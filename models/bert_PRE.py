from __future__ import absolute_import

import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from models.tcn import TemporalConvNet, TemporalConvNet_BN, TemporalConvNet_nochomp, TemporalConvNet_nochomp_BN
from models.module import *
import numpy as np
import copy
from collections import defaultdict
import models.PRE as PRE
import models.coattention as coattention
import models.gateunit as gate
__all__ = ['Transfromer_Baseline', 'Transfromer_Baseline_2class']

pretrain_model_dict = {
    'bert': BertModel,
    'roberta': BertModel,
}

pretrain_token_dict = {
    'bert': BertTokenizer,
    'roberta': BertTokenizer,
}

heads_no_need_permute = ['gru', 'bilstm']

heads_dict = {
    'res_block': ResBlock_Basic,
    'res_bottleneck_block': ResBlock_Bottleneck,
    'inception_block': Inception_block,
    'res_inception_block': Res_Inception_block,
    'tcn_nobn_chomp': TemporalConvNet,
    'tcn_bn_chomp': TemporalConvNet_BN,
    'tcn_nobn_nochomp': TemporalConvNet_nochomp,
    'tcn_bn_nochomp': TemporalConvNet_nochomp_BN,
    'gru': GRUBlock,
    'bilstm': BiLstmBlock,
}

entity_feat_module_dict = {
    'entity_avg_feats': entity_avg_feats,
    'entity_max_feats': entity_max_feats,
    'entity_avg_max_feats': entity_avg_max_feats,
    'entity_start_end_avg_max_feats': entity_start_end_avg_max_feats,
    'entity_avg_max_fc_bn_relu_feats': entity_avg_max_fc_bn_relu_feats,
    'entity_avg_max_fc_bn_feats': entity_avg_max_fc_bn_feats,
    'entity_avg_max_fc_bn_relu_feats_v2': entity_avg_max_fc_bn_relu_feats_v2,
    'entity_avg_max_fc_bn_feats_v2': entity_avg_max_fc_bn_feats_v2,
    'entity_avg_max_cls_feats': entity_avg_max_cls_feats,
    'entity_avg_max_globalmax_feats': entity_avg_max_globalmax_feats,
    'entity_avg_max_globalavg_feats': entity_avg_max_globalavg_feats,
    'entity_avg_max_globalavgmax_feats': entity_avg_max_globalavgmax_feats,
    'entity_avg_max_product_feats': entity_avg_max_product_feats,
}

feat_process_module_dict = {
    'fc_bn_relu': fc_bn_relu,

}


class Transfromer_Baseline(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transfromer_Baseline, self).__init__()
        self.args = args
        self.pretrain_model_name = args.base_model.lower()
        self.tokenizer = pretrain_token_dict[self.pretrain_model_name].from_pretrained(
            args.model_path)
        base_model = pretrain_model_dict[self.pretrain_model_name].from_pretrained(
            args.model_path)
        self.base_model_embeddings = base_model.embeddings
        self.base_model_encoder = base_model.encoder
        # self.base_model_pooler = base_model.pooler
        # BertTokenizer.from_pretrained('bert-base-uncased')
        # BertModel.from_pretrained('bert-base-uncased')
        self.configuration = base_model.config

        del base_model

        if not args.head == '':
            self.heads = heads_dict[args.head](self.configuration.to_dict()[
                                               'hidden_size'], args.head_out_channels)
            self.classifier = nn.Linear(
                self.heads.out_channels, args.num_classes)
        else:
            self.classifier = nn.Linear(self.configuration.to_dict()[
                                        'hidden_size'], args.num_classes)

        self.entity_feat_module_name = args.entity_feat_module_name
        self.entity_feat_module = entity_feat_module_dict[self.entity_feat_module_name](
            self.configuration.to_dict()['hidden_size'])
        self.predicate_classifier = nn.Linear(self.configuration.to_dict(
        )['hidden_size'] * self.entity_feat_module.feats_num_coef, 2)

        self.sigmoid = nn.Sigmoid()
        self.relattention = PRE.MultiHeadAttentionLayer(
            self.configuration.to_dict()['hidden_size'])
        self.coattention = coattention.ParallelCoAttentionNetwork(self.configuration.to_dict()['hidden_size'],self.configuration.to_dict()['hidden_size'])
        self.gate = gate.Gateunit(self.configuration.to_dict()['hidden_size'])
        # if args.bool_crf:
        #     self.crf = CRF(args.num_classes * args.max_seq_length, batch_first=True)

    def forward(self, x):
        seq_lens = x['seq_lens']
        gt_predicate_labels, gt_predicate_num = x['predicate_labels'], x['predicate_num']
        chtokenid=x['inputid_ch']
        uytokenid=x['inputid_uy']
        x_embed = self.base_model_embeddings(x['input_ids'])
        x_encoder = self.base_model_encoder(x_embed)
        ch_encoder = self.base_model_encoder(chtokenid)
        uy_encoder = self.base_model_encoder(uytokenid)
        # x_pool = self.base_model_pooler(x_encoder[0])
        # x_encoder = self.base_model(**x)

        # print(x_encoder.last_hidden_state.size())([20,128,768])
        pre_ch = self.relattention(
            ch_encoder.last_hidden_state, ch_encoder.last_hidden_state, ch_encoder.last_hidden_state,)
        pre_uy = self.relattention(
            uy_encoder.last_hidden_state, uy_encoder.last_hidden_state, uy_encoder.last_hidden_state,)
        chentity_id_h=pre_ch[x['ch_start'][0]:x['ch_end'][0]]
        chentity_id_t=pre_ch[x['ch_start'][1]:x['ch_end'][1]]
        uyentity_id_h=pre_uy[x['uy_start'][0]:x['uy_end'][0]]
        uyentity_id_t=pre_uy[x['uy_start'][1]:x['uy_end'][1]]
        chentity_id = torch.cat((chentity_id_h,chentity_id_t),0)
        uyentity_id = torch.cat((uyentity_id_h,uyentity_id_t),0)
        chentity_id,uyentity_id = self.coattention(chentity_id,uyentity_id)
        chentity_id = self.gate(chentity_id)
        uyentity_id = self.gate(uyentity_id)
        token = torch.cat((chentity_id,uyentity_id,x_embed[0]))
        x = self.classifier()

        with torch.no_grad():
            prob_x = self.sigmoid(x).cpu()
            prob_x_0_1 = torch.where(
                prob_x > 0.5, torch.tensor(1), torch.tensor(0)).numpy()

        return x, prob_x_0_1


