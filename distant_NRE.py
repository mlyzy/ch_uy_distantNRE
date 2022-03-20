import argparse
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys
from glob import glob
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset
import data_manager
from data_loader_torch import Duie_loader
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, AutoTokenizer
from transformers import AdamW, AutoConfig, get_linear_schedule_with_warmup
from adv_train_utils import FGM, PGD

import models
from utils import decoding, decoding_select, find_entity, get_precision_recall_f1, write_prediction_results, Logger, set_random_seed

from loss import BCELossForDuIE, BCELossForDuIE_BI, BCELossForDuIE_BIO, BCELossForDuIE_Smooth, BCELossForDuIE_Smooth_prob

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--eval_only", action='store_true', default=False, help="do predict")
parser.add_argument("--pre_only", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")

parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--trainset", default="duie_spo", type=str, required=False, help="trainset.")
parser.add_argument("--testset", default="duie_spo", type=str, required=False, help="testset.")
parser.add_argument("--trainset_dir", default="./data/train.json", type=str, required=False, help="trainset data path.")
parser.add_argument("--testset_dir", default="./data/dev.json", type=str, required=False, help="testset data path.")
parser.add_argument("--max_seq_length", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )

parser.add_argument("--model", default='transfromer', type=str, help="model.")
parser.add_argument("--base_model", default='bert', type=str, help="Pretrain model.")
parser.add_argument("--model_path", default='./Roberta_XLM', type=str, required=False, help="Path to data.")
parser.add_argument("--do_lower_case", action='store_true', default=False, help="whether lower_case")
parser.add_argument("--head", type=str, default='', help="classification classes")
parser.add_argument("--head_out_channels", type=int, nargs='+', default=1024, help="classification classes")

parser.add_argument('--bool_attention_mask', action='store_true', default=False, help="whether pretrain heads")

parser.add_argument('--loss_func', default='BCELossForDuIE_Smooth_prob', type=str, help="loss function")
parser.add_argument("--weight_coef", default=111.0, type=float, help="weight coef for loss calculate.", )
parser.add_argument("--exponent_coef", default=3, type=int, help="exponent coef for loss calculate.", )
parser.add_argument("--batch_size", default=20, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_epochs", default=1, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
parser.add_argument('--print_steps', type=int, default=100, help="print frequency")
parser.add_argument('--train_continue', action='store_true', default=False, help="whether pretrain heads")
parser.add_argument('--lr_backbone_coef', default=1.0, type=float, help="whether pretrain heads")
parser.add_argument("--entity_feat_module_name",
                    default='entity_avg_max_cls_feats', type=str, help="Pretrain model.")
parser.add_argument('--only_train_heads', action='store_true', default=False, help="whether pretrain heads")
parser.add_argument('--pretrain_heads', action='store_true', default=False, help="whether pretrain heads")
parser.add_argument('--pretrain_heads_epochs', type=int, default=1, help="print frequency")
parser.add_argument("--learning_rate_pretrain", default=2e-4, type=float, help="The initial learning rate for Adam.")

parser.add_argument("--use_cpu", action='store_true', default=False, help="do predict")
parser.add_argument("--n_gpu", default=4, type=int, help="number of gpus to use, 0 for cpu.")
parser.add_argument("--gpu_id", default='0', type=str, help="gpu devices for using.")
parser.add_argument("--num_workers", default=16, type=int, help="number of gpus to use, 0 for cpu.")

parser.add_argument("--load_trained_model", action='store_true', default=False, help="load trained model.")
parser.add_argument("--model_dict_test", default='./log/0421/Transfromer_bert/xlmroberta_batch32/model.pth', type=str, help="save_model.")
parser.add_argument("--output_dir", default="./log/0421", type=str, required=False, help="The output directory.")
parser.add_argument("--tag", default='uydistantNRE', type=str, help="tag for save.")
args = parser.parse_args()


def main():
    set_random_seed(args.seed, args.n_gpu)

    args.output_dir = os.path.join(args.output_dir, '%s_%s'%(args.model, args.base_model), args.tag)
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.eval_only:
        sys.stdout = Logger(os.path.join(args.output_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(os.path.join(args.output_dir, 'log_evaluate.txt'))

    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_id))
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")


    args.model = args.model.lower()
    model = models.init_model(args.model, args).cuda()
    tokenizer = model.tokenizer
    config = model.configuration

    if args.load_trained_model:
        print('======loading trained model======')
        if os.path.exists(args.model_dict_test):
            model_state_dict = torch.load(args.model_dict_test)
            try:
                model.load_state_dict(model_state_dict, strict=True)
            except:
                print('Error occured in state_dict loading!!!')
        else:
            print('No model found in %s!!!'%args.model_dict_test)
            sys.exit(0)


    if args.train_continue:
        print('======loading trained model======')
        if len(glob(os.path.join(args.output_dir, 'model*.pth'))) > 0:
            model_state_dict = torch.load(glob(os.path.join(args.output_dir, 'model*.pth'))[0])
            try:
                model.load_state_dict(model_state_dict, strict=True)
            except:
                print('Error occured in state_dict loading!!!')
        else:
            print('No model found in %s!!!'%args.output_dir)
            sys.exit(0)

    # Loads dataset.
    print('======Dataset loading======')
    pin_memory = True if use_gpu else False
    if (not args.eval_only) and (not args.pre_only):
        train_dataset = data_manager.init_dataset(args.trainset.lower(), args.trainset_dir, tokenizer, args.max_seq_length, pad_to_max_length=True)
        print('======Dataset loading Finihsed!!! trainset:{} '.format(train_dataset.length))
        trainloader = DataLoader(
            Duie_loader(train_dataset.dataset),
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=pin_memory, 
            drop_last=True,
            shuffle=True,
        )

    test_dataset = data_manager.init_dataset(args.testset.lower(), args.testset_dir, tokenizer, args.max_seq_length, pad_to_max_length=True)
    print('======Dataset loading Finihsed!!! testset: {}'.format(test_dataset.length))

    testloader = DataLoader(
        Duie_loader(test_dataset.dataset),
        batch_size=args.test_batch_size, 
        num_workers=args.num_workers,
        pin_memory=pin_memory, 
        drop_last=False,
    )   

    # loss function
    criterion = eval(args.loss_func)(args)


    if args.eval_only:
        # Does predictions.
        print("\n=====start predicting=====")
        precision, recall, f1 = evaluate(model, criterion, testloader, args.testset_dir, args.output_dir, "eval", use_gpu, args)
        print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %(100 * precision, 100 * recall, 100 * f1))
        print("=====predicting complete=====")
        sys.exit(0)

    if args.pre_only:
        # Does predictions.
        print("\n=====start predicting=====")
        evaluate(model, criterion, testloader, args.testset_dir, args.output_dir, "predict", use_gpu, args)
        print("=====predicting complete=====")
        sys.exit(0)

        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and (not any(nd in n for nd in no_decay)) and ('base' in n)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and (not any(nd in n for nd in no_decay)) and ('base' not in n)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and (any(nd in n for nd in no_decay)) and ('base' in n)],
            "weight_decay": 0.0
        }, 
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and (any(nd in n for nd in no_decay)) and ('base' not in n)],
            "weight_decay": 0.0
        }]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    # Defines learning rate strategy.
    steps_by_epoch = int(math.ceil(len(trainloader) // args.gradient_accumulation_steps))
    num_training_steps = steps_by_epoch * args.num_train_epochs
    warmup_steps = args.warmup_ratio * num_training_steps

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    if args.train_continue:
        start_epoch = int(glob(os.path.join(args.output_dir, 'optimizer*.pth'))[0].split('/')[-1].split('_')[2])
        last_epoch = start_epoch * steps_by_epoch
        for step_i in range(last_epoch):
            lr_scheduler.step()
    else:
        start_epoch = 0

    if args.train_continue:
        
        print('======loading optimizer param======')
        if len(glob(os.path.join(args.output_dir, 'optimizer*.pth'))) > 0:
            # scheduler
            optimizer_state_dict = torch.load(glob(os.path.join(args.output_dir, 'optimizer*.pth'))[0])
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except:
                print('Error occured in state_dict loading!!!')
        else:
            print('No model found in %s!!!'%args.model_dict_test)
            sys.exit(0)

    torch.save(args, os.path.join(args.output_dir, "training_args.pth"))

    #start training 
    best_score_f1 = 0.0
    best_score_precision = 0.0
    best_score_recall = 0.0


    for epoch in range(start_epoch, args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        epoch_time = time.time()
            
        train(model, optimizer, lr_scheduler, criterion, trainloader, epoch,args, use_gpu)

        if ((epoch + 1) % args.eval_epochs == 0 or (epoch == args.num_train_epochs -1)):
            print("\n=====start evaluating of %d epochs=====" %(epoch + 1))
            # output_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            # os.makedirs(output_dir, exist_ok=True)

            precision, recall, f1 = evaluate(model, criterion, testloader, args.testset_dir, args.output_dir, "eval", use_gpu, args)

            print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %(100 * precision, 100 * recall, 100 * f1))

            model_state_dict = model.module.state_dict() if args.n_gpu > 0 else model.state_dict()
            os.system('rm -rf %s/model*.pth'%args.output_dir)
            os.system('rm -rf %s/optimizer*.pth'%args.output_dir)
            os.system('rm -rf %s/scheduler*.pth'%args.output_dir)
            torch.save(model_state_dict, os.path.join(args.output_dir, 'model_epoch_%i_f1_%.2f.pth'%(epoch + 1, 100 * f1)))
            tokenizer.save_pretrained(args.output_dir)
            print("Saving model checkpoint to %s", args.output_dir)
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer_epoch_%i_f1_%.2f.pth"%(epoch + 1, 100 * f1)))
            torch.save(lr_scheduler.state_dict(), os.path.join(args.output_dir, "scheduler_epoch_%i_f1_%.2f.pth"%(epoch + 1, 100 * f1)))
            print("Saving optimizer and scheduler states to %s", args.output_dir)

            if f1 > best_score_f1:
                # Take care of distributed/parallel training
                best_score_f1 = f1
                best_score_precision = precision
                best_score_recall = recall
                model_state_dict = model.module.state_dict() if args.n_gpu > 0 else model.state_dict()
                output_dir = os.path.join(args.output_dir, 'model_best')
                os.makedirs(output_dir, exist_ok=True)
                os.system('rm -rf %s/model*.pth'%output_dir)
                torch.save(model_state_dict, os.path.join(output_dir, 'model_best_epoch_%i_f1_%.2f.pth'%(epoch + 1, 100 * best_score_f1)))
                tokenizer.save_pretrained(output_dir)
                print("Saving model checkpoint to %s"%output_dir)

        epoch_time = time.time() - epoch_time
        print("epoch time footprint: %d hour %d min %d sec" %
              (epoch_time // 3600, (epoch_time % 3600) // 60, epoch_time % 60))

    print("Best precision: %.2f\t recall: %.2f\t f1: %.2f\t" %(100 * best_score_precision, 100 * best_score_recall, 100 * best_score_f1))
    torch.save(args, os.path.join(args.output_dir, "training_args.pth"))

def train(model, optimizer, lr_scheduler, criterion, trainloader, epoch, args, use_gpu):

    model.train()
    step_i = 0
    loss_iter_i = 0
    loss_item = 0
    step_time = time.time()
    steps_by_epoch = int(math.ceil(len(trainloader) // args.gradient_accumulation_steps))
    for step, batch in enumerate(trainloader):
        # use smaller lr to keep the pre-training backbone
        lr_step = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr_step * args.lr_backbone_coef
        optimizer.param_groups[2]['lr'] = lr_step * args.lr_backbone_coef
        input_ids, inputid_ch, inputid_uy, seq_len, entity_start_index_ch,entity_end_index_ch, entity_start_index_uy,entity_end_index_uy, entities_ch,entities_uy,label = batch

        if use_gpu:
            input_ids = input_ids.cuda()
            label = label.cuda()

        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        # if args.bool_attention_mask:
        #     logits = model({'input_ids': input_ids, 'attention_mask': mask.long()})

        logits = model({'input_ids': input_ids,'inputid_ch': inputid_ch,'inputid_uy': inputid_uy,'length':seq_len,'ch_start':entity_start_index_ch,'ch_end':entity_end_index_ch,'uy_start':entity_start_index_uy,'uy_end':entity_end_index_uy})



        loss = criterion(logits, label, mask)
        loss /= args.gradient_accumulation_steps

        if step_i % (args.print_steps * args.gradient_accumulation_steps) == 0:
            print("epoch: %d / %d, steps: %d / %d, lr: %f, loss: %f, speed: %.2f step/s"
                % (epoch, args.num_train_epochs, loss_iter_i, steps_by_epoch, optimizer.param_groups[0]['lr'], 
                loss_item / args.print_steps / args.gradient_accumulation_steps, args.print_steps / (time.time() - step_time)))
            step_time = time.time()
            loss_item = 0

def evaluate(model, criterion, data_loader, test_file_path, save_path, mode, use_gpu, args):
    model.eval()
    probs_all = None
    seq_len_all = None
    tok_to_orig_start_index_all = None
    tok_to_orig_end_index_all = None
    loss_all = 0
    eval_steps = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids, inputid_ch, inputid_uy, seq_len, entity_start_index_ch,entity_end_index_ch, entity_start_index_uy,entity_end_index_uy, entities_ch,entities_uy,label = batch
        if use_gpu:
            input_ids = input_ids.cuda()
            label = label.cuda()
        logits = model({'input_ids': input_ids, 'length': seq_len})

        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        loss = criterion(logits, label, mask)
        probs = torch.sigmoid(logits)
        if probs_all is None:
            probs_all = probs.cpu().detach().numpy()
            seq_len_all = seq_len.numpy()
            tok_to_orig_start_index_all = tok_to_orig_start_index.numpy()
            tok_to_orig_end_index_all = tok_to_orig_end_index.numpy()

    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

        
        

if __name__ == "__main__":
    main()