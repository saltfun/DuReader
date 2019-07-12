import os
import args
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim
from GPUtil import showUtilization as gpu_usage

import evaluate
from optimizer import BertAdam
from dataset.dataloader import Dureader
from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model_dir.modeling import BertForQuestionAnswering, BertConfig

# 随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)
class FYExample(object):
    def __init__(self,
                 question_tokens,
                 doc_tokens,
                 start_position=None,
                 end_position=None,
                 answer_text=None,
                 id=None,
                 context=None):
        """
        tokenized完的例子
        无法回答：start = 1， end = 0
        yes： start = 2， end = 1
        no： start = 3， end = 2
        """
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.id = id
        self.context = context


class FYFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 id=None,
                 start_point=0,
                 only_tokenized_text=None,
                 context=None):
        """
        变成输入到模型中的features的例子
        无法回答：start = 1， end = 0
        yes： start = 2， end = 1
        no： start = 3， end = 2
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.id = id
        self.start_point = start_point
        self.only_tokenized_text = only_tokenized_text
        self.context = context


def train():
    # 加载预训练bert
    model = BertForQuestionAnswering.from_pretrained('/home/xiaxiaozheng/Documents/data/lawyia/chinese-bert_chinese_wwm_pytorch',
                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))
    device = args.device
    model.to(device)

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=args.num_train_optimization_steps)

    # 准备数据
    # data = Dureader()
    # train_dataloader, dev_dataloader = data.train_iter, data.dev_iter
    # print('data prepare complete, length of features:', len(train_dataloader) + len(dev_dataloader))


    # 方案2 /home/xiaxiaozheng/Documents/shyyer/CAIL_2019_Code
    USE_EXISTING_DATA = True
    if USE_EXISTING_DATA:
        train_dataloader = torch.load('/home/xiaxiaozheng/Documents/shyyer/CAIL_2019_Code/data/features_train.pt')
        dev_dataloader = torch.load('/home/xiaxiaozheng/Documents/shyyer/CAIL_2019_Code/data/features_valid.pt')
        print('data prepare complete, length of features:', len(train_dataloader) + len(dev_dataloader))
        print(len(train_dataloader))
        print(len(dev_dataloader))
    best_loss = 0  #  100000.0
    model.train()
    for i in range(args.num_train_epochs):
        for step in range(len(train_dataloader) // args.batch_size):
            # current_step = step + args.save_step + \
            #                epoch * len(features_train) + 1
            # current_epoch = current_step // len(features_train)

            batch = train_dataloader[step * args.batch_size: (step + 1) * args.batch_size]
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch")):

            input_ids = torch.Tensor([feature.input_ids for feature in batch]).long().to(device)
            segment_ids = torch.Tensor([feature.segment_ids for feature in batch]).long().to(device)
            input_mask = torch.Tensor([feature.input_mask for feature in batch]).long().to(device)
            start_positions = torch.Tensor([feature.start_position for feature in batch]).long().to(device)
            end_positions = torch.Tensor([feature.end_position for feature in batch]).long().to(device)

            # input_ids, input_mask, segment_ids, start_positions, end_positions = \
            #                             batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            # input_ids, input_mask, segment_ids, start_positions, end_positions = \
            #                             input_ids.to(device), input_mask.to(device), segment_ids.to(device), start_positions.to(device), end_positions.to(device)
            # print("GPU Usage after deleting the Tensors")
            # gpu_usage()
            # input_ids = torch.Tensor([feature.input_ids for feature in batch]).long().to(device)
            # 计算loss
            model.eval()
            loss, _, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # print("GPU Usage after deleting the Tensors")
            # gpu_usage()

            # print("GPU Usage after emptying the cache")
            torch.cuda.empty_cache()
            # gpu_usage()
            # 更新梯度
            if (step+1) % args.gradient_accumulation_steps == 0:
                with torch.no_grad():
                    optimizer.step()
                    optimizer.zero_grad()
                    model.eval()

            # 验证
            if step % args.log_step == 4:
                eval_loss = evaluate.evaluate2(model, dev_dataloader)
                print(eval_loss)
                model_file = os.path.join('./model_dir/', 'model_{}_{}.pt'.format(i, eval_loss))
                if eval_loss > best_loss:
                    best_loss = eval_loss
                    torch.save(model, model_file)
                    # torch.save(model.state_dict(), model_file)
                    model.train()

if __name__ == "__main__":
    train()
