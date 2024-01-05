# -*- coding: utf-8 -*-
"""
GlobalPointer参考: https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/models/GlobalPointer.py
稀疏多标签交叉熵损失参考: bert4keras源码
"""
import torch
import torch.nn as nn
import numpy as np
def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    https://kexue.fm/archives/8888
    '''
    #y_pred[batchsize,type,maxlen,maxlen]
    #y_true[batchsize,2/type,实体对个数,2] y_true[0]为头实体,y_true[1]为尾实体
    shape = y_pred.shape
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]#shape[2]为maxlen，位置数字=x*maxlen+x #[batch,type,实体对个数]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))#[batch,type,maxlen*maxlen]
    zeros = torch.zeros_like(y_pred[...,:1])#全0矩阵[batch,type,1]
    y_pred = torch.cat([y_pred, zeros], dim=-1)#[batch,type,maxlen*maxlen+1]这里应该是把全1加到最后一列了
    if mask_zero:
        infs = zeros + 1e12#全inf矩阵[batch,tyoe,1]
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)#此时y_pred[...,0]为全inf，y_pred[...,-1]为全0
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)#[batch,type,实体对个数] 这时正常位置会给出得分值，而0位置（补位）给出inf
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)#[batch,type,实体对个数+1]
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)#此时y_pred[...,0]为全-inf，y_pred[...,-1]为全0
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)#[batch,type,实体对个数] 这时正常位置会给出得分值，而0位置（补位）给出-inf
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss#对应公式正类-全部类，即b-a
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)#对应公式1-exp(b-a)
    neg_loss = all_loss + torch.log(aux_loss)#对应公式负类得分
    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss

class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size#分类数量
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize #应该与bert最后一层输出保持相同 
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)#[maxlen,1]

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)#[dim//2]
        indices = torch.pow(10000, -2 * indices / output_dim)#[dim//2]
        embeddings = position_ids * indices#[maxlen,dim//2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)#[maxlen,dim//2,2]
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))#[batch_size,maxlen,dim//2,2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))#[batch_size,maxlen,dim]
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, context_outputs,  attention_mask):
        self.device = attention_mask.device #device
        last_hidden_state = context_outputs[0] #[batchsize,maxlen,1024]1024为预训练bert模型最后一层的dim
        batch_size = last_hidden_state.size()[0] #batchsize
        seq_len = last_hidden_state.size()[1] #maxlen
        outputs = self.dense(last_hidden_state)#[batchsize,maxlen,1024] =》 #[batchsize,maxlen,type_num*64*2]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)#切片 tuple( type_num个[batchsize,maxlen,inner_dim * 2])
        outputs = torch.stack(outputs, dim=-2)#堆叠 [batchsize,maxlen,type_num,2*64]
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]#切开，均为[batchsize,maxlen,type_num,64]
        if self.RoPE:#只有计算ner时需要使用
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)#[batch_size,maxlen,dim]
            #::2 两个冒号直接写表示从所有的数据中隔行取数据。从0开始,1::2 两个冒号直接写表示从所有的数据中隔行取数据。从1开始,None 增加一维
            #repeat_interleave操作：复制指定维度的信息
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)# 是将奇数列信息抽取出来也就是cosm 拿出来并复制[4, 120, 1, 64]
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)# 是将偶数列信息抽取出来也就是sinm 拿出来并复制[4, 120, 1, 64]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)#奇数列乘上-1 #[4, 120, 2, 64]
            qw2 = qw2.reshape(qw.shape) 
            qw = qw * cos_pos + qw2 * sin_pos#融入位置信息
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos#融入位置信息
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)#bhmd*bhdn=>bhmn 先转置，再相乘
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12#有字的地方保留logits原数据，其他数据*0-1e12，相当于很大的负数
        # 排除下三角#只有计算ner时需要使用
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)#只保留下三角的全1，对角线和上三角全为0，若为0 对角线也会保留
            logits = logits - mask * 1e12#删除所有下三角的数据

        return logits / self.inner_dim ** 0.5