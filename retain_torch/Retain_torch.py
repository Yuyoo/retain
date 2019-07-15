# -*- coding: utf-8 -*-
# @Time    : 2019-07-15 15:21
# @Author  : Yuyoo
# @Email   : sunyuyaoseu@163.com
# @File    : Retain_torch.py


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Retain(nn.Module):
    def __init__(self, inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, outputDimSize, keep_prob=0.8):
        super(Retain, self).__init__()
        self.inputDimSize = inputDimSize
        self.embDimSize = embDimSize
        self.alphaHiddenDimSize = alphaHiddenDimSize
        self.betaHiddenDimSize = betaHiddenDimSize
        self.outputDimSize = outputDimSize
        self.keep_prob = keep_prob

        self.embedding = nn.Linear(self.inputDimSize, self.embDimSize)
        self.dropout = nn.Dropout(self.keep_prob)
        self.gru_alpha = nn.GRU(self.embDimSize, self.alphaHiddenDimSize)
        self.gru_beta = nn.GRU(self.embDimSize, self.betaHiddenDimSize)
        self.alpha_att = nn.Linear(self.alphaHiddenDimSize, 1)
        self.beta_att = nn.Linear(self.betaHiddenDimSize, self.embDimSize)
        self.out = nn.Linear(self.embDimSize, self.outputDimSize)

    def initHidden_alpha(self, batch_size):
        return torch.zeros(1, batch_size, self.alphaHiddenDimSize, device=torch.device('cuda:1'))
    
    def initHidden_beta(self, batch_size):
        return torch.zeros(1, batch_size, self.betaHiddenDimSize, device=torch.device('cuda:1'))

    # 两个attention的处理，其中att_timesteps是目前为止的步数
    # 返回的是一个3维向量，维度为(n_timesteps × n_samples × embDimSize)
    def attentionStep(self, h_a, h_b, att_timesteps):
        """
        两个attention的处理，其中att_timesteps是目前为止的步数
        返回的是一个3维向量，维度为(n_timesteps × n_samples × embDimSize)
        :param h_a: 
        :param h_b: 
        :param att_timesteps: 
        :return: 
        """
        reverse_emb_t = self.emb[:att_timesteps].flip(dims=[0])
        reverse_h_a = self.gru_alpha(reverse_emb_t, h_a)[0].flip(dims=[0]) * 0.5
        reverse_h_b = self.gru_beta(reverse_emb_t, h_b)[0].flip(dims=[0]) * 0.5

        preAlpha = self.alpha_att(reverse_h_a)
        preAlpha = torch.squeeze(preAlpha, dim=2)
        alpha = torch.transpose(F.softmax(torch.transpose(preAlpha, 0, 1)), 0, 1)
        beta = torch.tanh(self.beta_att(reverse_h_b))

        c_t = torch.mean((alpha.unsqueeze(2) * beta * self.emb[:att_timesteps]), dim=0)
        return c_t

    def forward(self, x):
        first_h_a = self.initHidden_alpha(x.shape[1])
        first_h_b = self.initHidden_beta(x.shape[1])
        
        self.emb = self.embedding(x)
        if self.keep_prob < 1:
            self.emb = self.dropout(self.emb)

        count = np.arange(x.shape[0]) + 1
        self.c_t = torch.zeros_like(self.emb)  # shape=(seq_len, batch_size, day_dim)
        for i, att_timesteps in enumerate(count):
            # 按时间步迭代，计算每个时间步的经attention的gru输出
            self.c_t[i] = self.attentionStep(first_h_a, first_h_b,att_timesteps)

        if self.keep_prob < 1.0:
            self.c_t = self.dropout(self.c_t)

        # output层
        y_hat = self.out(self.c_t)
        y_hat = torch.sigmoid(y_hat)
        return y_hat

    def padTrainMatrix(self,seqs):
        """
        对原始的seqs(patients × visits × medical code)进行处理，得到便于RNN训练的结构（maxlen × n_samples × inputDimSize）
        返回的length表示这个seqs中所有病人对应的visit的数目
        :param seqs: 
        :return: 
        """
        lengths = np.array( [ len(seq) for seq in seqs ] ).astype("int32")
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros([maxlen,n_samples,self.inputDimSize]).astype(np.float32)
        for idx,seq in enumerate(seqs):
            for xvec,subseq in zip(x[:,idx,:],seq):
                for tuple in subseq:
                    xvec[tuple[0]] = tuple[1]
        return x,lengths