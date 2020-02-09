# -*- coding: utf-8 -*-
"""Dialog agents for AVSD
"""

import sys
import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from atten import Atten, NaiveAttention
torch.manual_seed(1)


class A_Bot(nn.Module):


    def __init__(self, mm_encoder, history_encoder, input_encoder, response_decoder):
        """ Define model structure
            Args:
                history_encoder (~chainer.Chain): history encoder network
                input_encoder (~chainer.Chain): input encoder network
                response_decoder (~chainer.Chain): response decoder network
        """
        super(A_Bot, self).__init__()
        self.history_encoder = history_encoder
        self.mm_encoder = mm_encoder
        self.input_encoder = input_encoder
        self.response_decoder = response_decoder
        q_embed = 256
        self.s_embed = 256
        self.a_embed = 256
        self.h_embed = 128

        #used for sharing-weights
        high_order_utils = []

        self.emb_s = nn.Conv1d(512, self.s_embed, 1)
        self.emb_a = nn.Conv1d(128, self.a_embed, 1)
        self.emb_temporal_sp = nn.LSTM(self.s_embed, self.s_embed, dropout=0.5, batch_first=True)


        #self.atten = NaiveAttention()
        self.atten = Atten(util_e=[q_embed, self.s_embed, self.s_embed, self.s_embed, self.s_embed, self.a_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[10, 49, 49, 49, 49, 10, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)


    def loss(self, mx, hx, x, y, t, s):
         """ Forward propagation and loss calculation
            Args:
                es (pair of ~chainer.Variable): encoder state
                x (list of ~chainer.Variable): list of input sequences
                y (list of ~chainer.Variable): list of output sequences
                t (list of ~chainer.Variable): list of target sequences
                                   if t is None, it returns only states
            Return:
                es (pair of ~chainer.Variable(s)): encoder state
                ds (pair of ~chainer.Variable(s)): decoder state
                loss (~chainer.Variable) : cross-entropy loss
        """

        ei, ei_len = self.input_encoder(None, x)

        q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
        idx = torch.from_numpy(ei_len - 1).long().cuda()
        batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
        q_prior[batch_index, idx] = 1
        num_samples = s.shape[0]

        s = s.view(-1, s.size(2), s.size(3)).transpose(1, 2) # (256 = 4*64 , 512, 49)
        s = self.emb_s(s)  # print("s shape 3:", s.size()) is (256, 256, 49)
        s = s.view(num_samples, -1, s.size(1), s.size(2)).transpose(2, 3)
        # print("visual embedding before attention:", s.size()) is (4, 64, 49, 256)

        a = mx[0].cuda().permute(1, 2, 0)
        a = self.emb_a(a)
        a = a.transpose(1, 2) # (64, n, 256), n is the longest audio stream length in this batch
        eh_temp, eh = self.history_encoder(None, hx)

        ei = self.atten(utils=[ei, s[0], s[1], s[2], s[3], a, eh_temp], priors=[q_prior, None, None, None, None, None, None])
        a_q = ei[0] #print("question embeding after attention:", a_q.size()) (64, 256)

        a_s = [ei[1], ei[2], ei[3], ei[4]] #print("visual embedding after attention:", ei[1].size()) (64, 256)
        a_a = ei[5] #print("audio embedding after attention:", a_a.size()) (64, 256)
        a_h = ei[6].unsqueeze(0) #print("history after attention dim:", a_h.size()) (1, 64, 128)

        a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s], dim=1)
        _, hidden_temporal_state = self.emb_temporal_sp(a_a_s)
        # eh_temp, eh = self.history_encoder(None, hx)
        #print("history embedding:", eh_temp.size(), eh.size()) #is (1, 64, 128)

        # FUSION
        es = torch.cat((a_q, a_h[-1]), dim=1)





        

