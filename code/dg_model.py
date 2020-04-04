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
import random
from atten import Atten
torch.manual_seed(1)

class MMSeq2SeqModel(nn.Module):

    def __init__(self, history_encoder, a_caption_encoder, question_encoder, answer_encoder, q_summary_decoder):
        """ Define model structure
            Args:
                history_encoder (~chainer.Chain): history encoder network
                input_encoder (~chainer.Chain): input encoder network
                response_decoder (~chainer.Chain): response decoder network
        """
        super(MMSeq2SeqModel, self).__init__()
        #self.mm_encoder = mm_encoder
        self.history_encoder = history_encoder
        self.a_caption_encoder = a_caption_encoder
        self.question_encoder = question_encoder
        self.answer_encoder = answer_encoder
        self.q_summary_decoder = q_summary_decoder
        self.s_embed = 256
        self.a_embed = 256
        self.c_embed = 256
        self.h_embed = 128

        #used for sharing-weights
        high_order_utils = []

        self.a_emb_s = nn.Conv1d(512, self.s_embed, 1) # video for ABOT
        self.a_emb_a = nn.Conv1d(128, self.a_embed, 1) # audio for ABOT
        #self.a_emb_temporal_sp = nn.LSTM(self.s_embed, self.s_embed, dropout=0.5, batch_first=True)
        self.q_emb_s = nn.Conv1d(512, self.s_embed, 1) # video for QBOT
        self.q_emb_temporal_sp = nn.LSTM(self.s_embed, self.s_embed, dropout=0.5, batch_first=True)
        self.lin = nn.Linear(256,128)
        #self.q_emb_q = nn.Conv1d(128, q_embed, 1)
        #self.qalstm = nn.LSTM(128,128)
        #self.soft = nn.Softmax()

        #self.atten = NaiveAttention()
        self.a_atten = Atten(util_e=[self.s_embed, self.s_embed,self.s_embed, self.s_embed, self.a_embed, self.c_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[49, 49, 49, 49, 10, 10, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)
        self.q_atten = Atten(util_e=[self.s_embed, self.s_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[49, 49, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)


    def loss(self, mx, hx, c, q, a, y_s, t_s, s, opt_q, opt_a):
        """ Forward propagation and loss calculation
            Args:
                es (pair of ~chainer.Variable): encoder state
                x (list of ~chainer.Variable): list of input sequences - question
                y (list of ~chainer.Variable): list of output sequences
                t (list of ~chainer.Variable): list of target sequences
                                   if t is None, it returns only states
                mx: audio input
                hx: history input
                c: caption
                y_s & t_s: description
                s: video
                opt_q & opt_a: q/a options
            Return:
                es (pair of ~chainer.Variable(s)): encoder state
                ds (pair of ~chainer.Variable(s)): decoder state
                loss (~chainer.Variable) : cross-entropy loss
        """
        ###################################################################

        qa_id = len(hx)
        #print("hx", type(hx[0]),len(hx))
        #print("q", len(q), q[0].size())
        #print("opt_q", len(opt_q), opt_q[0].size(), opt_q[1].size())
        #print("opt_a", len(opt_q), opt_a[0].size(), opt_a[1].size())
        #eh_temp, eh = self.history_encoder(None, hx)
        round_n = 0
        while qa_id < 11:
            #print("hx", len(hx))
            eh_temp, eh = self.history_encoder(None, hx)

            ########Q-BOT###########################################################################################
            if round_n == 0:

                # visual input for Q BOT
                num_samples = s.shape[0]
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)

            # # Multimodal attention
            #     ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])

            # # Prepare the decoder
            #     a_s_for_q = [ei_for_q[0], ei_for_q[1]]
            #     a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
            #     _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)



            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])
            es_for_q = ei_for_q[2]

            a_s_for_q = [ei_for_q[0], ei_for_q[1]]
            a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
            _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

            #print("check es_for_q size", es_for_q.size()) # 64 * 128
            r_q, _, r_q_test = self.question_encoder(None,q)
            #print("check r_q", r_q.size(), r_q_test.size()) #size:(64,n,128), (1,64,128)
            q_score = torch.bmm(es_for_q.unsqueeze(1),r_q_test.transpose(0,1).transpose(1,2))
            #print("check q_score", q_score.size(), q_score)
            for i in range(99):
                r_qi=[]
                for j in range(len(opt_q)):
                    temp = opt_q[j]
                    r_qi.append(temp[i,:])
                #print("check r_qi:",i,len(r_qi),r_qi[0].size())
                _,_,d_qi = self.question_encoder(None, r_qi)
                #print("d_qi:", d_qi.size())
                q_score_i = torch.bmm(es_for_q.unsqueeze(1),d_qi.transpose(0,1).transpose(1,2))
                #print("check q_score_i", q_score_i.size())
                q_score = torch.cat((q_score,q_score_i),dim=1)
            q_score.squeeze()
            #print("check q_score", q_score.size(), q_score[0,:])

            #q_score_prob1 = self.soft(q_score)
            q_score_prob = F.softmax(q_score) #size(64,100)
            #print("q_score_prob1", q_score_prob1, q_score_prob1.size())
            #print("q_score_prob2", q_score_prob2, q_score_prob2.size())

            q_id = torch.argmax(q_score_prob, dim=1)
            q_id = np.asarray(q_id.squeeze().cpu().detach().numpy())


            ########A-BOT############################################################################################

            if round_n == 0:

                # caption embed for A BOT
                ei_c, ei_len_c = self.a_caption_encoder(None, c)
                #print('ei_c:', ei_c.size())
                #print('ei_len_c:', ei_len_c)
                c_prior = torch.zeros(ei_c.size(0), ei_c.size(1)).cuda()
                idx_c = torch.from_numpy(ei_len_c - 1).long().cuda()
                batch_index_c = torch.arange(0, ei_len_c.shape[0]).long().cuda()
                c_prior[batch_index_c, idx_c] = 1


                # visual input for A BOT
                num_samples = s.shape[0]
                s_for_a = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                # print("s shape 2:", s.size()) is (256 = 4*64 , 512, 49)
                s_for_a = self.a_emb_s(s_for_a)
                # print("s shape 3:", s.size()) is (256, 256, 49)
                s_for_a = s_for_a.view(num_samples, -1, s_for_a.size(1), s_for_a.size(2)).transpose(2, 3)
                 # print("visual embedding before attention:", s.size()) is (4, 64, 49, 256)

                 # Audio input for A BOT
                au = mx[0].cuda().permute(1, 2, 0)
                au = self.a_emb_a(au)
                au = au.transpose(1, 2)

                # # Multimodal attention
                # ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], au, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])

                # #print("question embeding after attention:", a_q.size())
                # #a_s: attended embeddings for 4 visual frames
                # a_s_for_a = [ei[0], ei[1], ei[2], ei[3]]
                # #print("visual embedding after attention:", ei[1].size())
                # #a_a: attended audio embedding
                # a_a = ei[4]
                # #print("audio embedding after attention:", a_a.size())
                # a_c = ei[5]

                # a_h = ei[6]
                #print("a_h and eh dim:", a_h.size(), eh.size())

                #a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                #_, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)

            ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], au, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])
            a_c = ei[5]
            a_h = ei[6]
            #print("check a_h", a_h.size())  # 128
            es_for_a = torch.cat((a_h, r_q_test.squeeze(0)), dim=1)
            #print("check es_for_a", es_for_a.size())  # 256
            es_for_a = self.lin(es_for_a)
            #print("check es_for_a", es_for_a.size())

            _,_,r_a_test = self.answer_encoder(None,a)

            a_score = torch.bmm(es_for_a.unsqueeze(1),r_a_test.transpose(0,1).transpose(1,2))
            #print("check a_score", a_score.size(), a_score)
            for i in range(99):
                r_ai=[]
                for j in range(len(opt_a)):
                    temp = opt_a[j]
                    r_ai.append(temp[i,:])
                #print("check r_ai:",i,len(r_ai),r_ai[0].size())
                _,_,d_ai = self.question_encoder(None, r_ai)
                #print("d_ai:", d_ai.size())
                a_score_i = torch.bmm(es_for_a.unsqueeze(1),d_ai.transpose(0,1).transpose(1,2))
                #print("check a_score_i", a_score_i.size())
                a_score = torch.cat((a_score,a_score_i),dim=1)
            a_score.squeeze()
            #print("check a_score", a_score.size(), a_score[0,:])

            a_score_prob = F.softmax(a_score) 

            a_id = torch.argmax(a_score_prob, dim=1)
            a_id = np.asarray(a_id.squeeze().cpu().detach().numpy())
            # #print("check a_id", type(a_id), a_id)
            # #print(len(opt_a),opt_a[0].size())
            # n1 = len(a_score_prob) # 
            # n2 = opt_a[0].size(1) #n2 has different length
            # print("n1 and n2", n1, n2)
            # r_a = np.zeros((n1,n2))
            # #print("check r_a:", r_a.shape())

            # #for i in range(len(opt_a)):
            # #    r_a[i] = opt_a[i][a_id[i,:]]
            # #print("check r_a:", r_a)

            #print("check update:", len(q_id), len(a_id)) # 64*1
            update_qa = []
            for i in range(len(opt_q)):
                q_idx = q_id[i]
                a_idx = a_id[i]
                #print("idx", i, q_idx, a_idx)
                temp_q = opt_q[i]
                temp_a = opt_a[i]
                #print("temp", temp_q.size(), temp_a.size())
                if q_idx == 0 and a_idx == 0:
                    #print("get q")
                    r_qi = q[i]
                    r_ai = a[i]
                if q_idx != 0 and a_idx == 0:
                    r_qi = temp_q[q_idx-1,:]
                    r_ai = a[i]
                if q_idx == 0 and a_idx != 0:
                    r_qi = q[i]
                    r_ai = temp_a[a_idx-1,:]
                else:
                    r_qi = temp_q[q_idx-1,:]
                    r_ai = temp_a[a_idx-1,:]

                #print("before", r_qi)
                r_qi = r_qi[r_qi.nonzero()].squeeze()
                r_ai = r_ai[r_ai.nonzero()].squeeze()
                #print("r_qi", r_qi.size(), r_qi)
                #print("-----")
                #print("r_ai", r_ai)
                r_qai = torch.cat([r_qi[0:len(r_qi)-1], r_ai])
                update_qa.append(r_qai)
                #print("r_qai", r_qai)

            hx.append(update_qa)
            qa_id += 1
            round_n += 1

            #r_p = torch.cat((r_q,r_a), dim=1).transpose(0,1)
            #_, (r_p, _) = self.qalstm(r_p)
            #eh_temp = torch.cat((eh_temp, r_p.transpose(0,1)), dim=1)

###################################################################################################################

        if qa_id == 11:
                es_final = es_for_q
            #print("check point 4 - es_final size:", es_final.size())
            #generate summary 
                if hasattr(self.q_summary_decoder, 'context_to_state') \
                    and self.q_summary_decoder.context_to_state==True:
                    ds_q, dy_q = self.q_summary_decoder(es_final, None, y_s) 
                else:
                    # decode
                    ds_q, dy_q = self.q_summary_decoder(hidden_temporal_state_for_q, es_final, y_s)

                # compute loss
                if t_s is not None:
                    tt = torch.cat(t_s, dim=0)
                    loss = F.cross_entropy(dy_q, torch.tensor(tt, dtype=torch.long).cuda())
                    #max_index = dy.max(dim=1)[1]
                    #hit = (max_index == torch.tensor(tt, dtype=torch.long).cuda()).sum()
                    #cul_loss += loss
                    return None, ds_q, loss
                else:  # if target is None, it only returns states
                    return None, ds_q
################################################################################################### 

    def generate(self, mx, hx, c, q, a, s, opt_q, opt_a, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1):
        """ Generate sequence using beam search
            Args:
                es (pair of ~chainer.Variable(s)): encoder state
                x (list of ~chainer.Variable): list of input sequences
                sos (int): id number of start-of-sentence label
                eos (int): id number of end-of-sentence label
                unk (int): id number of unknown-word label
                maxlen (int): list of target sequences
                beam (int): list of target sequences
                penalty (float): penalty added to log probabilities
                                 of each output label.
                nbest (int): number of n-best hypotheses to be output
            Return:
                list of tuples (hyp, score): n-best hypothesis list
                 - hyp (list): generated word Id sequence
                 - score (float): hypothesis score
                pair of ~chainer.Variable(s)): decoder state of best hypothesis
        """


        qa_id = len(hx)
        round_n = 0
        while qa_id < 11:

            eh_temp, eh = self.history_encoder(None, hx)
        # ###########Q BOT##################################################
            if round_n == 0:

                # visual input for Q BOT
                num_samples = s.shape[0]
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)


            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])
            es_for_q = ei_for_q[2]
            a_s_for_q = [ei_for_q[0], ei_for_q[1]]
            a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
            _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

            #print("check es_for_q size", es_for_q.size()) # 64 * 128
            r_q, _, r_q_test = self.question_encoder(None,q)
            #print("check r_q", r_q.size(), r_q_test.size()) #size:(64,n,128), (1,64,128)
            q_score = torch.bmm(es_for_q.unsqueeze(1),r_q_test.transpose(0,1).transpose(1,2))
            #print("check q_score", q_score.size(), q_score)
            for i in range(99):
                r_qi=[]
                for j in range(len(opt_q)):
                    temp = opt_q[j]
                    r_qi.append(temp[i,:])
                #print("check r_qi:",i,len(r_qi),r_qi[0].size())
                _,_,d_qi = self.question_encoder(None, r_qi)
                #print("d_qi:", d_qi.size())
                q_score_i = torch.bmm(es_for_q.unsqueeze(1),d_qi.transpose(0,1).transpose(1,2))
                #print("check q_score_i", q_score_i.size())
                q_score = torch.cat((q_score,q_score_i),dim=1)
            q_score.squeeze()
            #print("check q_score", q_score.size(), q_score[0,:])

            #q_score_prob1 = self.soft(q_score)
            q_score_prob = F.softmax(q_score) #size(64,100)
            #print("q_score_prob1", q_score_prob1, q_score_prob1.size())
            #print("q_score_prob2", q_score_prob2, q_score_prob2.size())

            q_id = torch.argmax(q_score_prob, dim=1)
            q_id = np.asarray(q_id.squeeze().cpu().detach().numpy())


        ###################################################################################
        #################   A bot #####################

            if round_n == 0:

                # caption embed for A BOT
                ei_c, ei_len_c = self.a_caption_encoder(None, c)
                c_prior = torch.zeros(ei_c.size(0), ei_c.size(1)).cuda()
                idx_c = torch.from_numpy(ei_len_c - 1).long().cuda()
                batch_index_c = torch.arange(0, ei_len_c.shape[0]).long().cuda()
                c_prior[batch_index_c, idx_c] = 1


                # visual input for A BOT
                num_samples = s.shape[0]
                s_for_a = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                # print("s shape 2:", s.size()) is (256 = 4*64 , 512, 49)
                s_for_a = self.a_emb_s(s_for_a)
                # print("s shape 3:", s.size()) is (256, 256, 49)
                s_for_a = s_for_a.view(num_samples, -1, s_for_a.size(1), s_for_a.size(2)).transpose(2, 3)
                 # print("visual embedding before attention:", s.size()) is (4, 64, 49, 256)

                 # Audio input for A BOT
                au = mx[0].cuda().permute(1, 2, 0)
                au = self.a_emb_a(au)
                au = au.transpose(1, 2)

            ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], au, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])
            a_c = ei[5]
            a_h = ei[6]
            es_for_a = torch.cat((a_h, r_q_test.squeeze(0)), dim=1)
            es_for_a = self.lin(es_for_a)

            _,_,r_a_test = self.answer_encoder(None,a)

            a_score = torch.bmm(es_for_a.unsqueeze(1),r_a_test.transpose(0,1).transpose(1,2))
            #print("check a_score", a_score.size(), a_score)
            for i in range(99):
                r_ai=[]
                for j in range(len(opt_a)):
                    temp = opt_a[j]
                    r_ai.append(temp[i,:])
                #print("check r_ai:",i,len(r_ai),r_ai[0].size())
                _,_,d_ai = self.question_encoder(None, r_ai)
                #print("d_ai:", d_ai.size())
                a_score_i = torch.bmm(es_for_a.unsqueeze(1),d_ai.transpose(0,1).transpose(1,2))
                #print("check a_score_i", a_score_i.size())
                a_score = torch.cat((a_score,a_score_i),dim=1)
            a_score.squeeze()
            #print("check a_score", a_score.size(), a_score[0,:])

            a_score_prob = F.softmax(a_score) 

            a_id = torch.argmax(a_score_prob, dim=1)
            a_id = np.asarray(a_id.squeeze().cpu().detach().numpy())

            #print("check update:", len(q_id), len(a_id)) # 64*1
            update_qa = []
            #print("idx", i, q_id, a_id)
            #print(len(opt_q), len(opt_a), opt_a[0].size())
            for i in range(len(opt_q)):
                q_idx = q_id
                a_idx = a_id
                #print("idx", i, q_idx, a_idx)
                temp_q = opt_q[i]
                temp_a = opt_a[i]
                #print("temp", temp_q.size(), temp_a.size())
                if q_idx == 0 and a_idx == 0:
                    #print("get q")
                    r_qi = q[i]
                    r_ai = a[i]
                if q_idx != 0 and a_idx == 0:
                    r_qi = temp_q[q_idx-1,:]
                    r_ai = a[i]
                if q_idx == 0 and a_idx != 0:
                    r_qi = q[i]
                    r_ai = temp_a[a_idx-1,:]
                else:
                    r_qi = temp_q[q_idx-1,:]
                    r_ai = temp_a[a_idx-1,:]

                #print("before", r_qi)
                r_qi = r_qi[r_qi.nonzero()].squeeze()
                r_ai = r_ai[r_ai.nonzero()].squeeze()
                #print("r_qi", r_qi.size(), r_qi)
                #print("-----")
                #print("r_ai", r_ai)
                r_qai = torch.cat([r_qi[0:len(r_qi)-1], r_ai])
                update_qa.append(r_qai)
                #print("r_qai", r_qai)

            hx.append(update_qa)
            qa_id += 1
            round_n += 1


        if qa_id == 11:
                es_final = es_for_q

        # beam search
        #print('es_final', es_final.size())
        ds = self.q_summary_decoder.initialize(hidden_temporal_state_for_q, es_final, torch.from_numpy(np.asarray([sos])).cuda())
        hyplist = [([], 0., ds)]
        best_state = None
        comp_hyplist = []
        for l in six.moves.range(maxlen):
            new_hyplist = []
            argmin = 0
            for out, lp, st in hyplist:
                logp = self.q_summary_decoder.predict(st)
                lp_vec = logp.cpu().data.numpy() + lp
                lp_vec = np.squeeze(lp_vec)
                if l >= minlen:
                    new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                    new_st = self.q_summary_decoder.update(st, torch.from_numpy(np.asarray([eos])).cuda())
                    comp_hyplist.append((out, new_lp))
                    if best_state is None or best_state[0] < new_lp:
                        best_state = (new_lp, new_st)

                for o in np.argsort(lp_vec)[::-1]:
                    if o == unk or o == eos:  # exclude <unk> and <eos>
                        continue
                    new_lp = lp_vec[o]
                    if len(new_hyplist) == beam:
                        if new_hyplist[argmin][1] < new_lp:
                            new_st = self.q_summary_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                            new_hyplist[argmin] = (out + [o], new_lp, new_st)
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                        else:
                            break
                    else:
                        new_st = self.q_summary_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                        new_hyplist.append((out + [o], new_lp, new_st))
                        if len(new_hyplist) == beam:
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

            hyplist = new_hyplist
        #print("hyplist", hyplist)
        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]

            return maxhyps, best_state[1]
        else:
            return [([], 0)], None

