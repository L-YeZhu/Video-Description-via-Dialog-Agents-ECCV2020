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
from atten import Atten, NaiveAttention, H_Q_Attention_q, H_Q_Attention_a, H_Q_Attention_g, IM_Attention
torch.manual_seed(1)


class MMSeq2SeqModel(nn.Module):

    def __init__(self, mm_encoder, a_history_encoder, a_caption_encoder, a_input_encoder, a_response_decoder, q_summary_decoder, q_question_decoder):
        """ Define model structure
            Args:
                history_encoder (~chainer.Chain): history encoder network
                input_encoder (~chainer.Chain): input encoder network
                response_decoder (~chainer.Chain): response decoder network
        """
        super(MMSeq2SeqModel, self).__init__()
        self.mm_encoder = mm_encoder
        self.a_history_encoder = a_history_encoder
        self.a_caption_encoder = a_caption_encoder
        self.a_input_encoder = a_input_encoder ## question encoder
        self.a_response_decoder = a_response_decoder ## answer decoder
        self.q_summary_decoder = q_summary_decoder
        self.q_question_decoder = q_question_decoder
        #self.summary_decoder = summary_decoder
        q_embed = 256
        self.s_embed = 256
        self.a_embed = 256
        self.c_embed = 256
        self.h_embed = 128

        #used for sharing-weights
        high_order_utils = []

        self.a_emb_s = nn.Conv1d(512, self.s_embed, 1)
        self.a_emb_a = nn.Conv1d(128, self.a_embed, 1)
        self.a_emb_temporal_sp = nn.LSTM(self.s_embed, 128, dropout=0.5, batch_first=True)
        self.q_emb_s = nn.Conv1d(512, self.s_embed, 1)
        #self.q_emb_a = nn.Conv1d(128, self.a_embed, 1)
        self.q_emb_temporal_sp = nn.LSTM(self.s_embed, 128, dropout=0.5, batch_first=True)
        self.q_emb_q = nn.Conv1d(128, q_embed, 1)
        self.lin1 = nn.Linear(384, 256)
        self.lin2 = nn.Linear(256, 128)
        self.a_dec = nn.LSTM(512, 128, dropout=0.5, batch_first= True)
        self.q_dec = nn.LSTM(128, 128, dropout=0.5, batch_first= True)

        #self.atten = NaiveAttention()
        self.a_atten = Atten(util_e=[q_embed, self.s_embed, self.s_embed,self.s_embed, self.s_embed, self.a_embed, self.c_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[10, 49, 49, 49, 49, 10, 10, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)
        self.q_atten = Atten(util_e=[self.s_embed, self.s_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[49, 49], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)

        #EXPERIMENTS
        self.q_pair_atten = H_Q_Attention_q()
        self.a_pair_atten = H_Q_Attention_a()
        self.g_pair_atten = H_Q_Attention_g()
        self.im_atten = IM_Attention()


    def loss(self, mx, hx, x, c, y_a, y_q, y_s, t_a, t_q, t_s, s, all_y_a, all_y_q):
        """ Forward propagation and loss calculation
            Args:
                es (pair of ~chainer.Variable): encoder state
                x (list of ~chainer.Variable): list of input sequences - question
                y (list of ~chainer.Variable): list of output sequences
                t (list of ~chainer.Variable): list of target sequences
                                   if t is None, it returns only states
            Return:
                es (pair of ~chainer.Variable(s)): encoder state
                ds (pair of ~chainer.Variable(s)): decoder state
                loss (~chainer.Variable) : cross-entropy loss
        """
        ##################################################################################
        # batch_size = len(all_y_a)
        # qa_id = len(hx)
        # # iter_count = 0
        # # y_a_i = []
        # pos_batch = []
        # # print(all_y_a[1])
        # for i in range(batch_size):
        #     iner_count = 0
        #     pos_j = []
        #     for j in range(len(all_y_a[i])):
        #         if all_y_a[i][j] == 2:
        #             pos_j.append(j)
        #     #print(pos_j)
        #     pos_batch.append(pos_j)
        # #print(len(pos_batch))

        # print(pos_batch[1])

        ###################################################################################
        #################   A bot #####################

        qa_id = len(hx)
        #cul_loss = 0
        #print("history length:",len(hx))
        iter_count = 0
        while qa_id < 11:
            #iter_count = 0
            #cul_loss = 0

            if iter_count == 0:
        # question embed for A BOT

                #cul_loss = 0
                ei, ei_len = self.a_input_encoder(None, x)
        # print("question embedding before attention:", ei.size())
        # print("question embedding example:", ei[0,:,:])
                q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
                idx = torch.from_numpy(ei_len - 1).long().cuda()
                batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
                q_prior[batch_index, idx] = 1

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
                a = mx[0].cuda().permute(1, 2, 0)
                a = self.a_emb_a(a)
                a = a.transpose(1, 2)


        # History embed for A BOT
                eh_temp_for_a, eh_for_a = self.a_history_encoder(None, hx)
                #print("eh_temp_for_a:", eh_temp_for_a.size())

        # Multimodal attention
        #ei = self.atten(utils=[s[0], s[3]], priors=[None, None])
                #ei, ei_len = self.a_input_encoder(None, x)
                ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])

        #a_q: attended question embeding
                a_q = ei[0]
        #print("question embeding after attention:", a_q.size())
        #a_s: attended embeddings for 4 visual frames
                a_s_for_a = [ei[1], ei[2], ei[3], ei[4]]
        #print("visual embedding after attention:", ei[1].size())
        #a_a: attended audio embedding
                a_a = ei[5]
        #print("audio embedding after attention:", a_a.size())
                a_c = ei[6]
        #print("a_h and eh dim:", a_h.size(), eh.size())
                a_h = ei[7]

                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
            #print("a_a_s:",a_a_s.size())
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
                a_q = self.lin2(a_q)
                #print("test1", test1.size())
                #print(a_a_s)
                #print(test1)



                #s2 = a_q 
        #eh_temp, eh = self.history_encoder(None, hx)
        #print("history embedding:", eh_temp.size(), eh.size()) #is (1, 64, 128)

            #print("check 1:", a_h.size(), a_c.size()) #(64, 128) and (64,256)
            # ei, ei_len = self.a_input_encoder(None, x)
            ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])
            a_h = ei[7]
            a_c = ei[6]

            s1 = torch.cat((a_h , a_c), dim=1) # 384
            s1 = self.lin1(s1)

            # pair_att_for_a = self.im_atten(s1, s2)
            # print("check:", pair_att_for_a.size())
            #print(eh_temp_for_a.size(), s1.size())
            pair_att_for_a = self.a_pair_atten(eh_temp_for_a, s1)
            #print(eh_temp_for_a.size(), s1.size(), pair_att_for_a.size())
        #print("check atten_test:", test_att_out, test_att_out.size())

        # concatenate encodings
        # es = ei[6]
        #print("es size:", es.size())
        # es = a_h.squeeze()
        #es = a_q
            #print("size check:", iter_count, a_q.size(), a_h.size())
            #pair_att_for_a = 
            #es_for_a = torch.cat((a_q, a_c, pair_att_for_a[-1]), dim=1)
            #a_q = dq
            #print("check", a_q.size())
            es_for_a = torch.cat((a_q, a_c, pair_att_for_a[-1]), dim=1)
            #es_for_a = torch.cat((a_q, a_c, a_h.squeeze(0)), dim=1)
        ## caption + his_with_att
        #es = torch.cat((a_q, a_h[-1]), dim=1)

        #generate answer dy for the given question
            #print("check point 1 - es_for_a size:", es_for_a.size())
            es_for_a = es_for_a.unsqueeze(1)
            #print("check point 1 - es_for_a size:", es_for_a.size())
            # if hasattr(self.a_response_decoder, 'context_to_state') \
            #     and self.a_response_decoder.context_to_state==True:
            #     _,  _, dy = self.a_response_decoder(es_for_a, None, y_a) 
            # else:
            # decode
            # print("check size:", y_a.size(), y_s.size(), c.size(), y_q.size())
            #print("y_a:",y_a, len(y_a))
            #_, _, dy = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, y_a)
            #print
            da, _ = self.a_dec(es_for_a)
            #print("y_a", len(y_a))
            #tem_lin = nn.Linear(694, len(y_a))
            #print("check a", a.size()) # is (694, 5645)
            #num = a.size()[0]
            # pos = a.argmax(dim=1)
            # print(pos.size())
            # print(pos)

            #print("check da", da.size()) # is (batch_n, 1, 128)


        ##############################################################################################
        ###########Q BOT##################################################

            if iter_count == 0:

                # visual input for Q BOT
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)

                # history embed for Q BOT
                #eh_temp_for_q, eh_for_q = self.a_history_encoder(None, hx)

        # Multimodal attention
                ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])

            # Prepare the decoder
            #ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])
                a_s_for_q = [ei_for_q[0], ei_for_q[1]]
                a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
                _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

                a_h_for_q = ei_for_q[2]
            #initial_ah = a_h_for_q

        # Pair level attention between history and answer generated by A BOT
        # print("eh_temp_for_q and dy size:", eh_temp_for_q.size(), dy.size())

            #pair_att_for_q = self.q_pair_atten(eh_temp_for_a, dy)
            #print(eh_for_q.size(), pair_att_for_q[-1].size())

            #es_for_q = torch.cat((eh_for_q.squeeze(0), pair_att_for_q[-1]), dim=1)
        #es_for_q = pair_att_for_q[-1]
            #es_for_q = torch.cat((a_h_for_q, pair_att_for_q[-1]), dim=1)
            #pair_att_for_q = self.q_pair_atten(eh_temp_for_a, dy)
            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])
            a_h_for_q = ei_for_q[2]
            es_for_q = a_h_for_q
            #a_q = dq



        # Q generate next question dq
            #print("check point 2 - es_for_q size:", es_for_q.size())
            # if hasattr(self.q_question_decoder, 'context_to_state') \
            #     and self.q_question_decoder.context_to_state==True:
            #     _,  _, dq = self.q_question_decoder(es_for_q, None, y_s) 
            # else:
            #     # decode

            #_, q, dq = self.q_question_decoder(hidden_temporal_state_for_q, es_for_q, y_q)
            #print("check q:", q.size())

            # dq = dq.unsqueeze(0)
            es_for_q = es_for_q.unsqueeze(1)
            dq, _ = self.q_dec(es_for_q)
            #print(es_for_q.size(),dq.size())

            #print()
            #eh_temp_for_a = torch.cat((eh_for_a.transpose(0,1),dq,dy),dim=1)
            eh_temp_for_a = torch.cat((eh_temp_for_a, da, dq), dim=1)
            eh_temp_for_q = eh_temp_for_a
            a_q = dq.squeeze(1)
            #a_q = self.lin1(a_q)

            #######UPDATE INPUT SEQUENCE LABEL FOR Y_A and Y_Q############

            # y_a = []
            # y_q = []
            # pos_a = a.argmax(dim=1)
            # pos_q = q.argmax(dim=1)
            # l_a = pos_a.size()[0] / batch_size
            # l_q = pos_q.size()[0] / batch_size
            # #print(torch.split(pos_a, 10, dim=0)[0])
            # split_a = torch.split(pos_a, l_a, dim=0)
            # split_q = torch.split(pos_q, l_q, dim=0)
            # #print(len(split_a))
            # for i in range(batch_size):
            #     # eos = torch.tensor(2)
            #     temp_a = split_a[i]
            #     temp_q = split_q[i]
            #     temp_a[0] = 2
            #     temp_q[0] = 2
            #     y_a.append(temp_a)
            #     y_q.append(temp_q)
            #print(y_a)
            #     pos_s = i * l_a
            #     pos_e = i * l_a + l_a
            #     #print(pos_a)
            #     temp_a = torch.cat((2, pos_a[pos_s: pos_e]))
            #     y_a.append(temp_a)
            # print(y_a)
            # pos_q = q.argmax(dim=1)
            # l_q = pos_q.size()[0]
            # for i in range(batch_size):
            #     pos_s = i * l_q
            #     pos_e = i * l_q + l_q
            #     temp_q = torch.cat((2, pos_q[pos_s: pos_e]))
            #     y_q.append(temp_q)
            # print(y_q)



            #a_q = dq
            qa_id += 1
            iter_count += 1


            #pair_att = self.g_pair_atten(dy, dq)
        # #pair_att = soft(pair_att)
        # #print("check point 3 - a_h_for_q and pair_att size:", a_h_for_q.size(), pair_att.size())

        #     es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]), dim=1)

        # #print("check point 4 - es_final size:", es_final.size())


        # #generate summary 
        #     if hasattr(self.q_summary_decoder, 'context_to_state') \
        #         and self.q_summary_decoder.context_to_state==True:
        #         ds_q, dy_q = self.q_summary_decoder(es_final, None, y_s) 
        #     else:
        #         # decode
        #         ds_q, dy_q = self.q_summary_decoder(hidden_temporal_state_for_q, es_final, y_s)

        #     # compute loss
        #     if t_s is not None:
        #         tt = torch.cat(t_s, dim=0)
        #         loss = F.cross_entropy(dy_q, torch.tensor(tt, dtype=torch.long).cuda())
        #         #max_index = dy.max(dim=1)[1]
        #         #hit = (max_index == torch.tensor(tt, dtype=torch.long).cuda()).sum()
        #         prob = random.random()
        #         if prob < 0.1 and iter_count != 0:
        #             cul_loss += loss*1
        #         if iter_count == 0:
        #             cul_loss = loss
        # #         else:
        # #             cul_loss = cul_loss
        #         # cul_loss += loss
        #     # if target is None, it only returns states
        #     qa_id += 1
        #     iter_count += 1

        # if qa_id == 11:
        #     if t_s is not None:
        #         #print("check loss 1 and 2:", loss, cul_loss)
        #         #print("check internal loss:", len(hx), cul_loss, loss)
        #         return None, ds_q, cul_loss
        #     else:
        #         return None, ds_q



        ### prepare updated input for summary decoder
        # dy is generared answer, dq is generated question
        #print("dy and dq size in training:", dy.size(), dq.size())
        #print("current history size:", eh_temp_for_q.size(), eh_for_q.size())
########################################################################################
        if qa_id == 11:
                #pair_att = self.g_pair_atten(dy, dq)
            #pair_att = soft(pair_att)
            #print("check point 3 - a_h_for_q and pair_att size:", a_h_for_q.size(), pair_att.size())

                #es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]), dim=1)
                #es_final = torch.cat(a_h_for_q, 

                #print("a_h_for_q", a_h_for_q.size(), da.size(), dq.size()) # 128*3
                es_final = torch.cat((a_h_for_q, da.squeeze(1), dq.squeeze(1)), dim = 1)
            #es_final = soft(es_final)

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
##################################################################################################


    def generate(self, mx, hx, x, c, s, y_a, y_q, y_s, all_y_a, all_y_q, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1):
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
        #soft = nn.Softmax()
        qa_id = len(hx)
        iter_count = 0
        while qa_id < 11:
            #iter_count = 0


            if iter_count == 0:
                #print("check point in generation 1 - hx size:", len(hx), hx[i-1])
                # encode
                ei, ei_len = self.a_input_encoder(None, x)
                q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
                idx = torch.from_numpy(ei_len - 1).long().cuda()
                batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
                q_prior[batch_index, idx] = 1

                ei_c, ei_len_c = self.a_caption_encoder(None, c)
                c_prior = torch.zeros(ei_c.size(0), ei_c.size(1)).cuda()
                idx_c = torch.from_numpy(ei_len_c - 1).long().cuda()
                batch_index_c = torch.arange(0, ei_len_c.shape[0]).long().cuda()
                c_prior[batch_index_c, idx_c] = 1

                num_samples = s.shape[0]
                s_for_a = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_a = self.a_emb_s(s_for_a)
                s_for_a = s_for_a.view(num_samples, -1, s_for_a.size(1), s_for_a.size(2)).transpose(2, 3)

                a = mx[0].cuda().permute(1, 2, 0)
                a = self.a_emb_a(a)
                a = a.transpose(1, 2)

                eh_temp_for_a, eh_for_a = self.a_history_encoder(None, hx)

            #ei = self.atten(utils=[s[0], s[3]], priors=[None, None])
            #ei, ei_len = self.a_input_encoder(None, x)
                ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])

                a_q = ei[0]
                a_s_for_a = [ei[1], ei[2], ei[3], ei[4]]
                a_a = ei[5]
                a_c = ei[6]
                a_h = ei[7]
                a_q = self.lin2(a_q)
        #a_h = ei[6].unsqueeze(0)

                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
        #eh_temp, eh = self.history_encoder(None, hx)
            #print("eh_temp_for_a and a_c size:", eh_temp_for_a.size(), a_c.size())
            ei, ei_len = self.a_input_encoder(None, x)
            ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])
            a_h = ei[7]
            a_c = ei[6]
            s1 = torch.cat((a_h , a_c), dim=1) # 384
            s1 = self.lin1(s1)
            pair_att_for_a = self.a_pair_atten(eh_temp_for_a, s1)


            # concatenate encodings
            # es = torch.cat((eh[-1]), dim=1)
            # es = a_h.squeeze(0)
            #es = torch.squeeze(eh, 0)
            # es = ei[6]
            #print("es size in generation:", es.size())
            #es = torch.cat((a_q, test_att_out[-1]), dim=1)
            #es = torch.cat((a_q, a_h[-1]), dim=1)
            #es = a_q
            #es_for_a = torch.cat((a_q, a_c, pair_att_for_a[-1]), dim=1)
            #a_q = dq
            es_for_a = torch.cat((a_q, a_c, pair_att_for_a[-1]), dim=1)

            #print("size check:", a_q.size(), a_h.size(), a_h.size())
            #es_for_a = torch.cat((a_q, a_c, a_h), dim=1)

            # if hasattr(self.a_response_decoder, 'context_to_state') \
            #     and self.a_response_decoder.context_to_state==True:
            #     dy = self.a_response_decoder(es_for_a, None, y_s) 
            # else:
            #     # decode
            
            #a, dy = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, y_a)
            #print("check es_for_a:", es_for_a.size()) #(1,640)
            #es_for_a = es_for_a.unsqueeze(1)
            da, _ = self.a_dec(es_for_a.unsqueeze(1))

            #################################Q BOT######################################################
            if iter_count == 0:
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)


                ################################################################################################
                ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])

                a_s_for_q = [ei_for_q[0], ei_for_q[1]]
                a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
                _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

                a_h_for_q = ei_for_q[2]
            #print("eh_temp_for_q , eh_for_q, a_h_for_q, and dy size:", eh_temp_for_q.size(), eh_for_q.size(), a_h_for_q.size(), dy.size())
# ####################################################################################################################
#             dy = dy.unsqueeze(0)
#             #print("check point 2- dy size:", dy.size())
#             pair_att_for_q = self.q_pair_atten(eh_temp_for_a, dy)
#             #print("pair_att_for_q size:",pair_att_for_q[-1].size())

#             #es_for_q = pair_att_for_q[-1]
#             #es_for_q = torch.cat((eh_for_q.squeeze(0), pair_att_for_q[-1]), dim=1)
#             es_for_q = torch.cat((a_h_for_q, pair_att_for_q[-1]), dim=1)
####################################################################################################################
            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])
            a_h_for_q = ei_for_q[2]
            es_for_q = a_h_for_q


            dq, _ = self.q_dec(es_for_q.unsqueeze(1))
            eh_temp_for_a = torch.cat((eh_temp_for_a, da, dq), dim=1)
            eh_temp_for_q = eh_temp_for_a




            # if hasattr(self.q_question_decoder, 'context_to_state') \
            #     and self.q_question_decoder.context_to_state==True:
            #     dq = self.q_question_decoder(es_for_q, None, y_s) 
            # else:
                # decode
            #q, dq = self.q_question_decoder(hidden_temporal_state_for_q, es_for_q, y_q)

            #print("check point in generation 2 - dy and dq size:", dy.size(), dq.size())

            # dq = dq.unsqueeze(0)
            # dy = dy.unsqueeze(0)
            a_q = dq.squeeze(1)
            #print("eh_temp_for_a:", eh_temp_for_a)
            #print("dq:", dq)
            #print(eh_for_q.size(),dq.size())
            # eh_temp_for_a = torch.cat((eh_for_a,dq,dy),dim=1)
            # eh_temp_for_q = eh_temp_for_a
            #print("eh_temp_for_a size:", eh_temp_for_a.size())
            #######UPDATE INPUT SEQUENCE LABEL FOR Y_A and Y_Q############

            # y_a = []
            # y_q = []
            # pos_a = a.argmax(dim=1)
            # pos_q = q.argmax(dim=1)
            # y_a = pos_a
            # y_a[0] = 2
            # y_q = pos_q
            # y_q[0] = 2
            # l_a = pos_a.size()[0] 
            # l_q = pos_q.size()[0] 
            #print(torch.split(pos_a, 10, dim=0)[0])
            # split_a = torch.split(pos_a, l_a, dim=0)
            # split_q = torch.split(pos_q, l_q, dim=0)
            # #print(len(split_a))
            # for i in range(batch_size):
            #     # eos = torch.tensor(2)
            #     temp_a = split_a[i]
            #     temp_q = split_q[i]
            #     temp_a[0] = 2
            #     temp_q[0] = 2
            #     y_a.append(temp_a)
            #     y_q.append(temp_q)

            # pair_att = self.g_pair_atten(dy, dq)
            # print("check point in generation 3 - pair_att size:", pair_att.size())
            qa_id += 1
            iter_count += 1
            # prob = random.random()
            # if prob < 0.1:
            #     #iter_count = 0
            #     iter_count += 1

        if (qa_id == 11):
            # pair_att = self.g_pair_atten(dy, dq)
            # dy = dy.transpose(0,1)
            # dq = dq.transpose(0,1)
            #soft = nn.Softmax()
            #pair_att = soft(pair_att)
            #pair_att = soft(dy)

            #print("q_h_for_q size:",a_h_for_q.size())
            es_final = torch.cat((a_h_for_q, da.squeeze(1), dq.squeeze(1)),dim=1)
            #es_final = torch.cat((a_h_for_q, pair_att[-1]),dim=1)
            #es_final = a_h_for_q
                #es_final = soft(es_final)
                #print("es es_final size:", es_final.size())
            #pair_att = pair_att.squeeze()
            #hx.append(pair_att.squeeze())
            #print("check point in generation 4 - es_final size:", )
            #es_final = torch.cat((a_h_for_q, pair_att[-1]), dim=1)
            #print("check point in generation 4 - es_final size:", )


            #print("check point in generation 4 - final input:", len(hx), a_h_for_q.size(), pair_att.size())
        #es_final = torch.cat((a_h_for_q, pair_att[-1]), dim=1)


        #print("check point in generation 4 - final input:", len(hx), es_final.size())
        # beam search
        ds = self.q_summary_decoder.initialize(hidden_temporal_state_for_q, es_final, torch.from_numpy(np.asarray([sos])).cuda())
        #print("check ds:", len(ds), ds[2].size())
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

        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
            #print("check point - maxhyps:", maxhyps[0][0])
            #print("check point - best_state:", best_state[1])
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None



    def loss_evaluate(self, mx, hx, x, c, y_a, y_q, y_s, t_a, t_q, t_s, s):

        soft = nn.Softmax()
        qa_id = len(hx)
        cul_loss = 0
        #print("history length:",len(hx))
        while qa_id < 11:
            iter_count = 0
            #cul_loss = 0

            if iter_count == 0:
        # question embed for A BOT

                #cul_loss = 0
                ei, ei_len = self.a_input_encoder(None, x)
        # print("question embedding before attention:", ei.size())
        # print("question embedding example:", ei[0,:,:])
                q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
                idx = torch.from_numpy(ei_len - 1).long().cuda()
                batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
                q_prior[batch_index, idx] = 1

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
                a = mx[0].cuda().permute(1, 2, 0)
                a = self.a_emb_a(a)
                a = a.transpose(1, 2)


        # History embed for A BOT
                eh_temp_for_a, eh_for_a = self.a_history_encoder(None, hx)

        # Multimodal attention
        #ei = self.atten(utils=[s[0], s[3]], priors=[None, None])
                ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])

        #a_q: attended question embeding
                a_q = ei[0]
        #print("question embeding after attention:", a_q.size())
        #a_s: attended embeddings for 4 visual frames
                a_s_for_a = [ei[1], ei[2], ei[3], ei[4]]
        #print("visual embedding after attention:", ei[1].size())
        #a_a: attended audio embedding
                a_a = ei[5]
        #print("audio embedding after attention:", a_a.size())
                a_c = ei[6]

                a_h = ei[7]
        #print("a_h and eh dim:", a_h.size(), eh.size())

                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
        #eh_temp, eh = self.history_encoder(None, hx)
        #print("history embedding:", eh_temp.size(), eh.size()) #is (1, 64, 128)
            #ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])

            pair_att_for_a = self.a_pair_atten(eh_temp_for_a, a_c)
        #print("check atten_test:", test_att_out, test_att_out.size())

        # concatenate encodings
        # es = ei[6]
        #print("es size:", es.size())
        # es = a_h.squeeze()
        #es = a_q
            es_for_a = torch.cat((a_q, a_c, pair_att_for_a[-1]), dim=1)
        ## caption + his_with_att
        #es = torch.cat((a_q, a_h[-1]), dim=1)

        #generate answer dy for the given question
        #print("check point 1 - es_for_a size:", es_for_a.size())
            if hasattr(self.a_response_decoder, 'context_to_state') \
                and self.a_response_decoder.context_to_state==True:
                _,  _, dy = self.a_response_decoder(es_for_a, None, y_a) 
            else:
            # decode
                _, _, dy = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, y_a)


        ##############################################################################################
        ###########Q BOT##################################################

            if iter_count == 0:

                # visual input for Q BOT
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)

                # history embed for Q BOT
                #eh_temp_for_q, eh_for_q = self.a_history_encoder(None, hx)

        # Multimodal attention
                ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])

            # Prepare the decoder
                a_s_for_q = [ei_for_q[0], ei_for_q[1]]
                a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
                _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

                a_h_for_q = ei_for_q[2]

        # Pair level attention between history and answer generated by A BOT
        # print("eh_temp_for_q and dy size:", eh_temp_for_q.size(), dy.size())

            #pair_att_for_q = self.q_pair_atten(eh_temp_for_a, dy)
            #print(eh_for_q.size(), pair_att_for_q[-1].size())

            #es_for_q = torch.cat((eh_for_q.squeeze(0), pair_att_for_q[-1]), dim=1)
        #es_for_q = pair_att_for_q[-1]
            #es_for_q = torch.cat((a_h_for_q, pair_att_for_q[-1]), dim=1)
            #pair_att_for_q = self.q_pair_atten(eh_temp_for_a, dy)
            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])
            a_h_for_q = ei_for_q[2]
            es_for_q = a_h_for_q



        # Q generate next question dq
        #print("check point 2 - es_for_q size:", es_for_q.size())
            if hasattr(self.q_question_decoder, 'context_to_state') \
                and self.q_question_decoder.context_to_state==True:
                _,  _, dq = self.q_question_decoder(es_for_q, None, y_q) 
            else:
                # decode
                _, _, dq = self.q_question_decoder(hidden_temporal_state_for_q, es_for_q, y_q)

            # dq = dq.unsqueeze(0)
            #print(eh_for_q.size(),dq.size())

            eh_temp_for_a = torch.cat((eh_for_a.transpose(0,1),dq,dy),dim=1)
            eh_temp_for_q = eh_temp_for_a
            #a_q = dy
            a_q = dq

        #     pair_att = self.g_pair_atten(dy, dq)
        # #pair_att = soft(pair_att)
        # #print("check point 3 - a_h_for_q and pair_att size:", a_h_for_q.size(), pair_att.size())

        #     es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]), dim=1)
        #es_final = soft(es_final)

        #print("check point 4 - es_final size:", es_final.size())


        # #generate summary 
        #     if hasattr(self.q_summary_decoder, 'context_to_state') \
        #         and self.q_summary_decoder.context_to_state==True:
        #         ds_q, dy_q = self.q_summary_decoder(es_final, None, y_s) 
        #     else:
        #         # decode
        #         ds_q, dy_q = self.q_summary_decoder(hidden_temporal_state_for_q, es_final, y_s)

        #     # compute loss
        #     if t_s is not None:
        #         tt = torch.cat(t_s, dim=0)
        #         loss = F.cross_entropy(dy_q, torch.tensor(tt, dtype=torch.long).cuda())
        #         #max_index = dy.max(dim=1)[1]
        #         #hit = (max_index == torch.tensor(tt, dtype=torch.long).cuda()).sum()
        #         prob = random.random()
        #         if prob < 0.1 and iter_count != 0:
        #             cul_loss += loss
        #         if iter_count == 0:
        #             cul_loss = loss
        #         else:
        #             cul_loss = cul_loss
        #         # cul_loss += loss
        #     # if target is None, it only returns states
            qa_id += 1
            iter_count += 1

        # if qa_id == 11:
        #     if t_s is not None:
        #         #print("check loss 1 and 2:", loss, cul_loss)
        #         #print("check internal loss:", len(hx), cul_loss, loss)
        #         return None, ds_q, cul_loss
        #     else:
        #         return None, ds_q



        ### prepare updated input for summary decoder
        # dy is generared answer, dq is generated question
        #print("dy and dq size in training:", dy.size(), dq.size())
        #print("current history size:", eh_temp_for_q.size(), eh_for_q.size())
########################################################################################
        if qa_id == 11:
                pair_att = self.g_pair_atten(dy, dq)
            #pair_att = soft(pair_att)
            #print("check point 3 - a_h_for_q and pair_att size:", a_h_for_q.size(), pair_att.size())

                #es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]), dim=1)
                es_final = torch.cat((a_h_for_q, pair_att[-1]), dim = 1)
            #es_final = soft(es_final)

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

    def generate_setting0(self, mx, hx, x, c, s, y_a, y_q, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1):
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
        soft = nn.Softmax()
        qa_id = len(hx)
        while qa_id < 11:
            iter_count = 0

            if iter_count == 0:
                #print("check point in generation 1 - hx size:", len(hx), hx[i-1])
                # encode
                ei, ei_len = self.a_input_encoder(None, x)
                q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
                idx = torch.from_numpy(ei_len - 1).long().cuda()
                batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
                q_prior[batch_index, idx] = 1

                ei_c, ei_len_c = self.a_caption_encoder(None, c)
                c_prior = torch.zeros(ei_c.size(0), ei_c.size(1)).cuda()
                idx_c = torch.from_numpy(ei_len_c - 1).long().cuda()
                batch_index_c = torch.arange(0, ei_len_c.shape[0]).long().cuda()
                c_prior[batch_index_c, idx_c] = 1

                num_samples = s.shape[0]
                s_for_a = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_a = self.a_emb_s(s_for_a)
                s_for_a = s_for_a.view(num_samples, -1, s_for_a.size(1), s_for_a.size(2)).transpose(2, 3)

                a = mx[0].cuda().permute(1, 2, 0)
                a = self.a_emb_a(a)
                a = a.transpose(1, 2)

                eh_temp_for_a, eh_for_a = self.a_history_encoder(None, hx)

                #ei = self.atten(utils=[s[0], s[3]], priors=[None, None])
                ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])

                a_q = ei[0]
                a_s_for_a = [ei[1], ei[2], ei[3], ei[4]]
                a_a = ei[5]
                a_c = ei[6]
            #a_h = ei[6].unsqueeze(0)

                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
            #eh_temp, eh = self.history_encoder(None, hx)
            #print("eh_temp_for_a and a_c size:", eh_temp_for_a.size(), a_c.size())
            pair_att_for_a = self.a_pair_atten(eh_temp_for_a, a_c)

            # concatenate encodings
            # es = torch.cat((eh[-1]), dim=1)
            # es = a_h.squeeze(0)
            #es = torch.squeeze(eh, 0)
            # es = ei[6]
            #print("es size in generation:", es.size())
            #es = torch.cat((a_q, test_att_out[-1]), dim=1)
            #es = torch.cat((a_q, a_h[-1]), dim=1)
            #es = a_q
            es_for_a = torch.cat((a_q, a_c, pair_att_for_a[-1]), dim=1)

            if hasattr(self.a_response_decoder, 'context_to_state') \
                and self.a_response_decoder.context_to_state==True:
                dy = self.a_response_decoder(es_for_a, None, y_a) 
            else:
                # decode
                dy = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, y_a)

            #################################Q BOT######################################################
            if iter_count == 0:
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)


                ################################################################################################
                ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])

                a_s_for_q = [ei_for_q[0], ei_for_q[1]]
                a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
                _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

                a_h_for_q = ei_for_q[2]
            #print("eh_temp_for_q , eh_for_q, a_h_for_q, and dy size:", eh_temp_for_q.size(), eh_for_q.size(), a_h_for_q.size(), dy.size())
# ####################################################################################################################
#             dy = dy.unsqueeze(0)
#             #print("check point 2- dy size:", dy.size())
#             pair_att_for_q = self.q_pair_atten(eh_temp_for_a, dy)
#             #print("pair_att_for_q size:",pair_att_for_q[-1].size())

#             #es_for_q = pair_att_for_q[-1]
#             #es_for_q = torch.cat((eh_for_q.squeeze(0), pair_att_for_q[-1]), dim=1)
#             es_for_q = torch.cat((a_h_for_q, pair_att_for_q[-1]), dim=1)
####################################################################################################################
            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp_for_a], priors=[None, None, None])
            a_h_for_q = ei_for_q[2]
            es_for_q = a_h_for_q




            if hasattr(self.q_question_decoder, 'context_to_state') \
                and self.q_question_decoder.context_to_state==True:
                dq = self.q_question_decoder(es_for_q, None, y_q) 
            else:
                # decode
                dq = self.q_question_decoder(hidden_temporal_state_for_q, es_for_q, y_q)

            #print("check point in generation 2 - dy and dq size:", dy.size(), dq.size())

            dq = dq.unsqueeze(0)
            dy = dy.unsqueeze(0)
            #a_q = dy
            a_q = dq
            #print("eh_temp_for_a:", eh_temp_for_a)
            #print("dq:", dq)
            #print(eh_for_q.size(),dq.size())
            eh_temp_for_a = torch.cat((eh_for_a,dq,dy),dim=1)
            eh_temp_for_q = eh_temp_for_a
            #print("eh_temp_for_a size:", eh_temp_for_a.size())

            # pair_att = self.g_pair_atten(dy, dq)
            # print("check point in generation 3 - pair_att size:", pair_att.size())
            qa_id += 1
            iter_count += 1

        if (qa_id == 11):
            pair_att = self.g_pair_atten(dy, dq)
            dy = dy.transpose(0,1)
            dq = dq.transpose(0,1)
            #soft = nn.Softmax()
            #pair_att = soft(pair_att)
            #pair_att = soft(dy)

            #print("q_h_for_q size:",a_h_for_q.size())
            #es_final = torch.cat((a_h_for_q, dy[-1]),dim=1)
            #es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]),dim=1)
            es_final = a_h_for_q
            es_final = torch.cat((a_h_for_q, pair_att[-1]), dim=1)
                #es_final = soft(es_final)
                #print("es es_final size:", es_final.size())
            #pair_att = pair_att.squeeze()
            #hx.append(pair_att.squeeze())
            #print("check point in generation 4 - es_final size:", )
            #es_final = torch.cat((a_h_for_q, pair_att[-1]), dim=1)
            #print("check point in generation 4 - es_final size:", )


            #print("check point in generation 4 - final input:", len(hx), a_h_for_q.size(), pair_att.size())
        #es_final = torch.cat((a_h_for_q, pair_att[-1]), dim=1)


        #print("check point in generation 4 - final input:", len(hx), es_final.size())
        # beam search
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

        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None