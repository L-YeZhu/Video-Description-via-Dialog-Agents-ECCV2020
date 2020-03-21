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
from atten import Atten, NaiveAttention, H_Q_Attention_q, H_Q_Attention_a, H_Q_Attention_g
torch.manual_seed(1)


class MMSeq2SeqModel(nn.Module):

    def __init__(self, mm_encoder, history_encoder, a_caption_encoder, a_input_encoder, a_response_decoder, q_summary_decoder, q_question_decoder):
        """ Define model structure
            Args:
                history_encoder (~chainer.Chain): history encoder network
                input_encoder (~chainer.Chain): input encoder network
                response_decoder (~chainer.Chain): response decoder network
        """
        super(MMSeq2SeqModel, self).__init__()
        self.mm_encoder = mm_encoder
        self.history_encoder = history_encoder
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
        self.a_emb_temporal_sp = nn.LSTM(self.s_embed, self.s_embed, dropout=0.5, batch_first=True)
        self.q_emb_s = nn.Conv1d(512, self.s_embed, 1)
        #self.q_emb_a = nn.Conv1d(128, self.a_embed, 1)
        self.q_emb_temporal_sp = nn.LSTM(self.s_embed, self.s_embed, dropout=0.5, batch_first=True)
        self.q_emb_q = nn.Conv1d(128, q_embed, 1)
        self.qalstm = nn.LSTM(128,128)
        #self.plstm = nn.LSTM(256,128)

        #self.atten = NaiveAttention()
        self.a_atten = Atten(util_e=[self.s_embed, self.s_embed,self.s_embed, self.s_embed, self.a_embed, self.c_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[49, 49, 49, 49, 10, 10, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)
        self.q_atten = Atten(util_e=[self.s_embed, self.s_embed, self.h_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[49, 49, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)

        # #EXPERIMENTS
        # self.q_pair_atten = H_Q_Attention_q()
        # self.a_pair_atten = H_Q_Attention_a()
        # self.g_pair_atten = H_Q_Attention_g()


    def loss(self, mx, hx, x, c, y_a, y_q, y_s, t_a, t_q, t_s, s, all_ai, all_qi):
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

        ###################################################################
        qa_id = len(hx)
        remian_len = 10 - len(hx)
        eh_temp, eh = self.history_encoder(None, hx)
        round_n = 0
        while qa_id < 11:
            #print('round_n', round_n)
            if qa_id < 9:
                seperate_ai = []
                seperate_qi = []
                for k in range(len(all_ai)):
                    temp_ai = all_ai[k]
                    temp_qi = all_qi[k]
                    ai_count = 0
                    qi_count = 0
                    for ki in range(len(temp_qi)):
                        if temp_qi[ki] == 2 and qi_count == round_n:
                            pos_s = ki
                        if temp_qi[ki] == 2 and qi_count == (round_n+1):
                            pos_e = ki
                        if temp_qi[ki] == 2:
                            qi_count += 1
                    qi = temp_qi[pos_s: pos_e]
                    #print('qi', qi)
                    # if (len(qi)< 1):
                    #     qi = torch.tensor[[]]
                    #     print("---------------------------------------------------")
                    seperate_qi.append(qi)
                            #break
                    for ki in range(len(temp_ai)):
                        if temp_ai[ki] == 2 and ai_count == round_n:
                            pos_s = ki
                        if temp_ai[ki] == 2 and ai_count == (round_n+1):
                            pos_e = ki
                        if temp_ai[ki] == 2:
                            ai_count += 1
                    ai = temp_ai[pos_s: pos_e]
                    # if len(ai) < 1:
                    #     print('***************************************************')
                    seperate_ai.append(ai)
            else:
                seperate_ai = y_a
                seperate_qi = y_q

            # print('y_a', len(y_a[0]), len(y_a[25]))
            # print('seperate_ai', len(seperate_ai[0]), len(seperate_ai[25]))
            # print('y_q', seperate_qi[6])
            # print('y_a', seperate_ai[6])

        # ##################################################################
        # qa_id = len(hx)
        # iter_count = 0

        # # history embed for two agents
        # eh_temp, eh = self.history_encoder(None, hx)

        # while qa_id < 11:
        # ###########Q BOT##################################################

            if round_n == 0:

                # visual input for Q BOT
                num_samples = s.shape[0]
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)

            # Multimodal attention
                ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])

            # Prepare the decoder
                a_s_for_q = [ei_for_q[0], ei_for_q[1]]
                a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
                _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])
            es_for_q = ei_for_q[2]

            _, _, dq = self.q_question_decoder(hidden_temporal_state_for_q, es_for_q, seperate_qi)

            #print('dq', dq.size())

            _, (r_dq,dc) = self.qalstm(dq.transpose(0,1))
            #print('dq', r_dq.size())            

        ###################################################################################
        #################   A bot #####################

            if round_n == 0:

                # # print("question embedding before attention:", ei.size())
                # # print("question embedding example:", ei[0,:,:])
                # q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
                # idx = torch.from_numpy(ei_len - 1).long().cuda()
                # batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
                # q_prior[batch_index, idx] = 1

                # caption embed for A BOT
                ei_c, ei_len_c = self.a_caption_encoder(None, c)
                # print('ei_c:', ei_c.size())
                # print('ei_len_c:', ei_len_c.size())
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

                # Multimodal attention
                ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])

                #print("question embeding after attention:", a_q.size())
                #a_s: attended embeddings for 4 visual frames
                a_s_for_a = [ei[0], ei[1], ei[2], ei[3]]
                #print("visual embedding after attention:", ei[1].size())
                #a_a: attended audio embedding
                a_a = ei[4]
                #print("audio embedding after attention:", a_a.size())
                a_c = ei[5]

                a_h = ei[6]
                #print("a_h and eh dim:", a_h.size(), eh.size())

                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
                #eh_temp, eh = self.history_encoder(None, hx)
            #print("history embedding:", eh_temp.size(), eh.size()) #is (1, 64, 128)
            #ei = self.a_atten(utils=[ei, s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp_for_a], priors=[q_prior, None, None, None, None, None, c_prior, None])


        #     # s1 = torch.cat((a_h , a_c), dim=1)
        #     pair_att_for_a = self.a_pair_atten(eh_temp_for_a, a_c)
        # #print("check atten_test:", test_att_out, test_att_out.size())

        # # concatenate encodings
        # # es = ei[6]
        # #print("es size:", es.size())
        # # es = a_h.squeeze()
        # #es = a_q
            #print("a_c", a_c.size())
            ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])
            a_c = ei[5]
            a_h = ei[6]
            es_for_a = torch.cat((a_c, a_h, r_dq.squeeze(0)), dim=1)
        ## caption + his_with_att
        #es = torch.cat((a_q, a_h[-1]), dim=1)

        #generate answer dy for the given question
        #print("check point 1 - es_for_a size:", es_for_a.size())
            if hasattr(self.a_response_decoder, 'context_to_state') \
                and self.a_response_decoder.context_to_state==True:
                _,  _, da = self.a_response_decoder(es_for_a, None, y_a) 
            else:
            # decode
                _, _, da = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, seperate_ai)

##################################################################################################################


            qa_id += 1
            round_n += 1
            # print('history len', len(hx))
            # print('eh_temp', eh_temp.size())
            # print('dq', dq.size())
            # print('da', da.size())
            r_p = torch.cat((dq,da), dim=1).transpose(0,1)
            _, (r_p, _) = self.qalstm(r_p)
            # print('r_p', r_p.size())

            eh_temp = torch.cat((eh_temp, r_p.transpose(0,1)), dim=1)

###################################################################################################################

        if qa_id == 11:
                #pair_att = self.g_pair_atten(dy, dq)
                #pair_att = soft(pair_att)
                #print("check point 3 - a_h_for_q and pair_att size:", a_h_for_q.size(), pair_att.size())

                #es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]), dim=1)
                #es_final = torch.cat((a_h_for_q, pair_att[-1]), dim = 1)
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


    def generate(self, mx, hx, x, c, s, y_a, y_q, all_ai, all_qi, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1):
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
        remian_len = 10 - len(hx)
        eh_temp, eh = self.history_encoder(None, hx)
        round_n = 0
        # print('qa_id', qa_id)
        # print('y_a and y_q', y_a, y_q)
        #print('all_ai and all_qi', all_ai, all_qi)
        while qa_id < 11:
            #print('round_n', round_n)
            if qa_id < 9:
                seperate_ai = []
                seperate_qi = []
                for k in range(len(all_ai)):
                    temp_ai = all_ai[k]
                    temp_qi = all_qi[k]
                    ai_count = 0
                    qi_count = 0
                    for ki in range(len(temp_qi)):
                        if temp_qi[ki] == 2 and qi_count == round_n:
                            pos_s = ki
                        if temp_qi[ki] == 2 and qi_count == (round_n+1):
                            pos_e = ki
                        if temp_qi[ki] == 2:
                            qi_count += 1
                    qi = temp_qi[pos_s: pos_e]
                    #print('qi', qi)
                    # if (len(qi)< 1):
                    #     qi = torch.tensor[[]]
                    #     print("---------------------------------------------------")
                    seperate_qi.append(qi)
                            #break
                    for ki in range(len(temp_ai)):
                        if temp_ai[ki] == 2 and ai_count == round_n:
                            pos_s = ki
                        if temp_ai[ki] == 2 and ai_count == (round_n+1):
                            pos_e = ki
                        if temp_ai[ki] == 2:
                            ai_count += 1
                    ai = temp_ai[pos_s: pos_e]
                    # if len(ai) < 1:
                    #     print('***************************************************')
                    seperate_ai.append(ai)
            else:
                seperate_ai = y_a
                seperate_qi = y_q

            # print('y_a', len(y_a[0]), len(y_a[25]))
            # print('seperate_ai', len(seperate_ai[0]), len(seperate_ai[25]))
            # print('y_q', seperate_qi)
            # print('y_a', seperate_ai)

        # ##################################################################
        # qa_id = len(hx)
        # iter_count = 0

        # # history embed for two agents
        # eh_temp, eh = self.history_encoder(None, hx)

        # while qa_id < 11:
        # ###########Q BOT##################################################

            if round_n == 0:

                # visual input for Q BOT
                num_samples = s.shape[0]
                s_for_q = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
                s_for_q = self.q_emb_s(s_for_q)
                s_for_q = s_for_q.view(num_samples, -1, s_for_q.size(1), s_for_q.size(2)).transpose(2, 3)

            # Multimodal attention
                ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])

            # Prepare the decoder
                a_s_for_q = [ei_for_q[0], ei_for_q[1]]
                a_a_s_for_q = torch.cat([u.unsqueeze(1) for u in a_s_for_q], dim=1)
                _, hidden_temporal_state_for_q = self.q_emb_temporal_sp(a_a_s_for_q)

            ei_for_q = self.q_atten(utils=[s_for_q[0], s_for_q[3], eh_temp], priors=[None, None, None])
            es_for_q = ei_for_q[2]

            #print('seperate_qi',seperate_qi)

            _, in_q, dq = self.q_question_decoder(hidden_temporal_state_for_q, es_for_q, seperate_qi)

            in_q_test = torch.argmax(in_q, dim=1)

            #print('in_q', in_q.size())
            print("in_q_test", in_q_test)

            _, (r_dq,dc) = self.qalstm(dq.transpose(0,1))
            #print('dq', r_dq.size())            

        ###################################################################################
        #################   A bot #####################

            if round_n == 0:

                # # print("question embedding before attention:", ei.size())
                # # print("question embedding example:", ei[0,:,:])
                # q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
                # idx = torch.from_numpy(ei_len - 1).long().cuda()
                # batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
                # q_prior[batch_index, idx] = 1

                # caption embed for A BOT
                ei_c, ei_len_c = self.a_caption_encoder(None, c)
                # print('ei_c:', ei_c.size())
                # print('ei_len_c:', ei_len_c.size())
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

                # Multimodal attention
                ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])

                #print("question embeding after attention:", a_q.size())
                #a_s: attended embeddings for 4 visual frames
                a_s_for_a = [ei[0], ei[1], ei[2], ei[3]]
                #print("visual embedding after attention:", ei[1].size())
                #a_a: attended audio embedding
                a_a = ei[4]
                #print("audio embedding after attention:", a_a.size())
                a_c = ei[5]

                a_h = ei[6]
                #print("a_h and eh dim:", a_h.size(), eh.size())

                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
                #eh_temp, eh = self.history_encoder(None, hx)

            ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])
            a_c = ei[5]
            a_h = ei[6]
            es_for_a = torch.cat((a_c, a_h, r_dq.squeeze(0)), dim=1)
        ## caption + his_with_att
        #es = torch.cat((a_q, a_h[-1]), dim=1)

        #generate answer dy for the given question
        #print("check point 1 - es_for_a size:", es_for_a.size())
            #print('seperate_ai', seperate_ai)
            if hasattr(self.a_response_decoder, 'context_to_state') \
                and self.a_response_decoder.context_to_state==True:
                _,  in_a, da = self.a_response_decoder(es_for_a, None, y_a) 
            else:
            # decode
                _, in_a, da = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, seperate_ai)

            #print('in_a', in_a.size())
            in_a_test = torch.argmax(in_a, dim=1)
            print('in_a_test', in_a_test)


##################################################################################################################


            qa_id += 1
            round_n += 1
            # print('history len', len(hx))
            # print('eh_temp', eh_temp.size())
            # print('dq', dq.size())
            # print('da', da.size())
            r_p = torch.cat((dq,da), dim=1).transpose(0,1)
            _, (r_p, _) = self.qalstm(r_p)
            # print('r_p', r_p.size())

            eh_temp = torch.cat((eh_temp, r_p.transpose(0,1)), dim=1)


        if qa_id == 11:
                #pair_att = self.g_pair_atten(dy, dq)
                #pair_att = soft(pair_att)
                #print("check point 3 - a_h_for_q and pair_att size:", a_h_for_q.size(), pair_att.size())

                #es_final = torch.cat((eh_for_a.squeeze(0), pair_att[-1]), dim=1)
                #es_final = torch.cat((a_h_for_q, pair_att[-1]), dim = 1)
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

        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None



