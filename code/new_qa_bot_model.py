# -*- coding: utf-8 -*-
"""Video-Description-via-Dialog-Agents
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
        remain_len = 10 - len(hx)
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

            _, (r_dq,dc) = self.qalstm(dq.transpose(0,1))      

        #################   A bot #####################

            if round_n == 0:

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
                a_s_for_a = [ei[0], ei[1], ei[2], ei[3]]
                a_a = ei[4]
                a_c = ei[5]
                a_h = ei[6]
                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)

            ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])
            a_c = ei[5]
            a_h = ei[6]
            es_for_a = torch.cat((a_c, a_h, r_dq.squeeze(0)), dim=1)

        #generate answer for the given question
            if hasattr(self.a_response_decoder, 'context_to_state') \
                and self.a_response_decoder.context_to_state==True:
                _,  _, da = self.a_response_decoder(es_for_a, None, y_a) 
            else:
            # decode
                _, _, da = self.a_response_decoder(hidden_temporal_state_for_a, es_for_a, seperate_ai)

##################################################################################################################


            qa_id += 1
            round_n += 1
            r_p = torch.cat((dq,da), dim=1).transpose(0,1)
            _, (r_p, _) = self.qalstm(r_p)

            eh_temp = torch.cat((eh_temp, r_p.transpose(0,1)), dim=1)

###################################################################################################################
        if qa_id == 11:
                es_final = es_for_q

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
        remain_len = 10 - len(hx)
        eh_temp, eh = self.history_encoder(None, hx)
        round_n = 0
	lin_layer = nn.Linear(256,128).cuda()

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

                    seperate_ai.append(ai)
            else:
                seperate_ai = y_a
                seperate_qi = y_q

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

	    ##################################################
	    inq_ds = self.q_question_decoder.initialize(hidden_temporal_state_for_q, es_for_q, torch.from_numpy(np.asarray([sos])).cuda())
	    inq_hyplist = [([], 0., inq_ds)]
	    inq_best_state = None
	    inq_comp_hyplist = []
	    for l in six.moves.range(maxlen):
		new_hyplist = []
		argmin = 0
		for out, lp, st in inq_hyplist:
		    logp = self.q_question_decoder.predict(st)
		    lp_vec = logp.cpu().data.numpy() + lp
		    lp_vec = np.squeeze(lp_vec)
		    if l >= minlen:
			new_lp = lp_vec[eos] + penalty * (len(out) + 1)
			new_st = self.q_question_decoder.update(st, torch.from_numpy(np.asarray([eos])).cuda())
			inq_comp_hyplist.append((out, new_lp))
			if inq_best_state is None or inq_best_state[0] < new_lp:
			    inq_best_state = (new_lp, new_st)

		    for o in np.argsort(lp_vec)[::-1]:
			if o == unk or o == eos:
			    continue
			new_lp = lp_vec[o]
			if len(new_hyplist) == 1:
			    if new_hyplist[argmin][1] < new_lp:
				new_st = self.q_question_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
				new_hyplist[argmin] = (out + [o], new_lp, new_st)
				argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
			    else:
				break
			else:
			    new_st = self.q_question_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
			    new_hyplist.append((out + [o], new_lp, new_st))
			    if len(new_hyplist) == 1:
				argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
		inq_hyplist = new_hyplist
	    if len(inq_comp_hyplist) > 0:
		inq_maxhyps = sorted(inq_comp_hyplist, key=lambda h: -h[1])[:nbest]
		r_dq = inq_best_state[1][1]
		r_dq = r_dq.cuda()
		r_dq = lin_layer(r_dq)
		
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
                s_for_a = self.a_emb_s(s_for_a)
                s_for_a = s_for_a.view(num_samples, -1, s_for_a.size(1), s_for_a.size(2)).transpose(2, 3)

                a = mx[0].cuda().permute(1, 2, 0)
                a = self.a_emb_a(a)
                a = a.transpose(1, 2)

                # Multimodal attention
                ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])

                a_s_for_a = [ei[0], ei[1], ei[2], ei[3]]
                a_a = ei[4]
                a_c = ei[5]
                a_h = ei[6]
                a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s_for_a], dim=1)
                _, hidden_temporal_state_for_a = self.a_emb_temporal_sp(a_a_s)
                #eh_temp, eh = self.history_encoder(None, hx)

            ei = self.a_atten(utils=[s_for_a[0], s_for_a[1], s_for_a[2], s_for_a[3], a, ei_c, eh_temp], priors=[None, None, None, None, None, c_prior, None])
            a_c = ei[5]
            a_h = ei[6]
            es_for_a = torch.cat((a_c, a_h, r_dq.squeeze(0)), dim=1)

            #generate answer for the given question
	    ina_ds = self.a_response_decoder.initialize(hidden_temporal_state_for_a, es_for_a, torch.from_numpy(np.asarray([sos])).cuda())
	    ina_hyplist = [([], 0., ina_ds)]
	    ina_best_state = None
	    ina_comp_hyplist = []
	    for l in six.moves.range(maxlen):
		new_hyplist = []
		argmin = 0
		for out, lp, st in ina_hyplist:
		    logp = self.a_response_decoder.predict(st)
		    lp_vec = logp.cpu().data.numpy() + lp
		    lp_vec = np.squeeze(lp_vec)
		    if l >= minlen:
			new_lp = lp_vec[eos] + penalty * (len(out) + 1) 
			new_st = self.a_response_decoder.update(st, torch.from_numpy(np.asarray([eos])).cuda())
			ina_comp_hyplist.append((out, new_lp))
			if ina_best_state is None or ina_best_state[0] < new_lp:
			    ina_best_state = (new_lp, new_st)

		    for o in np.argsort(lp_vec)[::-1]:
			if o == unk or o == eos:
			    continue
			new_lp = lp_vec[o]
			if len(new_hyplist) == 1:
			    if new_hyplist[argmin][1] < new_lp:
				new_st = self.a_response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
				new_hyplist[argmin] = (out + [o], new_lp, new_st)
				argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
			    else:
				break
			else:
			    new_st = self.a_response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
			    new_hyplist.append((out + [o], new_lp, new_st))
			    if len(new_hyplist) == 1:
				argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

		ina_hyplist = new_hyplist
	    if len(ina_comp_hyplist) > 0:
		ina_maxhyps = sorted(ina_comp_hyplist, key=lambda h: -h[1])[:nbest]
		r_da = ina_best_state[1][1].cuda()
		r_da = lin_layer(r_da)


##################################################################################################################


            qa_id += 1
            round_n += 1
	    r_p = torch.cat((r_dq,r_da),dim=2)
	    r_p = lin_layer(r_p)
            eh_temp = torch.cat((eh_temp, r_p.transpose(0,1)), dim=1)


        if qa_id == 11:
                es_final = es_for_q

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
	    #print("best_state:", best_state[1][1].size())
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None



