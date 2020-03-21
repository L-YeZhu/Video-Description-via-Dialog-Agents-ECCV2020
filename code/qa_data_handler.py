#!/usr/bin/env python
"""Functions for feature data handling for q_bot
"""

import copy
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
from random import randint


def get_npy_shape(filename):
    # read npy file header and return its shape
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            shape = pickle.load(f).shape
        else:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape


def get_vocabulary(dataset_file, cutoff=1, include_caption=False):
    vocab = {'<unk>':0, '<sos>':1, '<eos>':2}
    dialog_data = json.load(open(dataset_file, 'r'))
    word_freq = {}
    for dialog in dialog_data['dialogs']:
        if include_caption:
            for word in dialog['caption'].split():
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        for key in ['question', 'answer']:
            for turn in dialog['dialog']:
                for word in turn[key].split():
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
    for word, freq in word_freq.items():
        if freq > cutoff:
            vocab[word] = len(vocab) 

    # file = open('vocab.txt','w')
    # for k in range(len(vocab)):
    #     file.write(str(k))
    #     file.write(str(vocab[k]))
    # file.close()
    # print("vocab finished!")

    return vocab


def words2ids(str_in, vocab, unk=0, eos=-1):
    words = str_in.split()
    if eos >= 0:
        sentence = np.ndarray(len(words)+1, dtype=np.int32)
    else:
        sentence = np.ndarray(len(words), dtype=np.int32)
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i] = vocab[w]
        else:
            sentence[i] = unk
    if eos >= 0:
        sentence[len(words)] = eos
    return sentence



# Load text data
def load(fea_types, fea_path, dataset_file, vocabfile='', vocab={}, 
        include_caption=False, dictmap=None):
    dialog_data = json.load(open(dataset_file, 'r'))
    if vocabfile != '':
        vocab_from_file = json.load(open(vocabfile,'r'))
        for w in vocab_from_file:
            if w not in vocab:
                vocab[w] = len(vocab)
    unk = vocab['<unk>']
    eos = vocab['<eos>']
    dialog_list = []
    vid_set = set()
    qa_id = 0
    for dialog in dialog_data['dialogs']:
        if include_caption:
            caption_i = [words2ids(dialog['caption'], vocab, eos=eos)]
        else:
            caption_i = [np.array([eos], dtype=np.int32)]

        #summary = [words2ids(dialog['summary'], vocab)]
        #print("summary:", summary)

        # get questions and answers representations
        questions = [words2ids(d['question'], vocab) for d in dialog['dialog']]
        answers = [words2ids(d['answer'], vocab) for d in dialog['dialog']]
        summaries = [words2ids(dialog['summary'], vocab) for d in dialog['dialog']]
        captions = [words2ids(dialog['caption'], vocab) for d in dialog['dialog']]
        #print("summaries:", summaries)

        # get qa pairs
        qa_pair = [np.concatenate((q,a,[eos])).astype(np.int32) for q,a in zip(questions, answers)]

        vid = dictmap[dialog['image_id']] if dictmap is not None else dialog['image_id']
        vid_set.add(vid)
        # print("question length:", len(questions)) is 10, number of total questions
        # all_question_in = []
        # all_answer_in = []
        # len_q = []
        # len_a = []
        for n in six.moves.range(len(questions)):
            history = copy.copy(caption_i)
            #all_question_in =  
            for m in six.moves.range(n):
                history.append(qa_pair[m])
            #rand_n = randint(0, len(questions)-1)
            #print(rand_n)
            question = np.concatenate((questions[n], [eos])).astype(np.int32)
            answer_in = np.concatenate(([eos], answers[n])).astype(np.int32)
            answer_out = np.concatenate((answers[n], [eos])).astype(np.int32)
            question_in = np.concatenate(([eos], questions[n])).astype(np.int32)
            question_out = np.concatenate((questions[n], [eos])).astype(np.int32)
            summary_in = np.concatenate(([eos], summaries[n])).astype(np.int32)
            summary_out = np.concatenate((summaries[n], [eos])).astype(np.int32)
            caption = np.concatenate((captions[n], [eos])).astype(np.int32)
            #rand_questin_in = np.concatenate(([eos], questions[rand_n])).astype(np.int32)
            #rand_questin_out = np.concatenate((questions[rand_n], [eos])).astype(np.int32)
            # print(questions[n])
            # print(question_in)
            # all_question_in = all_question_in.append(list(question_in))
            # all_answer_in = all_answer_in.append(list(answer_in))
            # #print(all_question_in, all_answer_in, len_q, len_a)
            # len_q = len_q.append(len(question_in))
            # len_a = len_a.append(len(answer_in))
            # print("check point 1- ", n, all_answer_in)
            # print("check point 2- ", len_q, len_a)
            remian = 9 - n 
            for i in six.moves.range(remian):
                question_in_remain = np.concatenate(([eos], questions[n+i])).astype(np.int32)
                answer_in_remian = np.concatenate(([eos], answers[n+i])).astype(np.int32)
                if i == 0:
                    all_question_in = question_in_remain
                    all_answer_in = answer_in_remian
                else:
                    all_question_in = np.concatenate((all_question_in, question_in_remain)).astype(np.int32)
                    all_answer_in = np.concatenate((all_answer_in, answer_in_remian)).astype(np.int32)
            # if n == 0:
            #     all_answer_in = answer_in
            #     all_question_in = question_in
            #     # print(n, all_answer_in)
            #     # print(n, all_question_in)
            # else:
            #     # print(answer_in)
            #     # print(answer_out)
            #     all_answer_in = np.concatenate((all_answer_in, answer_in)).astype(np.int32)
            #     all_question_in = np.concatenate((all_question_in, question_in)).astype(np.int32)
                # print(n, all_answer_in)
                # print(n, all_question_in)
            dialog_list.append((vid, qa_id, history, question, answer_in, answer_out, question_in, question_out, summary_in, summary_out, caption, all_answer_in, all_question_in))
            qa_id += 1
        #dialog_list.append((all_answer_in, all_question_in))

    data = {'dialogs': dialog_list, 'vocab': vocab, 'features': [], 
            'original': dialog_data}
    for ftype in fea_types:
        basepath = fea_path.replace('<FeaType>', ftype)
        features = {}
        for vid in vid_set:
            filepath = basepath.replace('<ImageID>', vid)
            shape = get_npy_shape(filepath)
            features[vid] = (filepath, shape)
            #print (shape)
        data['features'].append(features)


    #print("history example:", history)
        
    return data 



def make_batch_indices(data, batchsize=100, max_length=20):
    # Setup mini-batches
    idxlist = []
    for n, dialog in enumerate(data['dialogs']):
        vid = dialog[0]  # video ID
        x_len = []
        for feat in data['features']:
            value = feat[vid]
            size = value[1] if isinstance(value, tuple) else len(value)
            if len(size) == 2:
                x_len.append(size[0])

        qa_id = dialog[1]  # QA-pair id, 1 - total_number_of_question
        h_len = len(dialog[2]) # history length, is the number of l-1 qa pairs
        # print("qa_id and h_len:", qa_id, h_len)
        q_len = len(dialog[3]) # question length
        a_len = len(dialog[4]) # answer length
        summary_len = len(dialog[8]) # summary length
        caption_len = len(dialog[10]) # caption length
        # rand_q_len = len(dialog[11])
        all_a_len = len(dialog[11])
        all_q_len = len(dialog[12])
        idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len, summary_len, caption_len, all_a_len, all_q_len))

    if batchsize > 1:
        idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[2][0],-s[4],-s[5],-s[6],-s[7], -s[8], -s[9]))
        #idxlist = sorted(idxlist, key=lambda s:(-s[7],-s[2][0],-s[3],-s[5],-s[6],-s[3]))

    n_samples = len(idxlist)
    batch_indices = []
    bs = 0
    while bs < n_samples:
        in_len = idxlist[bs][3]
        bsize = batchsize / (in_len / max_length + 1)
        be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
        x_len = [ max(idxlist[bs:be], key=lambda s:s[2][j])[2][j]
                for j in six.moves.range(len(x_len))]
        h_len = max(idxlist[bs:be], key=lambda s:s[3])[3]
        q_len = max(idxlist[bs:be], key=lambda s:s[4])[4]
        a_len = max(idxlist[bs:be], key=lambda s:s[5])[5]
        summary_len = max(idxlist[bs:be], key=lambda s:s[6])[6]
        caption_len = max(idxlist[bs:be], key=lambda s:s[7])[7]
        all_a_len = max(idxlist[bs:be], key=lambda s:s[8])[8]
        all_q_len = max(idxlist[bs:be], key=lambda s:s[9])[9]
        vids = [ s[0] for s in idxlist[bs:be] ]
        qa_ids = [ s[1] for s in idxlist[bs:be] ]
        batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, summary_len, caption_len, all_a_len, all_q_len, be - bs))
        bs = be
            
    return batch_indices, n_samples


def make_batch_a(data, index, eos=1):
    x_len, h_len, q_len, a_len, summary_len, caption_len, all_a_len, all_q_len, n_seqs = index[2:]
    feature_info = data['features']
    for j in six.moves.range(n_seqs):
        fea = []
        vid = index[0][j]

        #load all features
        #s_fea dim: 4 * 49 * 512
        for fi in feature_info:
            if len(fi[vid][1]) > 2:
                s_fea = np.load(fi[vid][0])
                #print('s_fea in data loading:', s_fea.shape)
            else:
                fea.append(np.load(fi[vid][0]))


        if j == 0:
            x_batch = [np.zeros((x_len[i], n_seqs, fea[i].shape[-1]),
                       dtype=np.float32) for i in six.moves.range(len(x_len))]


            s_batch = np.zeros((s_fea.shape[0], n_seqs) + s_fea.shape[1:])


        for i in six.moves.range(len(x_len)):
            x_batch[i][:len(fea[i]), j] = fea[i]
        s_batch[:s_fea.shape[0], j] = s_fea

    empty_sentence = np.array([eos], dtype=np.int32)
    h_batch = [ [] for _ in six.moves.range(h_len) ]
    q_batch = []
    a_batch_in = []
    a_batch_out = []
    q_batch_in = []
    q_batch_out = []
    summary_batch_in = []
    summary_batch_out = []
    c_batch = []
    # rand_q_batch_in = []
    # rand_q_batch_out = []
    dialogs = data['dialogs']
    for i in six.moves.range(n_seqs):
        qa_id = index[1][i]
        history, question, answer_in, answer_out, question_in, question_out, summary_in, summary_out, caption, all_answer_in, all_question_in = dialogs[qa_id][2:]
        for j in six.moves.range(h_len):
            if j < len(history):
                h_batch[j].append(history[j])
            else:
                h_batch[j].append(empty_sentence)
        q_batch.append(question)
        a_batch_in.append(answer_in)
        a_batch_out.append(answer_out)
        q_batch_in.append(question_in)
        q_batch_out.append(question_out)
        summary_batch_in.append(summary_in)
        summary_batch_out.append(summary_out)
        c_batch.append(caption)

    #print(summary_batch_in)

    return x_batch, h_batch, q_batch, a_batch_in, a_batch_out, s_batch, summary_batch_in, summary_batch_out, c_batch


def make_batch_q(data, index, eos=1):
    x_len, h_len, q_len, a_len, summary_len, caption_len, all_a_len, all_q_len, n_seqs = index[2:]
    feature_info = data['features']
    for j in six.moves.range(n_seqs):
        fea = []
        vid = index[0][j]

        #load all features
        #s_fea dim: 4 * 49 * 512
        for fi in feature_info:
            if len(fi[vid][1]) > 2:
                s_fea = np.load(fi[vid][0])
                #print('s_fea in data loading:', s_fea.shape)
            else:
                fea.append(np.load(fi[vid][0]))


        if j == 0:
            x_batch = [np.zeros((x_len[i], n_seqs, fea[i].shape[-1]),
                       dtype=np.float32) for i in six.moves.range(len(x_len))]


            s_batch = np.zeros((s_fea.shape[0], n_seqs) + s_fea.shape[1:])


        for i in six.moves.range(len(x_len)):
            x_batch[i][:len(fea[i]), j] = fea[i]
        s_batch[:s_fea.shape[0], j] = s_fea

    empty_sentence = np.array([eos], dtype=np.int32)
    h_batch = [ [] for _ in six.moves.range(h_len) ]
    q_batch = []
    a_batch_in = []
    a_batch_out = []
    q_batch_in = []
    q_batch_out = []
    summary_batch_in = []
    summary_batch_out = []
    c_batch = []
    all_a_batch_in = []
    all_q_batch_in = []
    # rand_q_batch_in = []
    # rand_q_batch_out = []
    dialogs = data['dialogs']
    for i in six.moves.range(n_seqs):
        qa_id = index[1][i]
        history, question, answer_in, answer_out, question_in, question_out, summary_in, summary_out, caption, all_answer_in, all_question_in = dialogs[qa_id][2:]
        for j in six.moves.range(h_len):
            if j < len(history):
                h_batch[j].append(history[j])
            else:
                h_batch[j].append(empty_sentence)
        q_batch.append(question)
        a_batch_in.append(answer_in)
        a_batch_out.append(answer_out)
        q_batch_in.append(question_in)
        q_batch_out.append(question_out)
        summary_batch_in.append(summary_in)
        summary_batch_out.append(summary_out)
        c_batch.append(caption)
        all_a_batch_in.append(all_answer_in)
        all_q_batch_in.append(all_question_in)
        # rand_q_batch_in.append(rand_question_in)
        # rand_q_batch_out.append(rand_questin_out)

    #print(summary_batch_in)

    return q_batch_in, q_batch_out, all_a_batch_in, all_q_batch_in


def feature_shape(data):
    dims = []
    for features in data["features"]:
        sample_feature = features.values()[0]
        if isinstance(sample_feature, tuple):
            sample_fea = np.load(sample_feature[0])
            if len(sample_fea.shape) > 2:
                print "Detected spatial features, ", sample_fea.shape
                spatial_dims = sample_fea.shape[1:]
            else:
                dims.append(sample_fea.shape[-1])
        else:
            dims.append(sample_feature.shape[-1])
    return dims, spatial_dims


# def make_batch_q(data, index, eos=1):
#     x_len, h_len, q_len, a_len, summary_len, n_seqs = index[2:]
#     feature_info = data['features']
#     for j in six.moves.range(n_seqs):
#         fea = []
#         vid = index[0][j]

#         #load all features
#         #s_fea dim: 4 * 49 * 512
#         for fi in feature_info:
#             if len(fi[vid][1]) > 2:
#                 s_fea = np.load(fi[vid][0])
#                 #print('s_fea in data loading:', s_fea.shape)
#             else:
#                 fea.append(np.load(fi[vid][0]))


#         if j == 0:
#             x_batch = [np.zeros((x_len[i], n_seqs, fea[i].shape[-1]),
#                        dtype=np.float32) for i in six.moves.range(len(x_len))]


#             s_batch = np.zeros((s_fea.shape[0], n_seqs) + s_fea.shape[1:])


#         for i in six.moves.range(len(x_len)):
#             x_batch[i][:len(fea[i]), j] = fea[i]
#         s_batch[:s_fea.shape[0], j] = s_fea

#     empty_sentence = np.array([eos], dtype=np.int32)
#     h_batch = [ [] for _ in six.moves.range(h_len) ]
#     q_batch = []
#     a_batch_in = []
#     a_batch_out = []
#     summary_batch_in = []
#     summary_batch_out = []
#     q_batch_in = []
#     q_batch_out = []
#     dialogs = data['dialogs']
#     for i in six.moves.range(n_seqs):
#         qa_id = index[1][i]
#         history, question, answer_in, answer_out, summary_in, summary_out, question_in, question_out = dialogs[qa_id][2:]
#         for j in six.moves.range(h_len):
#             if j < len(history):
#                 h_batch[j].append(history[j])
#             else:
#                 h_batch[j].append(empty_sentence)
#         q_batch.append(question)
#         a_batch_in.append(answer_in)
#         a_batch_out.append(answer_out)
#         summary_batch_in.append(summary_in)
#         summary_batch_out.append(summary_out)
#         q_batch_in.append(question_in)
#         q_batch_out.append(question_out)

#     #print(summary_batch_in)

#     return x_batch, h_batch, q_batch, a_batch_in, a_batch_out, s_batch, summary_batch_in, summary_batch_out, q_batch_in, q_batch_out
