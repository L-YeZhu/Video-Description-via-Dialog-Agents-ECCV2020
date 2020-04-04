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


def get_vocabulary(dataset_file, cutoff=1, include_caption=True):
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
            for word in dialog['summary'].split():
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

def find_len(in_list):
    max_len = 0
    for i in range(len(in_list)):
        if len(in_list[i]) > max_len:
            max_len = len(in_list[i])
    return max_len




# Load text data
def load(fea_types, fea_path, dataset_file, set_id, vocabfile='', vocab={}, 
        include_caption=False, dictmap=None):
    dialog_data = json.load(open(dataset_file, 'r'))
    if set_id == 1:
        dialog_opt = json.load(open('/home/Students/y_z34/agents/pre/train_opt.json', 'r'))
    if set_id == 2:
        dialog_opt = json.load(open('/home/Students/y_z34/agents/pre/val_opt.json', 'r'))
    if set_id == 3:
        dialog_opt = json.load(open('/home/Students/y_z34/agents/pre/test_opt.json', 'r'))
    if vocabfile != '':
        vocab_from_file = json.load(open(vocabfile,'r'))
        for w in vocab_from_file:
            if w not in vocab:
                vocab[w] = len(vocab)
    unk = vocab['<unk>']
    eos = vocab['<eos>']
    dialog_list = []
    option_list = []
    vid_set = set()
    qa_id = 0
    for dialog in dialog_data['dialogs']:
        #summary = [words2ids(dialog['summary'], vocab)]
        #print("summary:", summary)
        caption_i = [np.array([eos], dtype=np.int32)]
        # get questions and answers representations
        questions = [words2ids(d['question'], vocab) for d in dialog['dialog']]
        answers = [words2ids(d['answer'], vocab) for d in dialog['dialog']]
        summaries = [words2ids(dialog['summary'], vocab) for d in dialog['dialog']]
        captions = [words2ids(dialog['caption'], vocab) for d in dialog['dialog']]
        # questions_opt = [words2ids()]
        # answers_opt = [words2ids()]
        qa_pair = [np.concatenate((q,a,[eos])).astype(np.int32) for q,a in zip(questions, answers)]

        vid = dictmap[dialog['image_id']] if dictmap is not None else dialog['image_id']
        vid_set.add(vid)
        for n in six.moves.range(len(questions)):
            history = copy.copy(caption_i)
            #all_question_in =  
            for m in six.moves.range(n):
                history.append(qa_pair[m])

            question = np.concatenate((questions[n], [eos])).astype(np.int32)
            answer = np.concatenate((answers[n],[eos])).astype(np.int32)
            #answer_in = np.concatenate(([eos], answers[n])).astype(np.int32)
            #answer_out = np.concatenate((answers[n], [eos])).astype(np.int32)
            #question_in = np.concatenate(([eos], questions[n])).astype(np.int32)
            #question_out = np.concatenate((questions[n], [eos])).astype(np.int32)
            summary_in = np.concatenate(([eos], summaries[n])).astype(np.int32)
            summary_out = np.concatenate((summaries[n], [eos])).astype(np.int32)
            caption = np.concatenate((captions[n], [eos])).astype(np.int32)
            dialog_list.append((vid, qa_id, history, question, answer, summary_in, summary_out, caption))
            #print("check history", history)
            qa_id += 1


    #print("check1:", len(dialog_list)) # 76590
    #print("chcek2:", dialog_list[0]) # array with type int32
    opt_q = []
    opt_a = []
    opt_list = []
    for dialog in dialog_opt['questions']:
        opt = [words2ids(o['questions'], vocab) for o in dialog['opt_questions']]
        opt_q.append(opt)
    #print("chcek3:", len(opt_q))
    for dialog in dialog_opt['answers']:
        opt = [words2ids(o['answers'], vocab) for o in dialog['opt_answers']]
        opt_a.append(opt)
    #print("chcek4:", len(opt_a), len(opt_a[0]))
    #     for n in six.moves.range()
    for i in range(len(opt_q)):
        # n-q and n_a are the longest questions 
        n_q = find_len(opt_q[i])
        n_a = find_len(opt_a[i])
        #print("check5", n_q, n_a)
        # 2 should be added as the eos
        opt_questions = np.zeros((99,n_q+1)).astype(np.int32)
        opt_answers = np.zeros((99,n_a+1)).astype(np.int32)
        for j in range(len(opt_q[i])):
            temp_q = np.array(opt_q[i][j])
            temp_q = np.concatenate((temp_q,[eos])).astype(np.int32)
            #print("check6:", temp_q)
            temp_a = np.array(opt_a[i][j])
            temp_a = np.concatenate((temp_a,[eos])).astype(np.int32)
            #print("check7", temp_a)
            opt_questions[j,0:len(temp_q)] = temp_q
            opt_answers[j, 0:len(temp_a)] = temp_a

        option_list.append((opt_questions, opt_answers))

    #print("check 5:", len(option_list),option_list[0])    
    total_list = []
    for i in range(len(dialog_list)):
        temp = dialog_list[i] + option_list[i]
        total_list.append(temp)
    #print("check 6:",len(dialog_list[0]), len(total_list[0]),total_list[0])
    if (set_id == 3):
        data = {'dialogs': total_list, 'vocab': vocab, 'features': [], 'original': dialog_data}

    else:
        data = {'dialogs': total_list, 'vocab': vocab, 'features': []}


    for ftype in fea_types:
        basepath = fea_path.replace('<FeaType>', ftype)
        features = {}
        for vid in vid_set:
            filepath = basepath.replace('<ImageID>', vid)
            shape = get_npy_shape(filepath)
            features[vid] = (filepath, shape)
            #print (shape)
        data['features'].append(features)

        
    return data 



def make_batch_indices(data, batchsize=100, max_length=20):
    # Setup mini-batches 100
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
        q_len = len(dialog[3]) # question length
        a_len = len(dialog[4]) # answer length
        summary_len = len(dialog[5]) # summary length
        caption_len = len(dialog[7]) # caption length
        opt_q_len = dialog[8].shape[1]
        opt_a_len = dialog[9].shape[1]
        #print("check opt_len:", opt_q_len)
        idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len, summary_len, caption_len, opt_q_len, opt_a_len))

    if batchsize > 1:
        idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[2][0],-s[4],-s[5],-s[6],-s[7],-s[8],-s[9]))

    n_samples = len(idxlist) ##toal num. of questions/answers
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
        opt_q_len = max(idxlist[bs:be], key=lambda s:s[8])[8]
        opt_a_len = max(idxlist[bs:be], key=lambda s:s[9])[9]
        vids = [ s[0] for s in idxlist[bs:be] ]
        qa_ids = [ s[1] for s in idxlist[bs:be] ]
        batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, summary_len, caption_len, opt_q_len, opt_a_len, be - bs))
        bs = be
            
    return batch_indices, n_samples


def make_batch(data, index, eos=1):
    x_len, h_len, q_len, a_len, summary_len, caption_len, opt_q_len, opt_a_len, n_seqs = index[2:]
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
    a_batch = []
    # a_batch_in = []
    # a_batch_out = []
    # q_batch_in = []
    # q_batch_out = []
    summary_batch_in = []
    summary_batch_out = []
    c_batch = []
    q_opt_batch = []
    a_opt_batch = []
    dialogs = data['dialogs']
    for i in six.moves.range(n_seqs):
        qa_id = index[1][i]
        history, question, answer, summary_in, summary_out, caption, q_opt, a_opt = dialogs[qa_id][2:]
        for j in six.moves.range(h_len):
            if j < len(history):
                h_batch[j].append(history[j])
            else:
                h_batch[j].append(empty_sentence)
        q_batch.append(question)
        a_batch.append(answer)
        # a_batch_in.append(answer_in)
        # a_batch_out.append(answer_out)
        # q_batch_in.append(question_in)
        # q_batch_out.append(question_out)
        summary_batch_in.append(summary_in)
        summary_batch_out.append(summary_out)
        c_batch.append(caption)
        q_opt_batch.append(q_opt)
        a_opt_batch.append(a_opt)

    #print("check h in batch", len(h_batch), h_batch)

    return s_batch, x_batch, h_batch, q_batch, a_batch, summary_batch_in, summary_batch_out, c_batch, q_opt_batch, a_opt_batch



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

