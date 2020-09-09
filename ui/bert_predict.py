#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()
    model.zero_grad()

    return model

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    return model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] +         [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_bert_sub_embedding(input_ids, ref_input_ids,
                                   token_type_ids, ref_token_type_ids,
                                   position_ids, ref_position_ids, interpretable_embedding1, interpretable_embedding2, interpretable_embedding3):
    input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
    ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(position_ids)
    ref_input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(ref_position_ids)
    
    return (input_embeddings, ref_input_embeddings), (input_embeddings_token_type, ref_input_embeddings_token_type), (input_embeddings_position_ids, ref_input_embeddings_position_ids)
    
def construct_whole_bert_embeddings(interpretable_embedding, input_ids, ref_input_ids, token_type_ids=None, ref_token_type_ids=None, position_ids=None, ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    
    return input_embeddings, ref_input_embeddings


def get_result(question, text):

    with torch.no_grad():
        input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
        attention_mask = construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(indices)

        start_scores, end_scores = predict(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        predicted_answer =  ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        predicted_answer = predicted_answer.replace(' ##','')  
    
    return predicted_answer

def valid(indexes, m):
    index_set = set(indexes)
    for i in range(len(m)):
        if index_set==m[i]:
            return False

    return True


model = load_model("trained_model_sq1.1")
tokenizer = BertTokenizer.from_pretrained("trained_model_sq1.1") # load tokenizer

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
sep_token_id = tokenizer.sep_token_id 
