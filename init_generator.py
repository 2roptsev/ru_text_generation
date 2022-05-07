import numpy as np
from abc import ABC, abstractmethod

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE    

from source.generator import DistributionGenerator
from source.model import RNNModel
from source.utils import Lang


class StatedGenerator:
    def __init__(self, gen, model, device):
        self.gen = gen
        self.model = model
        self.device = device
        
    def roll(self):
        return self.gen.generate(self.model, self.device)


def get_stated_generator():
    emb_dim = 256
    hid_dim = 1024
    data_path = 'data/train.bpe.ru'
    model_path = 'pretrained_models/bpe_4000_1024'
    
    with open(data_path, 'r') as f_in:
        news_sentences_bpe = f_in.readlines()
    bpe_lang = Lang(tokenizer=lambda x: x.split(), lemmatizer=lambda x: x)
    bpe_lang.addDocument(news_sentences_bpe)
    NUM_EMBEDDINGS = len(bpe_lang.index2word)
    
    gen = DistributionGenerator(bpe_lang.index2word, max_length=150, k_max=20)
    
    device = torch.device('cpu')
    
    NUM_EMBEDDINGS = len(bpe_lang.index2word)
    empty_emb_layer = nn.Embedding(num_embeddings=NUM_EMBEDDINGS,
                                   embedding_dim=emb_dim)
    model = RNNModel(empty_emb_layer,
                     hidden_dim=hid_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return StatedGenerator(gen, model, device)
