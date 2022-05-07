from typing import Callable, Dict, List, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from source.utils import SOS_TOKEN_INDEX, EOS_TOKEN_INDEX, PAD_TOKEN_INDEX


class SequenceNLLLoss:
    """Torch NLLLoss class wrapper computing negative log likelihood loss
    between target[1:] and model prediction on target[:-1]. Used to train model
    to predict probabilities of next token under condition of previous tokens.
    """
    
    def __init__(self, **kwargs):
        self.l = nn.NLLLoss(**kwargs)

    def __call__(self, model, batched_sequence):
        target = batched_sequence[:, 1:]
        out = torch.transpose(model(batched_sequence[:, :-1]), 1, 2)
        return self.l(out, target)
    

class NewsDataset(Dataset):
    """Torch Dataset producing samples from sentence.
    
    Every sample is a string sentence tokenized, every token is lemmatized 
    and mapped to index in vocabulary. Start and end tokens added.
    """
    
    def __init__(self, 
                 sentences: List[str], 
                 tokenizer: Callable[[str], Iterable[str]], 
                 word2index: Dict[str, int],
                 lemmatizer: Callable[[str], str]):
        super(NewsDataset, self).__init__()
        self.sentences = np.array(sentences)
        self.nlp = tokenizer
        self.word2index = word2index
        self.lemmatizer = lemmatizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        doc = self.nlp(str(self.sentences[index]))
        sample_sentence = [self.word2index[self.lemmatizer(word)] for word in doc]
        sample_sentence = [SOS_TOKEN_INDEX] + sample_sentence + [EOS_TOKEN_INDEX]
        sample_sentence = torch.tensor(sample_sentence)     

        return sample_sentence
    
    
def padding_collater(sample_sentences: torch.Tensor):
    return nn.utils.rnn.pad_sequence(
        sample_sentences,
        batch_first=True,
        padding_value=PAD_TOKEN_INDEX)


def split_dataset(dataset: Dataset, train_part: float = 0.9):
    train_len = int(train_part * len(dataset))
    train_dataset, val_dataset = random_split(dataset, 
                                              [train_len, len(dataset) - train_len], 
                                              generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset
