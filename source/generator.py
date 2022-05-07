from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from source.model import RNNModel
from source.utils import WordsPostprocesser, SOS_TOKEN_INDEX, EOS_TOKEN_INDEX


class Generator(ABC):
    """Abstract class for text generation using RNNModel"""
    
    def __init__(self,
                 index2word: Dict[int, str],
                 max_length: int = 30):
        """Args:
            index2word (Dict[int, str]): vocabulary index to token mapping
            max_length (int): maximum generated sequence length
        """
        self.index2word = index2word
        self.max_length = max_length
        self._postprocess = WordsPostprocesser(self.index2word)

    @abstractmethod
    def _next_token(self, probas):
        pass

    @abstractmethod
    def _start_token(self):
        pass

    @abstractmethod
    def name(self):
        pass

    def generate(self, rnnmodel: RNNModel, device: torch.device = None):
        start_token = self._start_token()
        current_input = torch.tensor([[start_token]])
        if device is not None:
            current_input = current_input.to(device)
        end_output = torch.tensor([[EOS_TOKEN_INDEX]])
        if device is not None:
            end_output = end_output.to(device)

        current_length = 1
        sentence_tokens = [start_token]

        state = None
        while current_input != end_output \
                and len(sentence_tokens) < self.max_length:
            with torch.no_grad():
                out_probas, state = rnnmodel.step(current_input, state)
                out_probas = np.exp(out_probas.cpu().detach().numpy()[0][0])
            
            next_token = self._next_token(out_probas)
            sentence_tokens.append(int(next_token))

            current_input = torch.tensor([[next_token]]).to(device)

        return self._postprocess(sentence_tokens)

    
class ArgmaxGenerator(Generator):
    """Generator class accessor choosing next token 
    with maximum predicted probability.
    """
    
    def __init__(self,
                 index2word: Dict[int, str],
                 max_length: int = 30,
                 start_tokens: Optional[Iterable[int]] = None):
        """Args:
            index2word (Dict[int, str]): vocabulary index to token mapping
            max_length (int): maximum generated sequence length
            start_tokens (Optional[Iterable[int]]): list of token indices
                in vocabulary used as start of sequence. If not passed all
                tokens are used as start tokens.
        """
        self.index2word = index2word
        self.max_length = max_length
        if start_tokens is None:
            self.start_tokens = list(self.index2word.keys())
        else:
            self.start_tokens = start_tokens
        self._postprocess = WordsPostprocesser(self.index2word, cut_start=False)

    def _next_token(self, probas):
        return np.argmax(probas)

    def _start_token(self):
        return np.random.choice(list(self.start_tokens), 1)[0]

    def _trim(self, sentence_words):
        return sentence_words[:-1]

    def name(self):
        return 'Argmax generator'


class DistributionGenerator(Generator):
    """Generator class accessor sampling next token
    from predicted distribution.
    """
    
    def __init__(self,
                 index2word: Dict[int, str],
                 max_length: int = 30,
                 k_max: Optional[int] = None):
        """Args:
            index2word (Dict[int, str]): vocabulary index to token mapping
            max_length (int): maximum generated sequence length
            k_max (Optional[int]): if passed, next token is sampled among
                only k_max maximum normalized predicted probabilities.
        """
        super(DistributionGenerator, self).__init__(index2word, max_length)
        if k_max is None:
            self.k_max = len(self.index2word)
        else:
            self.k_max = k_max

    def _next_token(self, probas):
        max_k_ind = np.argpartition(probas, -self.k_max)[-self.k_max:]
        zeroed_probas = np.zeros_like(probas)
        zeroed_probas[max_k_ind] = probas[max_k_ind]
        zeroed_probas /= np.sum(zeroed_probas)
        return np.random.choice(len(self.index2word), 1, p=zeroed_probas)[0]

    def _start_token(self):
        return SOS_TOKEN_INDEX

    def name(self):
        return 'Distributed generator'


class BeamSearchGenerator(ArgmaxGenerator):
    """Generator using depth-first beam search for generation.
    """
    
    def __init__(self,
                 index2word,
                 max_length=30,
                 start_tokens=None,
                 beam_width=3):
        """Args:
            index2word (Dict[int, str]): vocabulary index to token mapping
            max_length (int): maximum generated sequence length
            start_tokens (Optional[Iterable[int]]): list of token indices
                in vocabulary used as start of sequence. If not passed all
                tokens are used as start tokens
            beam_width (int): number of max predicted probabilities checked at
                every step of generation.
        """
        super(BeamSearchGenerator, self).__init__(index2word, max_length)
        self.beam_width = beam_width
        if start_tokens is None:
            self.start_tokens = list(self.index2word.keys())
        else:
            self.start_tokens = start_tokens

    def name(self):
        return 'Beam search generator'

    def _generate(self, model, device, sum_prob, current_sequence, results,
                  best_res, state):
        if sum_prob < best_res:
            return
        if current_sequence[-1] == EOS_TOKEN_INDEX \
                or len(current_sequence) == self.max_length:
            if sum_prob > best_res:
                best_res = sum_prob
            results.append((sum_prob, current_sequence))
            return
        current_input = torch.tensor([[current_sequence[-1]]]).to(device)
        with torch.no_grad():
            probas, state = model.step(current_input, state)
            probas = np.exp(probas.cpu().detach().numpy()[0][0])
        for next_token in np.argsort(-probas)[:self.beam_width]:
            next_sum_prob = sum_prob * probas[next_token]
            next_current_sequence = current_sequence + [next_token]
            self._generate(model, device, next_sum_prob, 
                           next_current_sequence, results, best_res, state)
            
    def generate(self, model, device):
        results = []
        sum_prob = 0
        current_sequence = [self._start_token()]
        self._generate(model, device, sum_prob, current_sequence, results, 0, None)
        best_result = sorted(results, key=lambda r: r[0])[-1]
        return self._postprocess(best_result[1])
