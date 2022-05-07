from typing import Dict, Iterable


SOS_TOKEN = "SOS"
EOS_TOKEN = "EOS"
MASK_TOKEN = "MASK"
PAD_TOKEN = "PAD"

SOS_TOKEN_INDEX = 0
EOS_TOKEN_INDEX = 1
MASK_TOKEN_INDEX = 2
PAD_TOKEN_INDEX = 3


def spacy_lemmatizer(word):
    return word.lemma_


class Lang:
    """Class storing vocabulary
    """
    
    def __init__(self, tokenizer, lemmatizer=spacy_lemmatizer):
        self.tokenizer = tokenizer
        self.word2index: Dict[str, int] = {}
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {SOS_TOKEN_INDEX: SOS_TOKEN, 
                                           EOS_TOKEN_INDEX: EOS_TOKEN, 
                                           MASK_TOKEN_INDEX: MASK_TOKEN, 
                                           PAD_TOKEN_INDEX: PAD_TOKEN}
        self.n_words = 4
        self.lemmatizer = lemmatizer

    def addDocument(self, document: Iterable[str]):
        for sentence in document:
            self.addSentence(str(sentence))

    def addSentence(self, sentence):
        doc = self.tokenizer(sentence)
        for word in doc:
            self.addWord(self.lemmatizer(word))

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def getStat(self):
        print('Vocab size', self.n_words)
        print("Most frequent", 
              sorted([(v, k) for k, v in self.word2count.items()], 
                     key=lambda v: -self.word2count[v[1]])[:10])
        print("Less frequent",
              sorted([(v, k) for k, v in self.word2count.items()], 
                     key=lambda v: self.word2count[v[1]])[:10])


class WordsPostprocesser:
    """Class for processing vocabulary token indices into string sentence."""
    
    def __init__(self, index2word: Dict[int, str], cut_start: bool = True):
        """Args:
            index2word (Dict[int, str]): vocabulary index to token mapping
            cut_start (bool): whether to remove the first index of sequence
        """
        self.index2word = index2word
        self.cut_start = cut_start

    def _trim(self, sentence_tokens):
        if self.cut_start:
            return sentence_tokens[1:-1]
        return sentence_tokens[:-1]

    def _postprocess(self, sentence_tokens):
        sentence_tokens = self._trim(sentence_tokens)
        sentence_words = [self.index2word[index] 
                          for index in sentence_tokens]
        sentence = ' '.join(sentence_words)
        sentence = sentence.replace(MASK_TOKEN + ' ', '')
        sentence = sentence.replace(SOS_TOKEN + ' ', '')
        sentence = sentence.replace(EOS_TOKEN + ' ', '')
        sentence = sentence.replace(PAD_TOKEN + ' ', '')
        sentence = sentence.replace('@@ ', '')
        sentence = sentence.replace('@@', '')
        return sentence

    def __call__(self, sentence_tokens):
        return self._postprocess(sentence_tokens)
