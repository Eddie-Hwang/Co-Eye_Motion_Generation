import re
import unicodedata
# import bcolz
import numpy as np
# import spacy

from tqdm import tqdm


SOS_TOKEN = 0
EOS_TOKEN = 1


class Lang:
    
    def __init__(self, name, lang_model):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'UNK', 2: 'SOS', 3: 'EOS',}
        self.n_words = 2 # include SOS and EOS
        # self.lang_model = spacy.load(lang_model)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def build_emb_table(self, pretrained_path):
        vects, words, word2idx = load_pretrained_emb(pretrained_path)
        w_dim = vects.shape[1]
        # init embedding table
        if len(self.index2word) < 2:
            assert 'Vocab was not built yet.'
        vocab = self.index2word
        emb_table = np.zeros((len(vocab), w_dim))
        for key, val in vocab.items():
            try:
                emb_table[key] = vects[word2idx[val]]
            except KeyError:
                emb_table[key] = np.random.normal(scale=0.6, size=(w_dim, ))

        return emb_table


def unicode_to_ascii(string):
    return ''.join(
        c for c in unicodedata.normalize('NFD', string) 
        if unicodedata.category(c) != 'Mn'
    )
    

def normalize_string(string):
    # string = unicode_to_ascii(string.lower().strip())
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)

    return string


def load_pretrained_emb(pretrained_path):
    words = []
    idx = 0
    word2idx = {}
    vects = []
    with open(pretrained_path, 'r') as f:
        for line in tqdm(f):
            l = line.split()
            # save words in pretrained embedding file
            word = l[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            # save vectors in pretrained embedding file
            vect = np.array(l[1:]).astype(np.float)
            vects.append(vect)

    return np.array(vects), words, word2idx


