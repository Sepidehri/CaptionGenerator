# data/vocab.py

# include any imports the class uses
import nltk
from collections import Counter

class Vocabulary:
    """Vocabulary for text"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<UNK>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)