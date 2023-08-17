import os
import pandas as pd
import spacy 
import torch
import pickle
spacy_eng = spacy.load('en_core_web_sm')
class Vocab_Builder:
    def __init__ (self,freq_threshold):
        self.itos = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"} 
        self.stoi = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}  
        self.freq_threshold = freq_threshold
    def __len__(self):
        return len(self.itos)
    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    def build_vocabulary(self, sentence_list):
        frequencies = {}  
        idx = 4 
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1 
                if(frequencies[word] == self.freq_threshold):
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
        for token in tokenized_text ]    
    def denumericalize(self, tensors):
        text = [self.itos[token] if token in self.itos else self.itos[3]]
        return text
