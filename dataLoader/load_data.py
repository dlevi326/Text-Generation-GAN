
import dataset
import os
from nltk import word_tokenize
import numpy as np

import sys
sys.path.insert(0, '../gloveLoader/')
from loadGlove import loadGloveModel, getKeyedVect

def load_data(dir,num=10):
    # Receives sentences in list format 

    #file = open(filename, 'r') 
    #print(file.readline())
    sents = dataset.loopFiles(dir,num)
    return sents

def turn_sents_into_embeddings(keyed_vect_file,train_file,num=10,max_sent_len=15,embed_dim=50):
    model = getKeyedVect(keyed_vect_file)
    #print(model['hello'])
    #print(len(model.wv.vocab))
    sentences = load_data(train_file,num)
    sentences = sentences[0:2]
    tokens = []
    for s in sentences:
        tempTokens = np.zeros([max_sent_len,embed_dim])
        ind=0
        for t in word_tokenize(s):
            if ind>=max_sent_len:
                break
            if t.lower() in model.wv.vocab:
                tempTokens[ind] = model[t.lower()]
                ind+=1
        #tempTokens = [t.lower() for t in word_tokenize(s) if t.lower() in model.wv.vocab]
        #tempTokens = [model[t] for t in tempTokens]
        #tempTokens = tempTokens[:max_sent_len-5]
        #padding = max_sent_len-len(tempTokens)
        #tempTokens+=np.zeros([padding,embed_dim])
        tokens.append(tempTokens)
    #for sent in tokens:
    #    print('*'*60)
    #    print(sent)
    #    print('len:',len(sent))
    #tokens = np.reshape(tokens,[max_sent_len,50])
    #tokens = [t.lower() for t in tokens if t.lower() in model.wv.vocab]
    return tokens,model



if __name__=='__main__':

    #model = getKeyedVect('../gloveloader/glove_50d_keyed.txt')
    KEYED_VECTOR = '../gloveloader/glove_50d_keyed.txt'
    TRAIN_FILE = './newsfiles/newsfiles/'
    sent_embeddings = turn_sents_into_embeddings(KEYED_VECTOR,TRAIN_FILE)
    
